# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fallback for DeepGEMM's fp8_mqa_logits / fp8_paged_mqa_logits."""

import functools

import torch

from vllm import envs
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.ops.fp8_sm80 import get_e4m3fn_bf16_lut

_IS_SM80 = current_platform.is_cuda() and current_platform.get_device_capability() == (
    8,
    0,
)

# Paged decode: num_warps=4 dominated on A100/SM80 across {2,4,8}; the others
# were 1.5–1.7× slower at (num_heads=32, head_dim=128, block_size=64), so
# narrow the sweep to keep autotune from latching onto a bad pick under noise.
#
# Deliberately NOT carrying the `maxnreg=128` cap the prefill configs below do,
# measured on A100 at H=64 D=128 block_size=64 (benchmark_dsv4_sm80.py
# --kernel indexer-paged). Unconstrained the decode kernel takes 136 regs =
# 3 CTAs/SM; the cap reaches 4 CTAs/SM at 2 B of spill, and that trades one
# regime against another rather than winning outright:
#
#   CTAs (B x blocks)  |  15    64   209   418   3344   13376
#   vs uncapped        | +2.4% +1.7% +1.6% -3.7%  -3.4%  -3.5%
#
# The sign follows wave quantization, not latency hiding: 108 SMs hold 324
# concurrent CTAs at 3/SM and 432 at 4/SM, so the cap only pays past ~1.3
# waves (n_compressed >~ 20.7k at batch 1, i.e. >~83k context, or any batched
# decode), and below that the spill and the lost ILP cost ~2%. At the 107k
# batch-1 point that is 0.43 us on a 21-calls-per-step kernel = 0.009 ms/step
# against a ~10.9 ms step, so the win is not worth the mid-context loss. The
# autotune key carries no context length, so the choice cannot be made per
# call site without dropping @triton.autotune here.
_PAGED_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=ns) for ns in (2, 4)
]

# Prefill: BLOCK_N=128 with num_warps=4 measured fastest at every shape
# swept (M 1..2048, N 2048..131072) -- BN=64 is 1.25-1.40x worse, BN=32 up to
# 2.41x. The autotune key is (num_heads, head_dim), both fixed for a model,
# so a wider sweep cannot adapt per request; it only adds cold-cache JIT.
#
# maxnreg lives in VLLM_INDEXER_LOGITS_MAXNREG (0 = unconstrained, the
# default). It used to be pinned at 128 on the reading that the kernel took
# 132 regs, so 128 bought a 4th CTA/SM "with no spill regression", measured
# 7.7-8.5% faster over (M 8..2048, N 8192..28672) at KV_GROUP=1.
#
# That measurement has expired (rule 47): this kernel now takes 162 regs, so
# 128 is no longer a boundary but a 34-register cut, and it spills 6 B
# unfactored / 14 B with FACTOR_K_SCALE. Re-measured on this tree, the cap is
# neutral-to-worse everywhere and costs 4.0% at the shape serving actually
# runs:
#
#   (M, N, G)      | (8, 8k, 1) (240, 8k, 1) (2048, 8k, 1) (240, 61440, 8)
#   cap 128 vs off |     +6.7%       -0.0%         +1.2%          +4.0%
#
# Sixth confirmation of canon S8: every perturbation that raises resident
# parallelism on this kernel family costs it more than the occupancy buys.
_PREFILL_AUTOTUNE_CONFIGS = [
    triton.Config(
        {"BLOCK_N": 128},
        num_warps=4,
        num_stages=ns,
        maxnreg=envs.VLLM_INDEXER_LOGITS_MAXNREG or None,
    )
    for ns in (2, 4)
]

# KV_GROUP is selected by the wrapper from M and N, not autotuned: grouping 8
# tiles per CTA reuses the q/weights load 8x and measures 3.9% faster at the
# 128k prefill shape (7.11 -> 6.84 ms at M=2048, N=28672; G=16 turns back up),
# but only large SM80 prefill grids select it; smaller grids, short contexts,
# and other fallback devices keep the original ungrouped specialization.
#
# The gate used to be `m >= 512`, tuned at M=2048. Query sharding plus the
# logits budget's sub-chunking made the production call M=240, so the
# specialization the warmup path pre-compiles never once ran in serving
# (rule 47). M alone is also the wrong variable: grouping divides the grid's
# N dimension by KV_GROUP, so what decides it is whether the GROUPED grid
# still fills the machine. Measured at maxnreg unconstrained, N >= 16384:
#
#   M x grid_y (grouped CTAs) |    128    480    512  14400
#   grouped vs ungrouped      | +24.8%  -7.5%  -9.2%  -9.9%
#
# The sign flips at one wave (108 SMs x 3 CTAs/SM at 162 regs = 324), which
# is what `_kv_group_min_ctas` computes -- and it gets all four corners right
# where any single M threshold gets at most three.
_KV_GROUP = 8
_KV_GROUP_MIN_N = 16384
_KV_GROUP_CTAS_PER_SM = 3
_PREFILL_BLOCK_N = 128

# Warmup shape mirrors the chunked-prefill regime (small M, long N) so
# autotune picks a tile sized for real serving rather than a launch-overhead-
# dominated dummy grid.
_PREFILL_WARMUP_M = 8
_PREFILL_WARMUP_N = 8192


# NaN bytes pin to +-480 so a NaN cache entry cannot poison the whole
# logits row through the dot product.
_INDEXER_LUT_NAN_VALUE = 480.0


def _get_e4m3fn_bf16_lut(device: torch.device) -> torch.Tensor:
    return get_e4m3fn_bf16_lut(device, nan_value=_INDEXER_LUT_NAN_VALUE)


@triton.jit
def _decode_e4m3fn_bf16_lut(u, lut_ptr):
    return tl.load(lut_ptr + u.to(tl.uint32))


@triton.autotune(
    configs=_PAGED_AUTOTUNE_CONFIGS,
    key=["num_heads", "head_dim", "block_size"],
)
@triton.jit
def _fp8_paged_mqa_logits_kernel(
    q_ptr,
    kv_fp8_ptr,
    kv_scale_ptr,
    weights_ptr,
    fp8_lut_ptr,
    context_lens_ptr,
    block_tables_ptr,
    logits_ptr,
    stride_q_b,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_kvf_block,
    stride_kvf_s,
    stride_kvf_d,
    stride_kvs_block,
    stride_kvs_s,
    stride_w_t,
    stride_w_h,
    stride_bt_b,
    stride_bt_k,
    stride_l_t,
    stride_l_n,
    next_n: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_IS_BF16: tl.constexpr,
):
    token_id = tl.program_id(0)
    block_rk = tl.program_id(1)

    batch_id = token_id // next_n
    next_n_id = token_id % next_n

    context_len = tl.load(context_lens_ptr + batch_id)
    if block_rk * block_size >= context_len:
        return

    q_offset = context_len - next_n + next_n_id

    # int64: unified-KV-pool layer views carry a large block stride (~1e6
    # elements), so int32 `block_idx * stride` wraps once a batch touches
    # roughly >2k pool blocks (long context or concurrent sequences).
    block_idx = tl.load(
        block_tables_ptr + batch_id * stride_bt_b + block_rk * stride_bt_k
    ).to(tl.int64)

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    mask_h = offs_h < num_heads
    mask_d = offs_d < head_dim
    mask_n = offs_n < block_size

    # Q_IS_BF16: q arrives already decoded. The grid is
    # (B*next_n, num_block_cols), so an in-kernel LUT decode of q re-runs once
    # per KV block though q is identical across `block_rk` -- exactly half of
    # the kernel's 16k per-lane gathers, feeding one tl.dot. The wrapper
    # applies the same 256-entry table once instead. k keeps the LUT: it
    # differs per block, and reading it as fp8 is 6.8x worse (canon S1).
    q_base = q_ptr + batch_id * stride_q_b + next_n_id * stride_q_n
    q_offs = offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d
    q_mask = mask_h[:, None] & mask_d[None, :]
    if Q_IS_BF16:
        q = tl.load(q_base + q_offs, mask=q_mask, other=0.0)
    else:
        q = _decode_e4m3fn_bf16_lut(
            tl.load(q_base + q_offs, mask=q_mask, other=0), fp8_lut_ptr
        )

    kvf_base = kv_fp8_ptr + block_idx * stride_kvf_block
    k_byte = tl.load(
        kvf_base + offs_n[:, None] * stride_kvf_s + offs_d[None, :] * stride_kvf_d,
        mask=mask_n[:, None] & mask_d[None, :],
        other=0,
    )
    kvs_base = kv_scale_ptr + block_idx * stride_kvs_block
    k_scale = tl.load(
        kvs_base + offs_n * stride_kvs_s,
        mask=mask_n,
        other=0.0,
    )
    k = _decode_e4m3fn_bf16_lut(k_byte, fp8_lut_ptr)
    # Scale in fp32 after the dot to avoid an extra bf16 round-trip on K.
    s = tl.dot(q, tl.trans(k)) * k_scale[None, :]

    w = tl.load(
        weights_ptr + token_id * stride_w_t + offs_h * stride_w_h,
        mask=mask_h,
        other=0.0,
    )
    s = tl.where(s > 0, s, 0.0) * w[:, None]
    out = tl.sum(s, axis=0)

    k_offset = block_rk * block_size + offs_n
    # Store mask below covers mask_n and the context bound; -inf only has to
    # mask the causal tail inside the written region.
    out = tl.where(k_offset <= q_offset, out, float("-inf"))

    tl.store(
        logits_ptr + token_id * stride_l_t + k_offset * stride_l_n,
        out,
        mask=mask_n & (k_offset < context_len),
    )


def fp8_paged_mqa_logits_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Triton implementation of DeepGEMM's fp8_paged_mqa_logits.

    Args:
        q:             [B, next_n, H, D] fp8_e4m3fn
        kv_cache:      [num_blocks, block_size, 1, D+4] uint8 (FP8 + fp32 scale)
        weights:       [B*next_n, H] float32
        context_lens:  [B] int32
        block_tables:  [B, max_blocks] int32
        max_model_len: output width. Caller passes the active batch max so
            the logits buffer and grid stay tight.
        clean_logits: when False, skip the -inf pre-fill of the output
            (indexer top-k reads only `[:context_len]` per row).
    Returns:
        logits:        [B*next_n, max_model_len] float32
    """
    B, next_n, num_heads, head_dim = q.shape
    _, block_size, one, d_plus_4 = kv_cache.shape
    assert one == 1
    assert d_plus_4 == head_dim + 4

    # Cache layout from `indexer_k_quant_and_cache`: per block, FP8 K bytes
    # (block_size * head_dim) followed by fp32 scales (block_size * 4). The
    # `[NB, block_size, 1, head_dim+4]` shape is a stride trick; re-slice flat.
    # Kernel decodes FP8 from uint8 via LUT (SM80 Triton can't load fp8e4nv).
    num_blocks = kv_cache.shape[0]
    kv_flat = kv_cache.view(num_blocks, -1)
    k_end = block_size * head_dim
    kv_byte = kv_flat[:, :k_end].as_strided(
        (num_blocks, block_size, head_dim),
        (kv_flat.stride(0), head_dim, 1),
    )
    kv_scale = kv_flat[:, k_end:].view(torch.float32)
    q_byte = q.view(torch.uint8)
    q_is_bf16 = envs.VLLM_INDEXER_PAGED_Q_BF16

    if clean_logits:
        logits = torch.full(
            (B * next_n, max_model_len),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        logits = torch.empty(
            (B * next_n, max_model_len), dtype=torch.float32, device=q.device
        )

    BLOCK_H = max(16, triton.next_power_of_2(num_heads))
    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_N = triton.next_power_of_2(block_size)

    fp8_lut = _get_e4m3fn_bf16_lut(q.device)
    if q_is_bf16:
        # The same table the kernel would apply, applied once per call instead
        # of once per (query, KV block). `lut[byte]` produces exactly what
        # `_decode_e4m3fn_bf16_lut` produces -- NaN pin at +-480 included --
        # so the operands reaching tl.dot are bit-identical.
        q_in = fp8_lut.index_select(0, q_byte.reshape(-1).to(torch.int32)).view(
            q_byte.shape
        )
    else:
        q_in = q_byte
    # The block table is allocated at full max_model_len width and only
    # narrowed along dim 0, so sizing the grid by it launches a CTA per
    # possible block rather than per live one; each surplus CTA loads
    # context_lens and returns. max_model_len here is the active batch max.
    num_block_cols = min(block_tables.shape[1], triton.cdiv(max_model_len, block_size))
    grid = (B * next_n, num_block_cols)
    _fp8_paged_mqa_logits_kernel[grid](
        q_in,
        kv_byte,
        kv_scale,
        weights,
        fp8_lut,
        context_lens,
        block_tables,
        logits,
        q_in.stride(0),
        q_in.stride(1),
        q_in.stride(2),
        q_in.stride(3),
        kv_byte.stride(0),
        kv_byte.stride(1),
        kv_byte.stride(2),
        kv_scale.stride(0),
        kv_scale.stride(1),
        weights.stride(0),
        weights.stride(1),
        block_tables.stride(0),
        block_tables.stride(1),
        logits.stride(0),
        logits.stride(1),
        next_n=next_n,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
        BLOCK_N=BLOCK_N,
        Q_IS_BF16=q_is_bf16,
    )
    return logits


@triton.autotune(
    configs=_PREFILL_AUTOTUNE_CONFIGS,
    # Per-program work is N-independent; key on (heads, dim) only so chunked
    # prefill with varying N doesn't re-tune on every new chunk size.
    key=["num_heads", "head_dim"],
)
@triton.jit
def _fp8_mqa_logits_kernel(
    q_ptr,
    k_ptr,
    k_scale_ptr,
    weights_ptr,
    ks_ptr,
    ke_ptr,
    logits_ptr,
    stride_q_m,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_d,
    stride_w_m,
    stride_w_h,
    stride_l_m,
    stride_l_n,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    N,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_GROUP: tl.constexpr,
    FACTOR_K_SCALE: tl.constexpr,
):
    # bf16 q/k inputs: the wrapper pre-decodes FP8 → bf16. At compute-bound
    # prefill this is ~2× the in-kernel LUT (LUT lookups contend with the
    # matmul for ALU/regs). Paged-decode keeps the LUT path.
    #
    # Each CTA owns KV_GROUP consecutive BLOCK_N tiles for one query row, so
    # the q tile and weights load once per KV_GROUP tiles instead of once per
    # tile. gridY is huge at long context (N/BLOCK_N), so dividing it by
    # KV_GROUP costs no occupancy.
    m = tl.program_id(0)
    group = tl.program_id(1)

    group_start = group * (KV_GROUP * BLOCK_N)
    ks = tl.load(ks_ptr + m)
    ke = tl.load(ke_ptr + m)

    # Early-exit when this row's `[ks, ke)` range doesn't overlap the whole
    # group. Chunked prefill produces many such all-masked spans per row.
    if (group_start >= ke) | (group_start + KV_GROUP * BLOCK_N <= ks):
        # When `clean_logits=False` the caller skipped the -inf pre-fill, so
        # write -inf here for the early-exit span.
        offs_span = group_start + tl.arange(0, BLOCK_N)
        for _ in tl.static_range(KV_GROUP):
            tl.store(
                logits_ptr + m * stride_l_m + offs_span * stride_l_n,
                tl.full([BLOCK_N], float("-inf"), dtype=tl.float32),
                mask=offs_span < N,
            )
            offs_span += BLOCK_N
        return

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    mask_h = offs_h < num_heads
    mask_d = offs_d < head_dim

    q = tl.load(
        q_ptr
        + m * stride_q_m
        + offs_h[:, None] * stride_q_h
        + offs_d[None, :] * stride_q_d,
        mask=mask_h[:, None] & mask_d[None, :],
        other=0.0,
    )
    w = tl.load(
        weights_ptr + m * stride_w_m + offs_h * stride_w_h,
        mask=mask_h,
        other=0.0,
    )

    # The per-tile liveness branch is kept even though the tl.where below
    # would make a branch-free loop correct: without it Triton software-
    # pipelines the k tiles across the unrolled iterations, and the extra
    # live tiles blow the register budget (5x slower under maxnreg=128,
    # measured). At KV_GROUP=1 this body is exactly the ungrouped kernel.
    for g in tl.static_range(KV_GROUP):
        n_start = group_start + g * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        if (n_start >= ke) | (n_start + BLOCK_N <= ks):
            tl.store(
                logits_ptr + m * stride_l_m + offs_n * stride_l_n,
                tl.full([BLOCK_N], float("-inf"), dtype=tl.float32),
                mask=mask_n,
            )
        else:
            k = tl.load(
                k_ptr + offs_n[:, None] * stride_k_n + offs_d[None, :] * stride_k_d,
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            )
            k_scale = tl.load(k_scale_ptr + offs_n, mask=mask_n, other=0.0)
            s = tl.dot(q, tl.trans(k))

            # relu is positively homogeneous and k_scale is a quantization
            # magnitude (>= 0), so the scale factors straight out of the head
            # sum: BLOCK_H x BLOCK_N broadcast multiplies become BLOCK_N. The
            # relu's threshold is at zero and a non-negative scale cannot move
            # a sign, so the top-k's SELECTED set is bit-identical; only the
            # sum's rounding order changes (one scaling instead of BLOCK_H).
            if FACTOR_K_SCALE:
                s = tl.where(s > 0, s, 0.0) * w[:, None]
                out = tl.sum(s, axis=0) * k_scale
            else:
                s = s * k_scale[None, :]
                s = tl.where(s > 0, s, 0.0) * w[:, None]
                out = tl.sum(s, axis=0)

            # Store mask covers mask_n; -inf masks [ks, ke) only.
            out = tl.where((offs_n >= ks) & (offs_n < ke), out, float("-inf"))

            tl.store(
                logits_ptr + m * stride_l_m + offs_n * stride_l_n,
                out,
                mask=mask_n,
            )


@functools.cache
def _kv_group_min_ctas(device_index: int) -> int:
    return num_compute_units(device_index) * _KV_GROUP_CTAS_PER_SM


def _select_prefill_kv_group(m: int, n: int, device_index: int = 0) -> int:
    if not _IS_SM80 or n < _KV_GROUP_MIN_N:
        return 1
    min_m = envs.VLLM_INDEXER_LOGITS_KV_GROUP_MIN_M
    if min_m:
        return _KV_GROUP if m >= min_m else 1
    grouped_ctas = m * triton.cdiv(n, _PREFILL_BLOCK_N * _KV_GROUP)
    if grouped_ctas >= _kv_group_min_ctas(device_index):
        return _KV_GROUP
    return 1


def fp8_mqa_logits_triton(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Triton implementation of DeepGEMM's fp8_mqa_logits.

    Args:
        q:            [M, H, D] fp8_e4m3fn
        kv:           (k_fp8 [N, D], k_scales [N]) — fp8_e4m3fn, float32
        weights:      [M, H] float32
        cu_seqlen_ks: [M] int32
        cu_seqlen_ke: [M] int32
        clean_logits: when False, skip the -inf pre-fill of the output
            (indexer top-k reads only `[ks, ke)` per row). Matches DeepGEMM.
    Returns:
        logits:       [M, N] float32
    """
    return _fp8_mqa_logits_triton_impl(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        _select_prefill_kv_group(q.shape[0], kv[0].shape[0]),
    )


def _fp8_mqa_logits_triton_impl(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    kv_group: int,
) -> torch.Tensor:
    k_fp8, k_scales = kv
    k_scales = k_scales.reshape(-1)

    M, num_heads, head_dim = q.shape
    N = k_fp8.shape[0]

    # The grid covers every (m, n_block) and each tile stores its full row
    # span, so a -inf pre-fill would be entirely overwritten; `clean_logits`
    # is accepted for DeepGEMM signature parity only.
    logits = torch.empty((M, N), dtype=torch.float32, device=q.device)

    BLOCK_H = max(16, triton.next_power_of_2(num_heads))
    BLOCK_D = triton.next_power_of_2(head_dim)

    # Pre-decode FP8 → bf16; the kernel runs a straight `tl.dot`.
    q_bf16 = q.to(torch.bfloat16)
    k_bf16 = k_fp8.to(torch.bfloat16)

    # Grid depends on the autotuned BLOCK_N and the M/N-selected KV_GROUP.
    grid = lambda meta: (  # noqa: E731
        M,
        triton.cdiv(N, meta["BLOCK_N"] * meta["KV_GROUP"]),
    )
    _fp8_mqa_logits_kernel[grid](
        q_bf16,
        k_bf16,
        k_scales,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        logits,
        q_bf16.stride(0),
        q_bf16.stride(1),
        q_bf16.stride(2),
        k_bf16.stride(0),
        k_bf16.stride(1),
        weights.stride(0),
        weights.stride(1),
        logits.stride(0),
        logits.stride(1),
        num_heads=num_heads,
        head_dim=head_dim,
        N=N,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
        KV_GROUP=kv_group,
        FACTOR_K_SCALE=envs.VLLM_INDEXER_LOGITS_FACTOR_K_SCALE,
    )
    return logits


def warmup_fp8_mqa_logits_triton(
    num_heads: int,
    head_dim: int,
    device: torch.device,
) -> None:
    """Prime the prefill `@triton.autotune` cache so first-call doesn't pay
    the inline sweep (~5–8 s on A100 SM80). KV_GROUP is an M/N-selected
    constexpr, so on SM80 warm both specializations at the small warmup shape:
    the short-N shape runs the tuner sweep, then the forced grouped launch just
    JIT-compiles against the cached best config. Other devices need only the
    ungrouped specialization."""
    _warmup_fp8_mqa_logits_shape(
        num_heads, head_dim, _PREFILL_WARMUP_M, _PREFILL_WARMUP_N, device
    )
    if _IS_SM80:
        _warmup_fp8_mqa_logits_shape(
            num_heads,
            head_dim,
            _PREFILL_WARMUP_M,
            _KV_GROUP_MIN_N,
            device,
            kv_group_override=_KV_GROUP,
        )


def _warmup_fp8_mqa_logits_shape(
    num_heads: int,
    head_dim: int,
    m: int,
    n: int,
    device: torch.device,
    kv_group_override: int | None = None,
) -> None:
    q = torch.empty(m, num_heads, head_dim, dtype=torch.float8_e4m3fn, device=device)
    k = torch.empty(n, head_dim, dtype=torch.float8_e4m3fn, device=device)
    scales = torch.zeros(n, dtype=torch.float32, device=device)
    weights = torch.zeros(m, num_heads, dtype=torch.float32, device=device)
    ks = torch.zeros(m, dtype=torch.int32, device=device)
    ke = torch.full((m,), n, dtype=torch.int32, device=device)
    kv_group = (
        _select_prefill_kv_group(m, n)
        if kv_group_override is None
        else kv_group_override
    )
    _fp8_mqa_logits_triton_impl(q, (k, scales), weights, ks, ke, kv_group)


def warmup_fp8_paged_mqa_logits_triton(
    num_heads: int,
    head_dim: int,
    block_size: int,
    device: torch.device,
) -> None:
    """Prime the paged-decode `@triton.autotune` cache for the indexer's
    logits kernel (see `warmup_fp8_mqa_logits_triton` for rationale).
    """
    num_blocks = 2
    q = torch.empty(1, 1, num_heads, head_dim, dtype=torch.float8_e4m3fn, device=device)
    kv_cache = torch.zeros(
        num_blocks, block_size, 1, head_dim + 4, dtype=torch.uint8, device=device
    )
    weights = torch.zeros(1, num_heads, dtype=torch.float32, device=device)
    context_lens = torch.tensor([block_size], dtype=torch.int32, device=device)
    block_tables = torch.zeros(1, 1, dtype=torch.int32, device=device)
    fp8_paged_mqa_logits_triton(
        q, kv_cache, weights, context_lens, block_tables, max_model_len=block_size
    )


_MXFP4_GROUP_SIZE = 32
_MXFP4_PACKED_GROUP_SIZE = _MXFP4_GROUP_SIZE // 2


def supports_mxfp4_indexer_cache() -> bool:
    """Return whether DeepSeek V4 has an MXFP4 indexer path for this GPU."""
    return current_platform.is_cuda() and (
        current_platform.is_device_capability_family(100)
        or current_platform.get_device_capability() == (8, 6)
    )


@triton.jit
def _decode_e2m1_nibble(code):
    magnitude_code = code & 0x7
    magnitude = tl.where(magnitude_code == 0, 0.0, 0.5)
    magnitude = tl.where(magnitude_code == 2, 1.0, magnitude)
    magnitude = tl.where(magnitude_code == 3, 1.5, magnitude)
    magnitude = tl.where(magnitude_code == 4, 2.0, magnitude)
    magnitude = tl.where(magnitude_code == 5, 3.0, magnitude)
    magnitude = tl.where(magnitude_code == 6, 4.0, magnitude)
    magnitude = tl.where(magnitude_code == 7, 6.0, magnitude)
    return tl.where((code & 0x8) != 0, -magnitude, magnitude)


@triton.jit
def _decode_mxfp4_bytes(packed):
    low = _decode_e2m1_nibble(packed & 0xF)
    high = _decode_e2m1_nibble(packed >> 4)
    return tl.interleave(low, high).to(tl.bfloat16)


@triton.jit
def _decode_ue8m0_scale(encoded):
    return tl.exp2(encoded.to(tl.float32) - 127.0)


@triton.autotune(
    configs=_PREFILL_AUTOTUNE_CONFIGS,
    key=["num_heads", "head_dim"],
)
@triton.jit
def _mxfp4_mqa_logits_kernel(
    q_ptr,
    q_scale_ptr,
    k_ptr,
    k_scale_ptr,
    weights_ptr,
    ks_ptr,
    ke_ptr,
    logits_ptr,
    stride_q_m,
    stride_q_h,
    stride_q_d,
    stride_qs_m,
    stride_qs_h,
    stride_qs_g,
    stride_k_n,
    stride_k_d,
    stride_ks_n,
    stride_ks_g,
    stride_w_m,
    stride_w_h,
    stride_l_m,
    stride_l_n,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    N,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D_PACKED: tl.constexpr,
):
    tl.static_assert(head_dim % 32 == 0)
    m = tl.program_id(0)
    n_start = tl.program_id(1) * BLOCK_N
    offs_h = tl.arange(0, BLOCK_H)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_packed = tl.arange(0, BLOCK_D_PACKED)
    mask_h = offs_h < num_heads
    mask_n = offs_n < N
    ks = tl.load(ks_ptr + m)
    ke = tl.load(ke_ptr + m)
    scores = tl.zeros((BLOCK_H, BLOCK_N), tl.float32)

    for group in tl.static_range(head_dim // 32):
        q_packed = tl.load(
            q_ptr
            + m * stride_q_m
            + offs_h[:, None] * stride_q_h
            + (group * 16 + offs_packed[None, :]) * stride_q_d,
            mask=mask_h[:, None],
            other=0,
        )
        k_packed = tl.load(
            k_ptr
            + offs_n[:, None] * stride_k_n
            + (group * 16 + offs_packed[None, :]) * stride_k_d,
            mask=mask_n[:, None],
            other=0,
        )
        q_group = _decode_mxfp4_bytes(q_packed)
        k_group = _decode_mxfp4_bytes(k_packed)
        q_scale = _decode_ue8m0_scale(
            tl.load(
                q_scale_ptr
                + m * stride_qs_m
                + offs_h * stride_qs_h
                + group * stride_qs_g,
                mask=mask_h,
                other=0,
            )
        )
        k_scale = _decode_ue8m0_scale(
            tl.load(
                k_scale_ptr + offs_n * stride_ks_n + group * stride_ks_g,
                mask=mask_n,
                other=0,
            )
        )
        scores += (
            tl.dot(q_group, tl.trans(k_group)) * q_scale[:, None] * k_scale[None, :]
        )

    weights = tl.load(
        weights_ptr + m * stride_w_m + offs_h * stride_w_h,
        mask=mask_h,
        other=0.0,
    )
    logits = tl.sum(tl.maximum(scores, 0.0) * weights[:, None], axis=0)
    logits = tl.where((offs_n >= ks) & (offs_n < ke), logits, float("-inf"))
    tl.store(
        logits_ptr + m * stride_l_m + offs_n * stride_l_n,
        logits,
        mask=mask_n,
    )


def _mxfp4_scale_bytes(
    scales: torch.Tensor, leading_shape: tuple[int, ...]
) -> torch.Tensor:
    if scales.dtype == torch.int32:
        scales = scales.contiguous().view(torch.uint8)
    assert scales.dtype == torch.uint8
    return scales.reshape(*leading_shape, -1)


def mxfp4_mqa_logits_triton(
    q: tuple[torch.Tensor, torch.Tensor],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Compute sparse-indexer prefill logits from E2M1/UE8M0 rows on SM8x."""
    del clean_logits
    q_values, q_scales = q
    k_values, k_scales = kv
    q_values = q_values.view(torch.uint8)
    k_values = k_values.view(torch.uint8)
    M, num_heads, packed_head_dim = q_values.shape
    N = k_values.shape[0]
    head_dim = packed_head_dim * 2
    assert head_dim % _MXFP4_GROUP_SIZE == 0
    q_scale_bytes = _mxfp4_scale_bytes(q_scales, (M, num_heads))
    k_scale_bytes = _mxfp4_scale_bytes(k_scales, (N,))
    assert q_scale_bytes.shape[-1] == head_dim // _MXFP4_GROUP_SIZE
    assert k_scale_bytes.shape[-1] == head_dim // _MXFP4_GROUP_SIZE

    logits = torch.empty((M, N), dtype=torch.float32, device=q_values.device)
    block_h = max(16, triton.next_power_of_2(num_heads))
    grid = lambda meta: (M, triton.cdiv(N, meta["BLOCK_N"]))  # noqa: E731
    _mxfp4_mqa_logits_kernel[grid](
        q_values,
        q_scale_bytes,
        k_values,
        k_scale_bytes,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        logits,
        q_values.stride(0),
        q_values.stride(1),
        q_values.stride(2),
        q_scale_bytes.stride(0),
        q_scale_bytes.stride(1),
        q_scale_bytes.stride(2),
        k_values.stride(0),
        k_values.stride(1),
        k_scale_bytes.stride(0),
        k_scale_bytes.stride(1),
        weights.stride(0),
        weights.stride(1),
        logits.stride(0),
        logits.stride(1),
        num_heads=num_heads,
        head_dim=head_dim,
        N=N,
        BLOCK_H=block_h,
        BLOCK_D_PACKED=_MXFP4_PACKED_GROUP_SIZE,
    )
    return logits


@triton.autotune(
    configs=_PAGED_AUTOTUNE_CONFIGS,
    key=["num_heads", "head_dim", "block_size"],
)
@triton.jit
def _mxfp4_paged_mqa_logits_kernel(
    q_ptr,
    q_scale_ptr,
    kv_cache_ptr,
    weights_ptr,
    context_lens_ptr,
    block_tables_ptr,
    logits_ptr,
    stride_q_b,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_qs_b,
    stride_qs_n,
    stride_qs_h,
    stride_qs_g,
    stride_kv_block,
    stride_w_t,
    stride_w_h,
    stride_cl_b,
    stride_cl_n,
    stride_bt_b,
    stride_bt_k,
    stride_l_t,
    stride_l_n,
    next_n: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D_PACKED: tl.constexpr,
):
    tl.static_assert(head_dim % 32 == 0)
    token_id = tl.program_id(0)
    block_rk = tl.program_id(1)
    batch_id = token_id // next_n
    query_id = token_id % next_n
    context_len = tl.load(
        context_lens_ptr + batch_id * stride_cl_b + query_id * stride_cl_n
    )
    if block_rk * block_size >= context_len:
        return

    block_idx = tl.load(
        block_tables_ptr + batch_id * stride_bt_b + block_rk * stride_bt_k
    ).to(tl.int64)
    offs_h = tl.arange(0, BLOCK_H)
    offs_n = tl.arange(0, BLOCK_N)
    offs_packed = tl.arange(0, BLOCK_D_PACKED)
    mask_h = offs_h < num_heads
    mask_n = (offs_n < block_size) & (block_rk * block_size + offs_n < context_len)
    scores = tl.zeros((BLOCK_H, BLOCK_N), tl.float32)
    packed_head_dim: tl.constexpr = head_dim // 2
    num_scale_groups: tl.constexpr = head_dim // 32
    cache_block_base = kv_cache_ptr + block_idx * stride_kv_block

    for group in tl.static_range(num_scale_groups):
        q_packed = tl.load(
            q_ptr
            + batch_id * stride_q_b
            + query_id * stride_q_n
            + offs_h[:, None] * stride_q_h
            + (group * 16 + offs_packed[None, :]) * stride_q_d,
            mask=mask_h[:, None],
            other=0,
        )
        k_packed = tl.load(
            cache_block_base
            + offs_n[:, None] * packed_head_dim
            + group * 16
            + offs_packed[None, :],
            mask=mask_n[:, None],
            other=0,
        )
        q_group = _decode_mxfp4_bytes(q_packed)
        k_group = _decode_mxfp4_bytes(k_packed)
        q_scale = _decode_ue8m0_scale(
            tl.load(
                q_scale_ptr
                + batch_id * stride_qs_b
                + query_id * stride_qs_n
                + offs_h * stride_qs_h
                + group * stride_qs_g,
                mask=mask_h,
                other=0,
            )
        )
        k_scale = _decode_ue8m0_scale(
            tl.load(
                cache_block_base
                + block_size * packed_head_dim
                + offs_n * num_scale_groups
                + group,
                mask=mask_n,
                other=0,
            )
        )
        scores += (
            tl.dot(q_group, tl.trans(k_group)) * q_scale[:, None] * k_scale[None, :]
        )

    weights = tl.load(
        weights_ptr + token_id * stride_w_t + offs_h * stride_w_h,
        mask=mask_h,
        other=0.0,
    )
    output = tl.sum(tl.maximum(scores, 0.0) * weights[:, None], axis=0)
    key_offsets = block_rk * block_size + offs_n
    output = tl.where(key_offsets < context_len, output, float("-inf"))
    tl.store(
        logits_ptr + token_id * stride_l_t + key_offsets * stride_l_n,
        output,
        mask=mask_n,
    )


def _mxfp4_context_lens(
    context_lens: torch.Tensor, batch_size: int, next_n: int
) -> torch.Tensor:
    if context_lens.ndim == 2:
        assert context_lens.shape == (batch_size, next_n)
        return context_lens.contiguous()
    assert context_lens.shape == (batch_size,)
    final_lens = context_lens[:, None]
    offsets = torch.arange(
        next_n - 1,
        -1,
        -1,
        dtype=context_lens.dtype,
        device=context_lens.device,
    )
    return final_lens - offsets[None, :]


def mxfp4_paged_mqa_logits_triton(
    q: tuple[torch.Tensor, torch.Tensor],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Compute sparse-indexer paged decode logits from segregated MXFP4 rows."""
    q_values, q_scales = q
    q_values = q_values.view(torch.uint8)
    batch_size, next_n, num_heads, packed_head_dim = q_values.shape
    head_dim = packed_head_dim * 2
    assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
    _, block_size, row_bytes = kv_cache.shape
    assert row_bytes == packed_head_dim + head_dim // _MXFP4_GROUP_SIZE
    q_scale_bytes = _mxfp4_scale_bytes(q_scales, (batch_size, next_n, num_heads))
    normalized_lens = _mxfp4_context_lens(context_lens, batch_size, next_n)
    if clean_logits:
        logits = torch.full(
            (batch_size * next_n, max_model_len),
            float("-inf"),
            dtype=torch.float32,
            device=q_values.device,
        )
    else:
        logits = torch.empty(
            (batch_size * next_n, max_model_len),
            dtype=torch.float32,
            device=q_values.device,
        )

    block_h = max(16, triton.next_power_of_2(num_heads))
    block_n = triton.next_power_of_2(block_size)
    num_block_cols = min(block_tables.shape[1], triton.cdiv(max_model_len, block_size))
    _mxfp4_paged_mqa_logits_kernel[(batch_size * next_n, num_block_cols)](
        q_values,
        q_scale_bytes,
        kv_cache,
        weights,
        normalized_lens,
        block_tables,
        logits,
        q_values.stride(0),
        q_values.stride(1),
        q_values.stride(2),
        q_values.stride(3),
        q_scale_bytes.stride(0),
        q_scale_bytes.stride(1),
        q_scale_bytes.stride(2),
        q_scale_bytes.stride(3),
        kv_cache.stride(0),
        weights.stride(0),
        weights.stride(1),
        normalized_lens.stride(0),
        normalized_lens.stride(1),
        block_tables.stride(0),
        block_tables.stride(1),
        logits.stride(0),
        logits.stride(1),
        next_n=next_n,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        BLOCK_H=block_h,
        BLOCK_N=block_n,
        BLOCK_D_PACKED=_MXFP4_PACKED_GROUP_SIZE,
    )
    return logits


def warmup_mxfp4_mqa_logits_triton(
    num_heads: int, head_dim: int, device: torch.device
) -> None:
    """Compile and tune the SM8x MXFP4 sparse-indexer prefill kernel."""
    m, n = _PREFILL_WARMUP_M, _PREFILL_WARMUP_N
    q = torch.zeros(m, num_heads, head_dim // 2, dtype=torch.uint8, device=device)
    q_scales = torch.full(
        (m, num_heads, head_dim // _MXFP4_GROUP_SIZE),
        127,
        dtype=torch.uint8,
        device=device,
    )
    k = torch.zeros(n, head_dim // 2, dtype=torch.uint8, device=device)
    k_scales = torch.full(
        (n, head_dim // _MXFP4_GROUP_SIZE), 127, dtype=torch.uint8, device=device
    )
    weights = torch.zeros(m, num_heads, dtype=torch.float32, device=device)
    ks = torch.zeros(m, dtype=torch.int32, device=device)
    ke = torch.full((m,), n, dtype=torch.int32, device=device)
    mxfp4_mqa_logits_triton((q, q_scales), (k, k_scales), weights, ks, ke)


def warmup_mxfp4_paged_mqa_logits_triton(
    num_heads: int,
    head_dim: int,
    block_size: int,
    device: torch.device,
) -> None:
    """Compile and tune the SM8x MXFP4 sparse-indexer decode kernel."""
    num_blocks = 2
    q = torch.zeros(1, 1, num_heads, head_dim // 2, dtype=torch.uint8, device=device)
    q_scales = torch.full(
        (1, 1, num_heads, head_dim // _MXFP4_GROUP_SIZE),
        127,
        dtype=torch.uint8,
        device=device,
    )
    kv_cache = torch.zeros(
        num_blocks,
        block_size,
        head_dim // 2 + head_dim // _MXFP4_GROUP_SIZE,
        dtype=torch.uint8,
        device=device,
    )
    weights = torch.zeros(1, num_heads, dtype=torch.float32, device=device)
    context_lens = torch.tensor([block_size], dtype=torch.int32, device=device)
    block_tables = torch.zeros(1, 1, dtype=torch.int32, device=device)
    mxfp4_paged_mqa_logits_triton(
        (q, q_scales),
        kv_cache,
        weights,
        context_lens,
        block_tables,
        max_model_len=block_size,
    )
