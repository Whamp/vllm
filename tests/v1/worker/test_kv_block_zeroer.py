# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheLayout,
    SlidingWindowSpec,
)
from vllm.v1.worker.utils import (
    AttentionGroup,
    KVBlockZeroer,
    _zero_kv_blocks_kernel,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "spec",
    [
        SlidingWindowSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.uint8,
            sliding_window=4,
        ),
        ChunkedLocalAttentionSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.uint8,
            attention_chunk_size=4,
        ),
    ],
    ids=["sliding-window", "chunked-local"],
)
def test_attention_blocks_are_zeroed(spec):
    device = torch.device("cuda")
    storage = torch.ones((4, 1, 2, 2), dtype=torch.uint8, device=device)
    layer_name = "draft.self_attn"
    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=[AttentionGroup(None, [layer_name], spec, 0)],
        kernel_block_sizes=[2],
        static_forward_context={
            layer_name: SimpleNamespace(kv_cache=storage),
        },
    )

    zeroer.zero_block_ids([[1]])
    torch.accelerator.synchronize()

    expected = torch.ones_like(storage)
    expected[1] = 0
    assert torch.equal(storage, expected)


def _zeroer_for(
    storages: list[torch.Tensor],
    *,
    strides: list[int] | None = None,
    extents: list[int] | None = None,
    ratios: list[int] | None = None,
    group: int = 0,
) -> KVBlockZeroer:
    """Minimal zeroer state for contiguous [num_blocks, page] test storages.

    Built directly so tests can focus on kernel behavior without constructing
    model attention groups. Defaults describe the dense case: addressing
    stride == logical extent, no virtual block splitting.
    """
    device = storages[0].device
    pages = [s.shape[-1] for s in storages]
    meta = KVBlockZeroer.build_meta(
        [s.data_ptr() for s in storages],
        strides or pages,
        extents or pages,
        ratios or [1] * len(storages),
        device,
    )
    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._group_meta = {} if meta is None else {group: meta}
    return zeroer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_block_ids_are_not_overwritten_while_copy_is_in_flight():
    device = torch.device("cuda")
    num_blocks = 4
    page_size_el = 4
    storage = torch.ones((num_blocks, page_size_el), dtype=torch.int32, device=device)
    zeroer = _zeroer_for([storage])

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Keep the first nonblocking H2D copy pending while the host submits the
        # second call. Each call must stage from its own pinned source so the
        # first copy is not corrupted before it runs.
        torch.cuda._sleep(10_000_000)
        zeroer.zero_block_ids([[1]])
        zeroer.zero_block_ids([[2]])
    stream.synchronize()

    assert torch.all(storage[0] == 1)
    assert torch.all(storage[1] == 0)
    assert torch.all(storage[2] == 0)
    assert torch.all(storage[3] == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_non_uniform_page_sizes():
    """Two segments with different page sizes (e.g. MLA + DSA indexer)."""
    device = torch.device("cuda")
    num_blocks = 4
    storage_a = torch.ones((num_blocks, 10496), dtype=torch.int32, device=device)
    storage_b = torch.ones((num_blocks, 2112), dtype=torch.int32, device=device)
    zeroer = _zeroer_for([storage_a, storage_b])

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        zeroer.zero_block_ids([[1, 2]])
    stream.synchronize()

    for storage in (storage_a, storage_b):
        assert torch.all(storage[0] == 1)
        assert torch.all(storage[1] == 0)
        assert torch.all(storage[2] == 0)
        assert torch.all(storage[3] == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_interleaved_layer_views_zero_only_their_own_bytes():
    """THE STRIPE REGRESSION GUARD (#50576).

    DSv4's per-layer caches can be interleaved views over one pool: the block
    stride spans every layer, so a kernel that derives the zeroed EXTENT from
    the stride wipes a whole pool stripe -- including the head of neighboring
    live blocks. Model the pool as [num_blocks, num_layers, page]: each layer
    view has stride num_layers * page but owns only page elements per block.
    """
    device = torch.device("cuda")
    num_blocks, num_layers, page = 4, 3, 64
    pool = torch.ones((num_blocks, num_layers, page), dtype=torch.int32, device=device)
    stride = num_layers * page
    # One segment per layer view, addressed from the layer's first block.
    meta = KVBlockZeroer.build_meta(
        [pool.data_ptr() + layer * page * 4 for layer in (0, 2)],
        [stride, stride],
        [page, page],
        [1, 1],
        device,
    )
    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._group_meta = {0: meta}

    zeroer.zero_block_ids([[1]])
    torch.accelerator.synchronize()

    assert torch.all(pool[0] == 1) and torch.all(pool[2:] == 1)
    assert torch.all(pool[1, 0] == 0), "layer 0's block 1 must be zeroed"
    assert torch.all(pool[1, 2] == 0), "layer 2's block 1 must be zeroed"
    # The stripe bug would have wiped this live neighboring layer too.
    assert torch.all(pool[1, 1] == 1), "layer 1 is NOT registered and must survive"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_block_ids_are_group_scoped():
    """THE CROSS-GROUP REGRESSION GUARD (#50576).

    Block ids are only meaningful within their own kv-cache group: with
    virtual block splitting the same id maps to different pages in groups
    with different geometry. Zeroing group 0's new block must not touch
    group 1's identically-numbered live block.
    """
    device = torch.device("cuda")
    storage_a = torch.ones((4, 128), dtype=torch.int32, device=device)
    storage_b = torch.ones((4, 96), dtype=torch.int32, device=device)

    meta_a = KVBlockZeroer.build_meta([storage_a.data_ptr()], [128], [128], [1], device)
    meta_b = KVBlockZeroer.build_meta([storage_b.data_ptr()], [96], [96], [1], device)
    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._group_meta = {0: meta_a, 1: meta_b}

    zeroer.zero_block_ids([[1], [3]])
    torch.accelerator.synchronize()

    assert torch.all(storage_a[1] == 0) and torch.all(storage_b[3] == 0)
    # The flat-list bug applied every id to every group.
    assert torch.all(storage_a[3] == 1), "group 0 must not zero group 1's id"
    assert torch.all(storage_b[1] == 1), "group 1 must not zero group 0's id"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_virtual_block_split_zeroes_every_sub_block():
    """ratio > 1: one logical block spans ratio kernel blocks, each at its own
    stride offset, and each zeroed only over its logical extent."""
    device = torch.device("cuda")
    num_kernel_blocks, page = 8, 48
    stride = 2 * page  # interleaved with a neighbor view that must survive
    pool = torch.ones((num_kernel_blocks, 2, page), dtype=torch.int32, device=device)
    meta = KVBlockZeroer.build_meta([pool.data_ptr()], [stride], [page], [2], device)
    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._group_meta = {0: meta}

    # Logical block 1 = kernel blocks 2 and 3.
    zeroer.zero_block_ids([[1]])
    torch.accelerator.synchronize()

    assert torch.all(pool[:2, 0] == 1) and torch.all(pool[4:, 0] == 1)
    assert torch.all(pool[2, 0] == 0) and torch.all(pool[3, 0] == 0)
    assert torch.all(pool[:, 1] == 1), "the interleaved neighbor must survive"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_warmup_compiles_every_n_blocks_specialization():
    """After warmup, no launch should trigger a first-request JIT compile.

    The block count is carried by the launch grid, so changing it must reuse
    the warmup's compiled kernel.
    """
    device = torch.device("cuda")
    num_blocks = 64
    storage = torch.ones((num_blocks, 4), dtype=torch.int32, device=device)
    zeroer = _zeroer_for([storage])

    def compiled_variants() -> set:
        return {
            key
            for caches in _zero_kv_blocks_kernel.device_caches.values()
            for key in caches[0]
        }

    zeroer.warmup(num_blocks)
    torch.accelerator.synchronize()
    warmed = compiled_variants()
    assert warmed

    for n_blocks in (1, 2, 3, 16, 32):
        zeroer.zero_block_ids([list(range(n_blocks))])
    torch.accelerator.synchronize()

    assert compiled_variants() == warmed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_warmup_respects_available_block_count():
    """An empty KV cache must not be warmed with out-of-range block IDs."""
    device = torch.device("cuda")
    storage = torch.ones((1, 4), dtype=torch.int32, device=device)
    zeroer = _zeroer_for([storage])

    zeroer.warmup(0)
    torch.accelerator.synchronize()

    assert torch.all(storage == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pages_with_no_large_common_divisor_are_fully_zeroed():
    """Page sizes whose only common divisor is small must still zero fully.

    DeepSeek-V4 mixes a 9344-element MLA page, a 2112-element indexer page and
    an 8192-element compressor page. Neither 9344 nor 2112 is a multiple of
    the 1024-element chunk this kernel uses -- both leave a tail that only the
    store mask covers.
    """
    device = torch.device("cuda")
    num_blocks = 3
    page_sizes = [9344, 2112, 8192]
    storages = [
        torch.ones((num_blocks, ps), dtype=torch.int32, device=device)
        for ps in page_sizes
    ]
    zeroer = _zeroer_for(storages)

    zeroer.zero_block_ids([[1]])
    torch.accelerator.synchronize()

    for storage, ps in zip(storages, page_sizes):
        assert torch.all(storage[0] == 1), "block 0 must be untouched"
        assert torch.all(storage[2] == 1), "block 2 must be untouched"
        # The whole page, tail included -- a truncating chunk map would leave
        # the last ps % 1024 elements set.
        assert torch.all(storage[1] == 0), f"page {ps} not fully zeroed"


@pytest.mark.skip_global_cleanup
def test_every_launched_program_has_work():
    """No program may be launched only to exit empty.

    The chunk list is flattened per (segment, sub-block) rather than sized to
    the largest page, so the program count is the sum over segments of
    ``ratio * cdiv(extent, CHUNK_ELEMS)`` -- and no chunk crosses a sub-block
    boundary.
    """
    page_sizes = [9344, 2112, 8192, 4]
    ratios = [1, 2, 1, 4]
    meta = KVBlockZeroer.build_meta(
        [0] * len(page_sizes),
        [ps * 2 for ps in page_sizes],  # interleaved: stride != extent
        page_sizes,
        ratios,
        torch.device("cpu"),
    )
    assert meta is not None
    _, seg_periods, chunk_seg, chunk_base, chunk_len, n_chunks = meta
    chunk_elems = KVBlockZeroer.CHUNK_ELEMS

    expected = sum(
        r * ((ps + chunk_elems - 1) // chunk_elems) for ps, r in zip(page_sizes, ratios)
    )
    assert n_chunks == expected
    assert chunk_seg.numel() == n_chunks
    assert seg_periods.tolist() == [ps * 2 * r for ps, r in zip(page_sizes, ratios)]

    # Chunks tile each (segment, sub-block) extent exactly once, never
    # crossing into the interleaved gap between sub-blocks.
    covered: dict[tuple[int, int], set[int]] = {}
    for seg, base, length in zip(
        chunk_seg.tolist(), chunk_base.tolist(), chunk_len.tolist()
    ):
        stride = page_sizes[seg] * 2
        sub_block, within = divmod(base, stride)
        assert sub_block < ratios[seg]
        assert within + length <= page_sizes[seg], "chunk leaks past its extent"
        rows = covered.setdefault((seg, sub_block), set())
        assert not rows & set(range(within, within + length)), "overlap"
        rows.update(range(within, within + length))
    for (seg, _), rows in covered.items():
        assert rows == set(range(page_sizes[seg]))
    assert len(covered) == sum(ratios)


class _FakeBackend:
    """[num_blocks, page] storages: dim 0 is the block dim."""

    @staticmethod
    def get_kv_cache_block_dim(
        kernel_block_size, num_kv_heads, head_size, cache_dtype_str
    ):
        return 0


class _NonAttentionSpec:
    """Stands in for MambaSpec: a KV spec the zeroer must skip."""


def _zeroer_init_for(groups):
    """Drive KVBlockZeroer.__init__ with fake attention groups.

    groups: list of (group_id, spec, [(layer_name, tensor)]). Only __init__'s
    segment-table construction runs -- no kernel launch -- so this works on
    CPU and pins the coverage contract independently of CUDA.
    """
    static_ctx = {}
    fake_groups = []
    for group_id, spec, layers in groups:
        for name, tensor in layers:
            static_ctx[name] = SimpleNamespace(kv_cache=tensor)
        fake_groups.append(
            SimpleNamespace(
                kv_cache_spec=spec,
                kv_cache_group_id=group_id,
                backend=_FakeBackend,
                layer_names=[name for name, _ in layers],
            )
        )
    zeroer = KVBlockZeroer(
        torch.device("cpu"),
        attn_groups_iter=fake_groups,
        kernel_block_sizes=[16, 16],
        static_forward_context=static_ctx,
    )
    return zeroer


@pytest.mark.skip_global_cleanup
def test_sliding_window_group_gets_its_own_segment_table():
    """The coverage gate widened from FullAttentionSpec to AttentionSpec.

    A SlidingWindowSpec group (the SWA KV window / DeepseekV4 fp32 compressor
    state) previously produced NO segments, so its newly allocated blocks were
    never zeroed and a reused block's previous tenant's bytes survived.
    """
    page = 16 * 2 * 8  # block_size * num_kv_heads * head_size
    full = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16
    )
    swa = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        sliding_window=8,
        dtype=torch.float32,
    )
    t_full = torch.empty((4, page), dtype=torch.float16)
    t_swa = torch.empty((4, page), dtype=torch.float32)

    zeroer = _zeroer_init_for(
        [
            (0, full, [("fa.layer0", t_full)]),
            (1, swa, [("swa.state0", t_swa)]),
        ]
    )

    assert set(zeroer._group_meta) == {0, 1}


@pytest.mark.skip_global_cleanup
def test_non_attention_spec_is_skipped():
    page = 16 * 2 * 8
    full = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16
    )
    zeroer = _zeroer_init_for(
        [
            (0, full, [("fa.layer0", torch.empty((4, page), dtype=torch.float16))]),
            (
                1,
                _NonAttentionSpec(),
                [("mamba.state0", torch.empty((4, page), dtype=torch.float32))],
            ),
        ]
    )
    assert set(zeroer._group_meta) == {0}


@pytest.mark.skip_global_cleanup
def test_same_data_ptr_in_two_groups_keeps_both_segments():
    """Per-group dedup: the packed DeepseekV4 slab layout can surface the
    same base address to more than one group; a global pointer dedup silently
    dropped the later group's table entirely."""

    page = 16 * 2 * 8
    full = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16
    )
    swa = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        sliding_window=8,
        dtype=torch.float32,
    )
    # The SAME tensor object registered under both groups: identical
    # data_ptr. Group 1's table must survive anyway.
    shared = torch.empty((4, page), dtype=torch.float32)

    zeroer = _zeroer_init_for(
        [
            (0, full, [("g0.l0", shared)]),
            (1, swa, [("g1.l0", shared)]),
        ]
    )

    assert set(zeroer._group_meta) == {0, 1}
    assert zeroer._group_meta[0][5] >= 1
    assert zeroer._group_meta[1][5] >= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_slab_zeroes_only_the_owning_groups_bytes():
    """Two groups whose layers are strided views into one slab: each group's
    block step spans BOTH payloads, but a segment must zero only its own
    span. Zeroing the step wholesale would corrupt the neighbor group."""
    device = torch.device("cuda")
    num_blocks, page = 4, 16
    slab = torch.ones((num_blocks, 2 * page), dtype=torch.int32, device=device)
    g0_view = slab[:, :page]  # stride(0)=2*page, payload=page
    g1_view = slab[:, page:]  # same step, disjoint payload

    full = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=16, dtype=torch.int32
    )
    swa = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        sliding_window=8,
        dtype=torch.int32,
    )

    static_ctx = {
        "g0.l0": SimpleNamespace(kv_cache=g0_view),
        "g1.l0": SimpleNamespace(kv_cache=g1_view),
    }
    groups = [
        SimpleNamespace(
            kv_cache_spec=full,
            kv_cache_group_id=0,
            backend=_FakeBackend,
            layer_names=["g0.l0"],
        ),
        SimpleNamespace(
            kv_cache_spec=swa,
            kv_cache_group_id=1,
            backend=_FakeBackend,
            layer_names=["g1.l0"],
        ),
    ]
    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=groups,
        kernel_block_sizes=[16, 16],
        static_forward_context=static_ctx,
    )
    zeroer.zero_block_ids([[], [1]])

    assert torch.all(slab[:3, :] == 1), "untouched blocks must survive"
    assert torch.all(slab[1, :page] == 1), "group 0's payload must survive"
    assert torch.all(slab[1, page:] == 0), "group 1's payload must be zeroed"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_zeroes_exactly_one_block_per_layer(layout: KVCacheLayout):
    """The zeroer must zero every byte of the target block in every layer and nothing
    outside it — per head-group region under LHBNC, and never past the target block's
    tile under block-major layouts (no out-of-bounds writes, no clobbering)."""
    device = torch.device("cuda")
    num_blocks, num_layers = 4, 2
    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=2, head_size=8, dtype=torch.float32
    )
    raw = torch.empty(
        num_blocks * num_layers * spec.page_size_bytes,
        dtype=torch.int8,
        device=device,
    ).fill_(1)
    views = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)
    groups = [
        AttentionGroup(
            backend=None,
            layer_names=[f"layer.{i}" for i in range(num_layers)],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]
    ctx = {f"layer.{i}": SimpleNamespace(kv_cache=views[i]) for i in range(num_layers)}
    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=iter(groups),
        kernel_block_sizes=[spec.block_size],
        static_forward_context=ctx,
    )
    zeroer.zero_block_ids([2])
    torch.accelerator.synchronize()

    for view in views:
        assert (view[2] == 0).all(), layout
        for b in (0, 1, 3):
            assert (view[b].view(torch.int8) == 1).all(), layout
    zero_bytes = int((raw == 0).sum().item())
    assert zero_bytes == num_layers * spec.page_size_bytes, layout
