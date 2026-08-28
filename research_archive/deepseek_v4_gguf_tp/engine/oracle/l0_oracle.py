#!/usr/bin/env python3
"""Run the GGUF format L0 oracle against pinned llama.cpp dequantization.

Reference A is ref_a.so, a verbatim extraction from Whamp/llama.cpp at
0379cf4bf. Reference B is the independent NumPy decoder module written only
from FORMAT-CONTRACT.md. Pass requires bitwise-equal fp32 output across every
random and adversarial block.
"""

import ctypes
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from gguf_format_decoders import (
    FLOAT32,
    GGUF_BLOCK_SIZE,
    GGUF_ELEMENTS_PER_BLOCK,
    GGUF_FORMAT_DECODERS,
)
from gguf_oracle_corpora import (
    build_adversarial_gguf_blocks,
    build_nonfinite_q2_k_probe,
    generate_random_gguf_blocks,
)

ORACLE_DIRECTORY = Path(__file__).resolve().parent
PINNED_GGML_SOURCE = Path("/home/will/projects/llama.cpp-ds4-study/ggml/src")
EVIDENCE_DIRECTORY = ORACLE_DIRECTORY.parent / "evidence"
RANDOM_SEED = 20260817
RANDOM_BLOCK_COUNT = 10_000
GGUF_FORMAT_NAMES = ("q8_0", "q2_K", "iq2_xxs")


def build_pinned_llamacpp_reference() -> Path:
    """Compile the pinned llama.cpp dequantization extraction into ref_a.so."""
    shared_object = ORACLE_DIRECTORY / "ref_a.so"
    subprocess.run(
        [
            "cc",
            "-O2",
            "-shared",
            "-fPIC",
            f"-I{PINNED_GGML_SOURCE}",
            str(ORACLE_DIRECTORY / "ref_a.c"),
            "-o",
            str(shared_object),
        ],
        check=True,
    )
    return shared_object


def load_pinned_llamacpp_reference(shared_object: Path) -> ctypes.CDLL:
    """Load ref_a.so and declare its dequantization and metadata interfaces."""
    library = ctypes.CDLL(str(shared_object))
    for function_name in ("ref_a_q8_0", "ref_a_q2_K", "ref_a_iq2_xxs"):
        function = getattr(library, function_name)
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    library.ref_a_table.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t)]
    library.ref_a_table.restype = ctypes.c_void_p
    library.ref_a_sizes.restype = ctypes.POINTER(ctypes.c_size_t)
    library.ref_a_sizes.argtypes = []
    library.ref_a_off_qs.restype = ctypes.POINTER(ctypes.c_size_t)
    library.ref_a_off_qs.argtypes = []
    return library


def run_pinned_dequantization(
    library: ctypes.CDLL, format_name: str, packed_blocks: np.ndarray
) -> np.ndarray:
    """Decode contiguous packed blocks through the pinned C reference."""
    element_count = packed_blocks.shape[0] * GGUF_ELEMENTS_PER_BLOCK[format_name]
    output = np.empty(element_count, dtype=FLOAT32)
    getattr(library, f"ref_a_{format_name}")(
        packed_blocks.ctypes.data,
        output.ctypes.data,
        ctypes.c_int64(element_count),
    )
    return output


def load_iq2_lookup_tables(
    library: ctypes.CDLL,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Load IQ2 lookup table bytes and compute evidence hashes."""
    tables: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    table_specs = (
        ("iq2xxs_grid", np.uint64, 256),
        ("ksigns_iq2xs", np.uint8, 128),
        ("kmask_iq2xs", np.uint8, 8),
    )
    for table_name, dtype, expected_elements in table_specs:
        table_bytes = ctypes.c_size_t()
        pointer = library.ref_a_table(table_name.encode(), ctypes.byref(table_bytes))
        if not pointer:
            raise RuntimeError(f"GGUF L0 oracle table missing: {table_name}")
        buffer = (ctypes.c_char * table_bytes.value).from_address(pointer)
        table = np.frombuffer(bytes(buffer), dtype=dtype).copy()
        if table.size != expected_elements:
            raise RuntimeError(
                f"GGUF L0 oracle table size mismatch: {table_name} "
                f"got {table.size}, expected {expected_elements}"
            )
        tables[table_name] = table
        hashes[table_name] = hashlib.sha256(table.tobytes()).hexdigest()
    tables["iq2xxs_grid"] = tables["iq2xxs_grid"].view(np.uint8).reshape(256, 8)
    return tables, hashes


def bitwise_float32_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare float32 output bit patterns, including signed zero."""
    return bool(np.array_equal(left.view(np.uint32), right.view(np.uint32)))


def nanaware_bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare bitwise except that any two NaNs are equivalent payloads."""
    both_nan = np.isnan(left) & np.isnan(right)
    scrubbed_left = np.where(both_nan, FLOAT32(1.0), left)
    scrubbed_right = np.where(both_nan, FLOAT32(1.0), right)
    return bitwise_float32_equal(scrubbed_left, scrubbed_right)


def verify_reference_layout(library: ctypes.CDLL) -> tuple[list[int], list[int]]:
    """Verify C struct sizes and qs offsets against FORMAT-CONTRACT.md."""
    sizes = np.ctypeslib.as_array(library.ref_a_sizes(), shape=(3,)).tolist()
    qs_offsets = np.ctypeslib.as_array(library.ref_a_off_qs(), shape=(3,)).tolist()
    expected_sizes = [
        GGUF_BLOCK_SIZE["q8_0"],
        GGUF_BLOCK_SIZE["q2_K"],
        GGUF_BLOCK_SIZE["iq2_xxs"],
    ]
    if sizes != expected_sizes or qs_offsets != [2, 16, 2]:
        raise RuntimeError(
            f"GGUF L0 oracle layout mismatch: sizes={sizes}, offsets={qs_offsets}"
        )
    return sizes, qs_offsets


def check_one_gguf_format(
    library: ctypes.CDLL,
    tables: dict[str, np.ndarray],
    format_name: str,
    random_generator: np.random.Generator,
) -> dict[str, object]:
    """Check random, named boundary, and nonfinite blocks for one format."""
    random_blocks = np.ascontiguousarray(
        generate_random_gguf_blocks(format_name, random_generator, RANDOM_BLOCK_COUNT)
    )
    pinned_random = run_pinned_dequantization(library, format_name, random_blocks)
    independent_random = GGUF_FORMAT_DECODERS[format_name](random_blocks, tables)
    random_pass = bitwise_float32_equal(pinned_random, independent_random)

    adversarial_cases = build_adversarial_gguf_blocks(format_name)
    adversarial_names = [case_name for case_name, _ in adversarial_cases]
    adversarial_blocks = np.ascontiguousarray(
        np.concatenate([block for _, block in adversarial_cases], axis=0)
    )
    pinned_adversarial = run_pinned_dequantization(
        library, format_name, adversarial_blocks
    )
    independent_adversarial = GGUF_FORMAT_DECODERS[format_name](
        adversarial_blocks, tables
    )
    adversarial_pass = bitwise_float32_equal(
        pinned_adversarial, independent_adversarial
    )

    nonfinite_pass = True
    nonfinite_blocks = build_nonfinite_q2_k_probe(format_name)
    if nonfinite_blocks is not None:
        with np.errstate(invalid="ignore"):
            nonfinite_pass = nanaware_bitwise_equal(
                run_pinned_dequantization(library, format_name, nonfinite_blocks),
                GGUF_FORMAT_DECODERS[format_name](nonfinite_blocks, tables),
            )

    result: dict[str, object] = {
        "random_blocks": RANDOM_BLOCK_COUNT,
        "random_bitwise_pass": random_pass,
        "adversarial_cases": adversarial_names,
        "adversarial_bitwise_pass": adversarial_pass,
        "nonfinite_nanaware_pass": nonfinite_pass,
    }
    if not random_pass:
        mismatch = np.nonzero(
            pinned_random.view(np.uint32) != independent_random.view(np.uint32)
        )[0][:5]
        result["first_random_mismatches"] = [int(index) for index in mismatch]
        result["sample_ref"] = [float(pinned_random[index]) for index in mismatch[:3]]
        result["sample_mine"] = [
            float(independent_random[index]) for index in mismatch[:3]
        ]
    return result


def run_gguf_format_l0_oracle() -> int:
    """Run every GGUF format check and write l0-report.json."""
    EVIDENCE_DIRECTORY.mkdir(exist_ok=True)
    library = load_pinned_llamacpp_reference(build_pinned_llamacpp_reference())
    sizes, qs_offsets = verify_reference_layout(library)
    tables, table_hashes = load_iq2_lookup_tables(library)
    random_generator = np.random.default_rng(RANDOM_SEED)

    format_results = {
        name: check_one_gguf_format(library, tables, name, random_generator)
        for name in GGUF_FORMAT_NAMES
    }
    passed = all(
        result[check]
        for result in format_results.values()
        for check in (
            "random_bitwise_pass",
            "adversarial_bitwise_pass",
            "nonfinite_nanaware_pass",
        )
    )
    report = {
        "seed": RANDOM_SEED,
        "n_random_blocks": RANDOM_BLOCK_COUNT,
        "pinned_source": "Whamp/llama.cpp@0379cf4bf889f3d28038a005210c4bc193fc8ba1",
        "struct_sizes": sizes,
        "qs_offsets": qs_offsets,
        "table_sha256": table_hashes,
        "formats": format_results,
        "pass": bool(passed),
    }
    report_path = EVIDENCE_DIRECTORY / "l0-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(run_gguf_format_l0_oracle())
