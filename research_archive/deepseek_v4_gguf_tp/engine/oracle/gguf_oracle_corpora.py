"""Random and adversarial block corpora for the GGUF format L0 oracle."""

import numpy as np
from gguf_format_decoders import GGUF_BLOCK_SIZE


def _finite_fp16(bits: np.ndarray) -> np.ndarray:
    return (bits & np.uint16(0x7C00)) != np.uint16(0x7C00)


def _fp16_bytes(bits: int) -> np.ndarray:
    return np.frombuffer(int(bits).to_bytes(2, "little"), dtype=np.uint8)


def generate_random_gguf_blocks(
    format_name: str, rng: np.random.Generator, block_count: int
) -> np.ndarray:
    """Generate random packed blocks while keeping scale fields finite."""
    blocks = rng.integers(
        0, 256, size=(block_count, GGUF_BLOCK_SIZE[format_name]), dtype=np.uint8
    )
    scale_fields = {
        "q8_0": [(0, 2)],
        "q2_K": [(80, 82), (82, 84)],
        "iq2_xxs": [(0, 2)],
    }[format_name]
    for first, last in scale_fields:
        scale_bits = blocks[:, first:last].view(np.uint16).reshape(-1)
        while True:
            invalid = ~_finite_fp16(scale_bits)
            replacement_count = int(invalid.sum())
            if replacement_count == 0:
                break
            replacements = (
                rng.integers(0, 256, size=(replacement_count, 2), dtype=np.uint8)
                .view(np.uint16)
                .reshape(-1)
            )
            scale_bits[invalid] = np.where(
                _finite_fp16(replacements), replacements, np.uint16(0x3C00)
            )
    return blocks


def _new_block(format_name: str, scale_bits: int | None = None) -> np.ndarray:
    block = np.zeros((1, GGUF_BLOCK_SIZE[format_name]), dtype=np.uint8)
    if scale_bits is not None:
        block[0, 0:2] = _fp16_bytes(scale_bits)
    return block


def _q8_adversarial_blocks() -> list[tuple[str, np.ndarray]]:
    cases: list[tuple[str, np.ndarray]] = []
    for name, value in (("qs=-128", 0x80), ("qs=127", 0x7F)):
        block = _new_block("q8_0")
        block[:, 2:34] = value
        cases.append((name, block))
    block = _new_block("q8_0")
    block[:, 2:34] = np.tile(np.array([0x80, 0x7F], dtype=np.uint8), 16)
    cases.append(("qs=alt", block))
    for name, scale_bits, code in (
        ("d=max", 0x7BFF, 0x7F),
        ("d=subnorm", 0x0001, 0x80),
        ("d=-max", 0xFBFF, 0x7F),
    ):
        block = _new_block("q8_0", scale_bits)
        block[:, 2:34] = code
        cases.append((name, block))
    return cases


def _q2_k_adversarial_blocks() -> list[tuple[str, np.ndarray]]:
    cases: list[tuple[str, np.ndarray]] = []
    for name, value in (
        ("scales=0xFF", 0xFF),
        ("scales=0x0F", 0x0F),
        ("scales=0xF0", 0xF0),
    ):
        block = _new_block("q2_K")
        block[:, 0:16] = value
        cases.append((name, block))
    block = _new_block("q2_K", 0x7B00)
    block[:, 0:16] = 0xFF
    block[:, 16:80] = 0xFF
    cases.append(("qs+scale-max", block))
    for name, scale_bits in (
        ("d=+max", 0x7BFF),
        ("d=subnorm", 0x0001),
        ("d=-max", 0xFBFF),
    ):
        block = _new_block("q2_K", scale_bits)
        block[:, 0:16] = 0xFF
        block[:, 16:80] = 0xFF
        cases.append((name, block))
    block = _new_block("q2_K")
    block[:, 82:84] = _fp16_bytes(0x7BFF)
    block[:, 0:16] = 0xFF
    cases.append(("dmin=+max", block))
    block = _new_block("q2_K", 0x3C00)
    block[:, 0:16] = np.arange(16, dtype=np.uint8) * np.uint8(0x11)
    block[:, 16:80] = np.arange(64, dtype=np.uint8)
    cases.append(("distinct-scales", block))
    block = _new_block("q2_K", 0x3C00)
    block[:, 0:16] = 0x0F
    block[:, 16:48] = 0xA5
    block[:, 48:80] = 0x5A
    cases.append(("chunk-boundary", block))
    return cases


def _iq2_xxs_adversarial_blocks() -> list[tuple[str, np.ndarray]]:
    cases = [("all-zero-qs", _new_block("iq2_xxs"))]
    block = _new_block("iq2_xxs")
    block[:, 2:66] = 0xFF
    cases.append(("all-0xFF-qs", block))
    block = _new_block("iq2_xxs", 0x3C00)
    block[:, 2:66] = 0xFF
    block[:, 6:10] = 0x00
    cases.append(("hi-ls=0", block))
    block = _new_block("iq2_xxs", 0x3C00)
    block[:, 6:10] = 0xFF
    cases.append(("hi-ls=max", block))
    block = _new_block("iq2_xxs", 0x3C00)
    block[:, 3:6] = 0xFF
    cases.append(("sign-sel-ones", block))
    block = _new_block("iq2_xxs", 0x3C00)
    block[:, 58:66] = 0xFF
    block[:, 60:64] = 0x00
    cases.append(("group7-boundary", block))
    for name, first, last in (
        ("aux8[0]=255", 2, 4),
        ("aux8[last]=255", 62, 64),
    ):
        block = _new_block("iq2_xxs")
        block[:, first:last] = 0xFF
        cases.append((name, block))
    for name, scale_bits in (
        ("d=+max", 0x7BFF),
        ("d=subnorm", 0x0001),
        ("d=-max", 0xFBFF),
    ):
        block = _new_block("iq2_xxs", scale_bits)
        block[:, 2:66] = 0xFF
        cases.append((name, block))
    return cases


def build_adversarial_gguf_blocks(format_name: str) -> list[tuple[str, np.ndarray]]:
    """Return named boundary cases for one GGUF block format."""
    return {
        "q8_0": _q8_adversarial_blocks,
        "q2_K": _q2_k_adversarial_blocks,
        "iq2_xxs": _iq2_xxs_adversarial_blocks,
    }[format_name]()


def build_nonfinite_q2_k_probe(format_name: str) -> np.ndarray | None:
    """Return the NaN/Inf scale probe for Q2_K; other formats return none."""
    if format_name != "q2_K":
        return None
    blocks = np.zeros((2, GGUF_BLOCK_SIZE[format_name]), dtype=np.uint8)
    blocks[:, 0:16] = 0x0F
    blocks[0, 80:82] = _fp16_bytes(0x7C00)
    blocks[1, 80:82] = _fp16_bytes(0x7E00)
    return blocks
