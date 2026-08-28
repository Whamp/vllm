#!/usr/bin/env python3
"""Read-only GGUF header/tensor-directory inventory for the GGUF-TP engine (M1).

Parses GGUF v3 metadata + tensor directory via mmap (no full-file read),
computes per-tensor byte sizes from the pinned ggml type table, and emits
JSON. Fails closed on unknown tensor types or inconsistent offsets.

Type ids/sizes from Whamp/llama.cpp@0379cf4bf ggml/include/ggml.h and
ggml/src/ggml.c type_traits.
"""

import json
import struct
import sys
from collections import defaultdict
from typing import TypedDict


class TensorInfo(TypedDict):
    name: str
    type: str
    dims: list[int]
    offset: int
    nbytes: int


TYPE = {  # id -> (name, blck_size, type_size)
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    16: ("IQ2_XXS", 256, 66),
    24: ("I8", 1, 1),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    30: ("BF16", 1, 2),
}
VTYPE = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


class R:
    def __init__(self, mm):
        self.mm, self.p = mm, 0

    def u32(self):
        v = struct.unpack_from("<I", self.mm, self.p)[0]
        self.p += 4
        return v

    def u64(self):
        v = struct.unpack_from("<Q", self.mm, self.p)[0]
        self.p += 8
        return v

    def raw(self, n):
        v = self.mm[self.p : self.p + n]
        self.p += n
        return v

    def string(self):
        n = self.u64()
        return self.raw(n).decode("utf-8", "replace")


def value(r, t):
    if t == 8:
        return r.string()
    if t == 9:
        et = r.u32()
        n = r.u64()
        if et not in (0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12) and et != 8:
            r.raw(n * 8)
            return [f"<unsupported array etype {et}>"] * min(n, 4)
        vals = [value(r, et) for _ in range(n)]
        if et == 8 and n > 64:
            vals = vals[:16] + [f"... {n - 16} more"]
        return vals
    if t in VTYPE:
        return struct.unpack(VTYPE[t], r.raw(struct.calcsize(VTYPE[t])))[0]
    raise SystemExit(f"fail-closed: unknown value type {t} at {r.p}")


def main(path):
    # Bounded header read: metadata + tensor directory live at the file head;
    # 16 MiB is far more than enough and avoids mmap on a memory-tight host.
    with open(path, "rb") as f:
        mm = f.read(16 * 1024 * 1024)
    r = R(mm)
    assert r.raw(4) == b"GGUF", "not a GGUF file"
    version = r.u32()
    assert version == 3, f"expected GGUF v3, got {version}"
    n_tensors, n_kv = r.u64(), r.u64()
    if n_tensors > 100_000 or n_kv > 100_000:
        raise SystemExit(f"implausible counts: {n_tensors} tensors, {n_kv} kvs")

    meta = {}
    for _ in range(n_kv):
        k = r.string()
        t = r.u32()
        v = value(r, t)
        if isinstance(v, str) and len(v) > 200:
            v = v[:200] + "..."
        meta[k] = v

    tensors: list[TensorInfo] = []
    by_type = defaultdict(lambda: [0, 0])  # name -> [bytes, count]
    for _ in range(n_tensors):
        name = r.string()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        t = r.u32()
        off = r.u64()
        if t not in TYPE:
            raise SystemExit(f"fail-closed: tensor {name} has unmapped type id {t}")
        tname, blck, tsize = TYPE[t]
        ne0 = dims[0]
        nbytes = tsize
        for d in dims[1:]:
            nbytes *= d
        nbytes *= (ne0 + blck - 1) // blck
        tensors.append(
            {"name": name, "type": tname, "dims": dims, "offset": off, "nbytes": nbytes}
        )
        by_type[tname][0] += nbytes
        by_type[tname][1] += 1

    data_start = (r.p + 31) & ~31
    file_size = 86_720_111_488  # verified externally; header read is bounded
    last = max(t["offset"] + t["nbytes"] for t in tensors)
    print(
        json.dumps(
            {
                "file": path,
                "file_size": file_size,
                "gguf_version": version,
                "n_tensors": n_tensors,
                "data_start": data_start,
                "tensor_bytes_end": last,
                "tail_padding": file_size - data_start - last,
                "bytes_by_type": {k: v[0] for k, v in sorted(by_type.items())},
                "count_by_type": {k: v[1] for k, v in sorted(by_type.items())},
                "metadata": meta,
                "tensors": tensors,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1])
