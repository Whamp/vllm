# Qwen3.8 capacity-kernel gate evidence

This directory preserves the first RTX 3090/SM86 acceptance results for the compressed hyperconnection and direct Q8-K/Q4-V QSA implementations.

- `hyper-result.json`: real layer-0 hyperconnection numerical, CUDA-Graph, storage, and timing result.
- `qsa-result.json`: fixed-shape QSA numerical, CUDA-Graph, writer, and reader timing result.
- `hyper-gate.py.txt` and `qsa-gate.py.txt`: exact executed gate sources.
- `hc_int8.py.txt` and `qsa_q8k_q4v.py.txt`: exact executed candidate sources.
- `run.sh.txt`: fail-closed runner with the exact production restore contract.
- `final-state.json`: post-gate production identity, health, restart, swap, safety, and timer state.
- `SHA256SUMS`: hashes for every evidence artifact above plus `final-state.json`.

The hyperconnection gate corresponds to Whamp/vLLM commit `070ffd51032a4261d9b051d8e8274d19c99fa8ba`. The QSA gate corresponds to `7de82f834dfd28642c13b461143fc13c8a71e515`.

Both candidates passed numerical and deterministic CUDA-Graph checks and failed their unchanged performance thresholds. They were not used in a full-model launch.

Verify from this directory:

```bash
sha256sum -c SHA256SUMS
```
