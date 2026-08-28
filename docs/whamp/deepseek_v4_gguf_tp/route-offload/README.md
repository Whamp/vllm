# Route-offload evidence

This directory owns the cold-expert offload route-skew evidence for goal `c81a9590-5605-4098-b899-86264f759b49`.

Read [ROUTE-OFFLOAD.md](ROUTE-OFFLOAD.md) first. Full 43-layer capture rejects the proposed H=224 cache. The separate fusion trace is the next GPU task.

## Verify the local tools

```bash
cd .research/gguf-tp-engine/route-offload
python3 -m unittest -v test_*.py
uv run --with ruff -- ruff check *.py
uv run --with ruff -- ruff format --check *.py
```

## Rebuild a replay request

`build_deepswe_route_replay.py` needs one Pi version-3 session and the first two bounded provider-request captures from that run:

```bash
./build_deepswe_route_replay.py \
  --session /path/to/session.jsonl \
  --provider-request-1 /path/to/provider_request_0001.json \
  --provider-request-2 /path/to/provider_request_0002.json \
  --output-request /tmp/task.request.json \
  --output-metadata /tmp/task.metadata.json
```

The command fails unless its conversion matches `provider_request_0002.json` exactly.

## Render token IDs without GPUs

Use the production GGUF-TP image. Mount the deterministic model view and its blob directory so its tokenizer symlinks resolve:

```bash
docker run --rm --network none \
  --entrypoint /opt/venv/bin/python \
  -v /path/to/render_deepswe_route_tokens.py:/render.py:ro \
  -v /path/to/requests:/requests:ro \
  -v /path/to/output:/out:rw \
  -v /path/to/model-view:/runtime-model:ro \
  -v /path/to/huggingface/blobs:/blobs:ro \
  club-3090/deepseek-v4-gguf-tp:3ec20ceb \
  /render.py --requests-dir /requests --model-dir /runtime-model --output-dir /out
```

The checked-in 12-task token files were rendered twice. All request, count, and token hashes matched. `render-manifest-v2.json` is the checked-in manifest from the scripted pass.

## Rebuild the exact static-layer summaries

The immutable GGUF has data start offset 5,333,824. The three I32 `tid2eid` tensors are contiguous 3,102,720-byte spans at relative offsets 0, 3,102,720, and 6,205,440. Extract those spans from GGUF SHA-256 `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`, then run:

```bash
./build_static_route_workload.py \
  --workload-id deepswe-pilot-final-context \
  --token-ids deepswe-pilot-final-context-token-ids.json \
  --tid2eid /tmp/tid2eid-layer-000.i32le \
  --tid2eid /tmp/tid2eid-layer-001.i32le \
  --tid2eid /tmp/tid2eid-layer-002.i32le \
  --output deepswe-pilot-static-routes.json
```

Repeat `--token-ids` once per session for the 12-task corpus. The summarizer resets LRU state at every file boundary.

## Rebuild the analysis

```bash
./analyze_route_skew.py \
  deepswe-pilot-static-routes.json \
  deepswe-12task-static-routes.json \
  --output-json static-layers-analysis.json \
  --output-markdown static-layers-analysis.md
```

The analyzer reports `NO-GO` as soon as any observed layer requires H99 at or above 248. It never reports `GO` unless all 43 layers are present and every workload stays at or below H99=224.

## Rebuild the dynamic capture analysis

```bash
uv run --with 'torch==2.13.0+cpu' \
  --default-index https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match -- \
  python extract_dynamic_route_histograms.py \
    --snapshot-dir dynamic-capture-20260820/snapshots \
    --baseline 00004 00005 \
    --pilot 00022 00023 \
    --corpus 00388 00389 \
    --pilot-output /tmp/pilot-routes.json \
    --corpus-output /tmp/corpus-routes.json

./analyze_route_skew.py /tmp/pilot-routes.json /tmp/corpus-routes.json \
  --output-json /tmp/analysis.json \
  --output-markdown /tmp/analysis.md
```

The extractor requires four identical TP-rank snapshots at every boundary and equal token-row totals across all 43 layer deltas.
