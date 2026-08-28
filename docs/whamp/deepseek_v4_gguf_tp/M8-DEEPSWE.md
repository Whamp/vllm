<!-- markdownlint-disable MD060 -->

# M8 — DeepSWE quality gate (Will revision 2026-08-18)

Status: **pilot approved to execute.** Full multi-seed grid **cancelled.**

## Will's decision (2026-08-18)

Will explicitly **rejects the ≥72-cell multi-seed DeepSWE grid** (12 tasks × ≥3
seeds × 2 engines). It is too expensive on local compute and **must not be
run or scheduled**, regardless of pilot outcome.

M8 quality evidence for this project is instead:

1. **One cell:** GGUF-TP runs **one task, one seed (rep0)** on the locked
   SuperJSON pilot harness.
2. **Baseline reuse:** llama.cpp results for the same task are **already
   available** from prior Antirez-GGUF DeepSWE runs; do **not** re-run
   llama.cpp for M8 unless the existing artifact is incompatible with the
   locked pilot task revision.
3. **Pass criterion:** Will's judgment that GGUF-TP is **close enough** to the
   llama.cpp baseline on strict solve + partial reward for this cell,
   acknowledging high run-to-run variance from a single sample. **No
   pre-registered permutation/bootstrap statistic gates M8 or M9.**

This replaces PLAN §6 class-D paired multi-seed requirements and the M8 row's
"pilot-priced full grid" language for this project only. Historical review
findings recommending ≥3 seeds remain archived under `reviews/`; they are
superseded by this owner decision.

## Approved pilot (GGUF-TP run only)

| Field | Value |
|---|---|
| Approval | **Granted 2026-08-18 by Will** |
| Plan identity | `sha256:7ac3e4c4992a0c2aea9b3b7e75e483b127d0a3f678fd235767899b02bd9a6859` |
| Config release | `baseline-vllm-deepseek-v4-flash-0731-gguf-tp@1.0.0` |
| Config lock | `sha256:8b553e2d059aad34d8a71c7b5e9f8370bb995349f5338b4d67cce35aba721494` |
| Task | `superjson-error-stack-serialization` |
| Task revision | `sha256:15bc4b7c2a8f14e2edfa8903048344a9606281b8f1569f19db5a7c93e8dd4aeb` |
| Cells | **1** (1 task × 1 rep × GGUF-TP only) |
| Launch plan | `deep-swe-bench/.worktrees/gguf-tp-deepswe/plans/gguf-tp-superjson-pilot/` |
| Harness worktree | `deep-swe-bench` branch `eval/gguf-tp-deepswe` |

**Permission:** agents may **launch the approved GGUF-TP pilot** under the
locked plan hash above. No further Will approval is required for this single
cell. **Do not expand** to additional tasks, seeds, or configs without a new
explicit owner decision.

## Baseline comparison (llama.cpp reuse)

| Field | Value |
|---|---|
| Baseline config | `baseline-llamacpp-deepseek-v4-flash-0731-iq2xxs@1.1.0` |
| Baseline plan (reference) | `sha256:2fab5b9d289d4e9451222ad1648d5a00bdf6bca1275833fc8eaead705836a84f` |
| Historical 12-task anchor | Antirez GGUF via llama.cpp: **6 strict solves / 96.57% partial** (post-DSML-fix final comparison) |

For M8, compare the **pilot task cell only** against the existing llama.cpp
result for `superjson-error-stack-serialization` at rep0. Record strict solve,
partial reward, and qualitative notes (tool-use path, degeneration, finish
reason). Present side-by-side to Will for the closeness call.

## Prerequisites and ordering

- **M6 remains required before M8 counts toward promotion.** The layer-oracle
  class-B gate must pass first (`M6-LAYER-ORACLE-SPEC.md`).
- The pilot may be **scheduled** while M6 bisect work continues, but **M8 PASS**
  cannot be recorded until both M6 passes and the pilot comparison is
  reviewed.
- Quick quality packs (27/30) and NIAH remain supporting smoke evidence, not
  substitutes for this cell or for M6.

## M8 pass / fail (revised)

- **Pass:** M6 passed; GGUF-TP pilot cell completed under locked plan; Will
  judges the result close enough to the reused llama.cpp baseline for this
  task (document the judgment in `PROGRESS.md` with result paths).
- **Fail:** Will judges material regression on the pilot cell, or the pilot
  run fails harness verification / shows clear degeneration → bisect per PLAN
  §8 M8 kill ("divergence → component bisect"). **Do not** attempt to recover
  by running the cancelled 72-cell grid.

## Evidence bundle (on completion)

- GGUF-TP `result.json` path from the pilot run
- Reused llama.cpp baseline `result.json` path and provenance
- Side-by-side strict/partial summary + Will's recorded judgment
- Wall time and any harness warnings from the launch receipt

## What agents must not do

- Run or propose the ≥72-cell multi-seed grid
- Re-run llama.cpp for M8 unless the existing baseline artifact is provably
  incompatible with the locked task revision
- Treat a single SuperJSON cell as statistically equivalent to the historical
  12-task 6/12 headline — it is a **targeted sanity check**, not a replica of
  the full grid
- Mark M8 or the overall goal complete without Will's explicit closeness
  judgment recorded
