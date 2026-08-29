# Agent guidance for Will's vLLM fork

This repository is a long-lived fork of `vllm-project/vllm`. Optimize for this
fork. It is not a staging ground for upstream contributions.

## Fork stance

- Treat upstream as a source of code and ideas, not as the authority for local
  design decisions.
- Optimize for the hardware, models, and workloads named in the task. If a
  choice depends on an unstated target, ask instead of inheriting upstream's
  priorities.
- Judge agent-authored changes by their correctness and evidence. Use ordinary
  commit authorship; add AI disclosures, attribution trailers, or DCO signoffs
  only when Will requests them.
- Use upstream issues and pull requests as technical research when relevant,
  not as a gate on fork-local work.

## Default performance target

Unless the task names another target, optimize and validate for one host with
four NVIDIA RTX 3090 GPUs, compute capability 8.6 (SM86). For changes that
affect kernels, memory capacity, partitioning, collectives, or scheduling,
exercise the relevant single-GPU and four-GPU paths. Inspect the host before
making a topology-dependent decision; the GPU count alone does not establish
available peer links or their layout.

## Git and upstream sync

- Inspect `git remote -v` before fetching, rebasing, pushing, or opening a pull
  request. Confirm the destination instead of assuming `origin` is the fork.
- Base fork work on the fork's integration branch. Use an upstream branch only
  when inspecting or integrating upstream changes.
- Preserve local behavior during upstream merges and rebases. Before resolving
  a conflict, inspect the fork-side commit and its tests; do not discard a local
  change merely because upstream changed the same code.
- Prefer contained changes when they fully meet fork requirements. Do not
  sacrifice correctness or 4x3090 performance merely to resemble upstream.

## Before changing code

Establish the affected execution path and its observable success condition.
For hardware-sensitive work, also identify the target GPU, backend, model,
dtype, tensor shapes, and parallel configuration that matter. Search for the
nearest implementation, tests, and benchmarks before adding a new path.

## Python environment

- Reuse the repository's `.venv` when it exists.
- Create and manage Python environments with `uv`. Install packages with
  `uv pip`; do not use bare `pip` or the system Python.
- Run Python tools through `.venv/bin/python` or an executable installed in the
  active `.venv`.
- Follow the source-install guide for the target platform under
  [`docs/getting_started/installation/`](docs/getting_started/installation/).
  For repeated C++/CUDA builds, follow
  [`docs/contributing/incremental_build.md`](docs/contributing/incremental_build.md).

## Validation

- Prove the requested behavior at the cheapest reliable level. Run the narrow
  test first, then the relevant broader suite and pre-commit hooks.
- Extend the nearest existing test module and fixtures when they fit. Assert
  observable behavior rather than private wiring.
- Put kernel correctness coverage in the relevant test suite. Extend the
  nearest existing kernel benchmark for performance experiments.
- Back GPU kernel changes with an independent correctness reference and results
  from the target hardware. Record the GPU, dtype, shapes, configuration,
  command, and before/after measurements.
- For changes that affect model output, accuracy, memory use, or serving
  behavior, exercise the affected model path. If local hardware or model access
  blocks that validation, report the exact untested path instead of claiming
  completion.

A change is complete when every requested behavior has corresponding test,
runtime, or benchmark evidence, and the final report names any validation that
remains unavailable.

## Local style

Use the code and configuration next to the change as the style source of truth.
Preserve established fork terminology and behavior. Keep comments brief and
use them for constraints or reasoning that the code cannot state.

## Context pointers

- **Model implementation or model tests:** Read
  [`docs/contributing/model/tests.md`](docs/contributing/model/tests.md) before
  changing model behavior or its coverage.
- **Security-sensitive work:** Read [`SECURITY.md`](SECURITY.md),
  [`docs/usage/security.md`](docs/usage/security.md), and
  [`docs/contributing/vulnerability_management.md`](docs/contributing/vulnerability_management.md)
  before changing trust boundaries, request handling, or deployment security.
- **Agent instruction changes:** Read
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  before editing this file or a guide it links to.
