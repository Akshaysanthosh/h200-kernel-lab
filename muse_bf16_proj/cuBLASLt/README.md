> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# cuBLASLt preparation and reset handoff

Canonical project: `muse_bf16_proj`.
The earlier workspace remains intact as a second copy. Continue in this repository.

## Current best

`submission.py` and `champions/2026-10-30_1.227412/submission.py` are the unchanged scored winner:
1.227411869x, 9/9 tests, no flags, job `35d40255820f4df4`.
SHA256: `9ea10c81639eb23d6c9eb832465190328664fd655e89e9ff81732b94e10b8a36`.
The winner's old introductory docstring is stale; RESEARCH.md describes its actual implementation.

## Candidate architecture

`lt_plan.cpp` is an original host-side cuBLASLt wrapper using BF16 matrices and FP32 compute.
It creates descriptors and selects an eligible heuristic result during preparation, outside graph capture.
Selection is bounded (32 heuristic descriptions, no benchmark/tuning loop). It accepts only no-reduction
or compute-type reduction schemes, excluding BF16 intermediate reductions. GPU execution reads the
fresh supplied X and supplied/prepared W, allocates fresh output and invocation-owned workspace,
and launches on PyTorch's current CUDA stream. No reference imports, harness I/O, timing changes,
cached outputs, precision changes, GPU network activity, or paid resources.

The wrapper computes column-major Y^T[N,M] = W[N,K] * X^T[K,M]. Original row-major W[N,K]
storage requires op(A)=T and lda=K; prepared row-major W^T[K,N] storage requires op(A)=N and lda=N.
The output's column-major storage is the required row-major Y[M,N]. Pointer alignment is checked
before dispatch. Smaller unpooler slices, one-token cases, router, other batch sizes, and unavailable
heuristic ranks retain the champion implementation. In particular, a GPU check must confirm that
the intended cuBLASLt path actually ran rather than assuming an all-fallback result is a win.

Six standalone Python candidates compare original/transposed storage and eligible heuristic ranks
0, 1, 2, with a 32 MiB workspace cap. Algorithm IDs are queried from the installed library, not copied
from a different CUDA version. Candidates are generated from the hash-verified champion and contain
the C++ source inline, as required by the competition. Candidates remain GPU-unvalidated.

## Local verification

```sh
python3 tools/build_lt_candidates.py
python3 -m unittest discover -s tests -v
python3 tools/run_lt_check.py candidates/lt/lt_original_h0.py
```

The last command is a quota-inspecting dry run and never launches a check. CPU tests verify the
rectangular column-major math mapping and quota/manifest guards, not actual BF16 accuracy,
CUDA-graph behavior, allocation lifetimes, or performance on a GPU.

An isolated local Colima profile `okc-build` is installed: 2 CPUs, 4 GiB RAM, 8 GiB root disk,
6 GiB container disk. It is not the default Docker context. The optional local image uses CPU PyTorch
headers plus CUDA/cuBLAS development headers to compile the wrapper to a host object. It cannot
run GPU kernels or fully link the PyTorch CUDA extension. Check `results/local-build/` for the actual
outcome; do not infer success from the Dockerfile. Full build/link/runtime verification still requires
the official H200 environment.

```sh
docker --context colima-okc-build build -t okc-lt-build:local cuBLASLt/local-build
docker --context colima-okc-build run --rm -v "$PWD:/work:ro" okc-lt-build:local
```

## First phase after reset

The one-time thread follow-up is scheduled for October 31, 2026 at 05:35 IST (00:05 UTC), after the
00:00 UTC quota reset. Verify the actual service quota first. No paid GPU or paid API provisioning.

```sh
python3 tools/run_lt_check.py candidates/lt/lt_original_h0.py --execute
```

Run one candidate at a time and inspect its result. Stop after a compile/correctness failure to diagnose
before spending another check. The wrapper prints a library version, number of native plans, and
selected pooler algorithm metadata. Use returned profiles to determine which dense shapes improved.
The runner enforces the reset time, positive live quota, manifest hash, preserved champion hash,
single-local-run lock, and an initial-phase cap of eight check attempts per UTC day. Interrupted or
ambiguous attempts count conservatively; inspect `okc jobs` and the lock before retrying.

The runner contains no submit command. Do not use scored submissions to bypass free-check limits.
Do not overwrite the champion or main submission with an unvalidated candidate. Retain improved
validated candidates separately and report evidence; do not automatically run all six candidates.

Inspect `results/lt/YYYY-MM-DD/` for records. If source is repaired, regenerate candidates/manifest,
repeat relevant local checks, and document the changed hypothesis before the next official check.

## API references

- https://docs.nvidia.com/cuda/cublas/index.html
- https://docs.pytorch.org/docs/stable/cpp_extension.html
- https://github.com/NVIDIA/CUDALibrarySamples/tree/main/cuBLASLt

The sample API patterns were consulted; no NVIDIA sample implementation was copied into the wrapper.

## Verified preparation status

All seven CPU preflight tests passed. The native wrapper compiled to a host object against PyTorch 2.9.0 CPU headers and CUDA 12.8 headers. CPU wheels omit a generated CUDA CMake header; the local probe uses PyTorch's supported C10_CUDA_NO_CMAKE_CONFIGURE_FILE switch. This switch is not passed to the submitted build. Neither GPU execution nor full CUDA extension linking was validated locally. See results/local-build/status.json and compile-probe.log. The build VM is stopped after validation; restart it only when further local compilation is needed. No official GPU checks were used in this preparation turn.
