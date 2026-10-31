> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# Top-10 projection leaderboard plan — 2026-10-30

Scope: top ten on muse_bf16_proj. This does not qualify for cash prizes by itself; those belong to the serving track. No GPU checks, submissions, paid compute, or paid APIs were used during this planning pass. The current best source is unchanged.

## Target and evidence

Latest CLI snapshot: rank 768 of 1,448, speedup 1.227411869x. Rank ten: 1.5915x; rank one: 1.6498x. Rankings are provisional and moving. Need a further 1.2966x speed factor, equivalent to 22.88% less aggregate time. Aim at least 1.65x as a working target, not a guarantee of future top ten.

Analysis is reproducible with planning/analyze_headroom.py, which consumes the latest scored profile and saved leaderboard snapshot and writes planning/headroom.json. It follows the six-load geometric-mean scoring and the supplied shape weights; it does not average individual kernel speedups.

Mean within-load time shares:

| Operation | Share |
|---|---:|
| Pooler | 33.5% |
| QKV/gate | 23.0% |
| Unpooler | 21.3% |
| Output projection | 11.0% |
| Router | 11.2% |

Counterfactual estimates, holding other costs fixed:

| Change | Estimated total score |
|---|---:|
| Router costs zero | 1.382x |
| Dense projections take 20% less time | 1.492x |
| Dense projections take 25% less time | 1.578x |
| Dense projections take 30% less time | 1.673x |
| Dense projections reach reference hardware floors; router unchanged | 1.790x |

The hardware-floor scenario is optimistic and not an attainable guarantee. Our timings include a judge-specific replay-overhead correction, so a roofline is a diagnostic guide rather than an exact physical bound on the reported metric.

## First-principles model

For X[M,K] @ W[N,K]^T with BF16 operands:

- FLOPs = 2*M*N*K.
- Minimal input/output traffic roughly = 2*N*K + 2*M*K + 2*M*N bytes.
- Arithmetic intensity approaches M FLOPs/byte when weight traffic dominates.
- Reference lower-time yardstick = max(bytes / bandwidth, FLOPs / peak compute).
- The task reference uses 4.8 TB/s and 989 TFLOP/s; their ratio is approximately 206 FLOPs/byte. Low-batch cases are therefore primarily memory-oriented in this optimistic model; M=256 can enter a more compute-sensitive regime. Achieved hardware efficiency is not equal to peak efficiency.

Practical time also includes launch/reduction overhead, insufficient parallelism, redundant loads, layout conversion, instruction overhead, and register/shared-memory constraints. Split-K trades more parallelism against partial-output traffic and reduction work. Persistence/TMA are mechanisms to evaluate, not automatic wins.

Examples from our final scored run (microseconds):

| Operation / M | Current | Reference floor |
|---|---:|---:|
| QKV / 32 | 14.24 | 8.00 |
| Pooler / 32 | 20.24 | 14.14 |
| QKV / 128 | 16.74 | 8.42 |
| Pooler / 128 | 22.78 | 14.64 |
| Pooler / 256 | 29.77 | 17.37 |
| Unpooler / 256 | 24.07 | 17.37 |

This makes QKV plus pooler at M=8/16/32/128/256 the primary work, with large unpooler next. Router-only tuning cannot close the gap.

## Priority 1: explicit cuBLASLt selection

Hypothesis: default PyTorch/SGLang heuristics leave meaningful performance on the table for these narrow-M shapes. The prepared-transpose experiment already demonstrates layout sensitivity.

Build a minimal native wrapper using only APIs available in the judge's pinned image. Use FP32 accumulation with BF16 inputs/output and graph-safe workspace management. Compare legal algorithm choices, layouts, workspace sizes, and split-K/reduction choices. Bind choices to public shape metadata, not input values or hidden test identities. Keep a general fallback for nearby shapes.

Query/configure descriptors outside captured execution, but do not assume host-side heuristic overhead explains measured gains: the judge times CUDA graph replay, which already removes most host overhead. The goal is selecting better GPU kernels.

Do not run an unbounded search inside the judge or turn initialization into a quota workaround. Each official check evaluates a bounded, reasoned candidate. Library-version-specific algorithm choices must be validated against the actual installed library; current documentation may expose newer APIs.

First gate: improve dominant dense shapes by at least roughly 10% beyond noise without sacrificing correctness. If there are gains, use the per-load model to choose the next algorithm family. Do not claim this alone guarantees 1.59x.

## Priority 2: specialized Hopper kernels

If library tuning leaves enough gap, use native CUDA/CUTLASS/CuTe where needed, rather than another blanket Triton matmul replacement.

- M=1: retain the existing near-bandwidth GEMV unless a measured improvement emerges.
- M=8/16/32: design skinny GEMM around weight reuse, coalesced/vectorized loads, sufficient resident work, and bounded split-K reduction cost.
- M=128/256: test larger compute tiles, pipelined global-to-shared movement, and asynchronous tensor-core work. Evaluate TMA/WGMMA and Stream-K/persistent scheduling only where the work grid and reuse support them.
- Focus on QKV and pooler first, then reuse applicable kernels on unpooler.

For each candidate, inspect compiler register/spill/shared-memory information where available through the supported environment and kernel profile. Our earlier latency profiles do NOT establish register spills or occupancy as the cause of regressions; those remain hypotheses until measured. Do not assume Nsight access is provided by the service.

## Priority 3: redesigned lossless compression, conditional

Our first compression implementation failed on performance, not correctness. It does not prove all lossless codecs are ineffective on H200.

DFloat11 motivates exponent redundancy; ZipServ emphasizes a decoder designed for tensor-core consumption. A new attempt needs vectorized packed loads, cheap fixed-width decode into the consumer's layout, exact escapes, low register pressure, and no full BF16 intermediate in global memory. Native CUDA/PTX may be appropriate, but has no automatic advantage without these properties.

Break-even example: a 64 MiB BF16 matrix needs about 14 us at 4.8 TB/s. Reducing its common representation to 12 bits offers about 3.5 us of ideal weight-transfer savings before escape traffic. If decode, extra instructions, layout conversions, and spills consume more than that, it loses. Effective bytes must include escape cache-line amplification and metadata, not just nominal packed size.

Gate the branch on a cost model, exact bit-pattern tests, measured compression/escape statistics from supplied weights, and a plausible compiler resource footprint. If those fail, reallocate its checks to the library/native-kernel paths. Never trade precision for leaderboard speed.

## What changes in experimentation

The previous pass spent too many runs on generic all-shape replacements and compiler failures. Future candidates need:

1. One stated hardware bottleneck and a falsifiable mechanism.
2. A predicted score contribution using measured within-load time shares.
3. Static/source validation and CPU tests where meaningful. Investigate whether target-only compilation can run locally; do not assume the Mac can validate CUDA or that a compatible Linux toolchain is available.
4. One primary architectural change per experiment; use shape-level profiles to distinguish improvements from aggregate noise.
5. A kill criterion. After two correct implementations substantially lose for the same mechanism, stop unless new evidence explains how to fix it.
6. Correctness on fresh inputs, permitted precision, graph capture, nearby shapes and output lifetimes. Don't overfit the displayed scored cases.
7. Integrate only measured winners; check the integrated version and independently confirm through scored submission.

Suggested next daily allowance of 20 checks, adaptable to evidence:
- 2 diagnostic/baseline candidates with useful compiler metadata.
- 6 cuBLASLt configuration candidates.
- 6 targeted native Hopper candidates.
- 3 compression candidates only if the break-even gate passes; otherwise reallocate.
- 3 integration/correctness/confirmation candidates.

Daily check reset is 00:00 UTC = 05:30 IST; the next reset is October 31. This plan does not schedule an automatic run. Recheck actual CLI quotas before execution. No paid GPU resources are needed to pursue these experiments, but limited feedback makes top-ten success uncertain.

## Evidence and reading

- NVIDIA cuBLAS documentation: https://docs.nvidia.com/cuda/cublas/index.html — explicit cuBLASLt heuristics, algorithm configuration, workspace, numerical requirements, graph support. Match API usage to the installed judge version.
- NVIDIA library samples: https://github.com/NVIDIA/CUDALibrarySamples — official implementation patterns; reuse only compatible code with required licensing/attribution.
- NVIDIA Hopper tuning guide: https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html — TMA, occupancy, register/shared-memory constraints, memory access.
- Stream-K (Osama et al., 2023): https://arxiv.org/abs/2301.03598 — work-centric load balancing across GEMM shapes; reported paper gains are not predictions for our workload.
- ZipServ (Fan et al., 2026): https://arxiv.org/abs/2603.17435 — fused lossless decompression/GEMM and layout-aware encoding; evaluate hardware/workload differences.
- DFloat11 (Zhang et al., 2025): https://arxiv.org/abs/2504.11651 — lossless BF16 exponent coding, primarily capacity/offloading motivation.
- Public participant report: https://github.com/NathenFernandes/muse-bf16-proj — reports 1.5186x using a mix of native Hopper kernels, exact packing, and selected cuBLASLt algorithms. Supports investigating a hybrid approach; not independent proof of performance or current top-ten status. No participant source was copied into our submission during this planning pass.

Decision: a hybrid of tuned library kernels and a few specialized kernels is the highest-priority path. Compression is a conditional research branch. Target 1.65x with explicit per-load time budgets; stop spending the main budget on router micro-tuning. Top ten remains an ambitious outcome, not a commitment we can substantiate yet.
