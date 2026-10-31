> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# Optimization experiments — 2026-10-30

Budget: official challenge GPU evaluations only; no paid GPU rentals.

## Measured bottlenecks

The starter router at m=256 took 45.45 us against SGLang at 17.54 us. Other projections were approximately at baseline.

## Experiments

| Variant | Change | Passed | Overall speedup | Job |
|---|---|---|---|---|
| v01 | Fused BF16 hi/lo router, split-K 8 | True | 1.0809x | 1ec842ce12994e91 |
| v02 | Router split-K 32 for small batches, 4 for large | True | 1.0996x | 043668b09a5b48b7 |
| v03 | Direct one-token GEMV; large-router split-K 16 | True | 1.1134x | 1129663419614fd0 |

| v04 | Select measured winners by batch size | True | 1.1355x | f740c6845bfb42e1 |

## Interpretation

All speedups refer only to the muse_bf16_proj benchmark, not full-model serving. Each check includes correctness and paired CUDA-graph timing on H200. Reported intervals account for measurement noise. Small differences between experimental scores may be noise.

## Implementation boundaries

Router FP32 weights are represented with the explicitly permitted BF16 high/low pair and FP32 accumulation. Other weights and activations keep their required precision. No harness changes, cached outputs, network calls, or input-dependent host synchronization. Fresh output buffers are allocated per invocation. Original starter and every experimental version are preserved under experiments/.

## Scored verification

Job 3a776756bf404e0b passed all 9 tests with no flags. Speedup 1.1274395823432366x, 95% interval [1.110527988608088, 1.144351176078385]. The scored rerun independently confirms the earlier check improvement.

Research follow-up: see RESEARCH.md for v05–v19 and the final scored 1.227412x result, rank 766.
