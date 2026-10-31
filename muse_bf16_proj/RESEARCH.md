> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# Research-led optimization plan — 2026-10-30

## Sources read

- DFloat11 (Zhang et al., 2025): https://arxiv.org/abs/2504.11651 — low BF16 exponent entropy enables lossless compression. Its capacity/offloading results are not assumed to translate to our H200 microbenchmark.
- ZipServ (Fan et al., ASPLOS 2026): https://arxiv.org/abs/2603.17435 — fuse decompression directly into GEMM; separate decompression can negate bandwidth savings. Hardware and workload differ from this challenge.
- Official Triton matrix multiplication tutorial: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html — tiled dot products, cache reuse, and configuration selection.
- Official Triton persistent matmul tutorial: https://github.com/triton-lang/triton/blob/main/python/tutorials/09-persistent-matmul.py — persistent scheduling and TMA. Blackwell CLC variant is not appropriate for H200.

## Hypotheses

1. Router-only improvements have limited overall impact. Dense projection weight traffic dominates the remaining work.
2. A simple fixed-width lossless exponent codec may reduce traffic enough to improve low-batch projections. Sign + mantissa use eight bits, exponent uses a four-bit offset or an escape. Escapes load the original BF16 value. This is a new simple implementation inspired by the papers, not their codec or a reproduction of their results.
3. Decode inside the consumer kernel, with FP32 accumulation. Never quantize or discard original BF16 bits.
4. Compare compressed and uncompressed tiled split-K GEMMs and retain per-shape winners only when correctness passes. Keep native library fallback for unsupported or slower regions.

## Exactness check

CPU bitwise encode/decode was checked exhaustively across all 65,536 BF16 bit patterns at four exponent bases (0, 110, 120, 240). All reconstructed exactly, including escape values. This validates the format, not GPU kernel execution; official GPU correctness checks remain required.

## Budget

15 official free checks remained at the start of this research pass. No paid GPU or paid API provisioning. Previous scored best source is preserved at experiments/best_before_research.py.

## Measured research pass

| Variant | Passed | Speedup | Job |
|---|---|---|---|
| v05 | False | compiler failure | a239cef06bac4e75 |
| v06 | False | compiler failure | 5e5b2a588ab642ae |
| v07 | True | 0.2917x | 6733d142e2644366 |
| v08 | True | 0.6852x | 882816e9a2b646f2 |
| v09 | True | 1.1500x | d6b58a39cf10444a |
| v10 | True | 0.8036x | 617f6c8e76ae4e18 |
| v11 | True | 1.1894x | 95170932c4684554 |
| v12 | True | 1.0649x | b4b73bae87b9415a |
| v13 | True | 1.1839x | b3192122dbf8499b |
| v14 | True | 0.9479x | 312b969dacbe4c50 |
| v15 | True | 1.2131x | d6949ae288974a2c |
| v16 | True | 1.2045x | b5d7fc3e346e4a68 |
| v17 | True | 1.2251x | d6fcb69d7e614108 |
| v18 | True | 1.2202x | d72fd06fe97f439b |
| v19 | True | 1.2124x | e1eebb0a5a7a41e6 |

## Decision

The exact v17 source is selected for scored submission: it had the highest measured check score, 1.225115x. The cleaned equivalent v19 also passed at 1.212380x; the intervals overlap, so the difference is not strong evidence of a real speed difference. The unmodified v17 is retained to reproduce the best observed result.

Compression passed correctness after a compiler workaround, but both tested consumer implementations were slower. Generic dense Triton matmuls and tile-packed dense matmuls also lost. These are rejected experiments, not features of the selected submission.

Selected implementation:
- Direct FP32-accumulating GEMV for one-token inputs, including the original FP32 router weights. Use one output row per program except the short-K output projection, which uses two.
- Eight warps for long-K GEMV, four for short-K.
- Prepared transposed weights for pooler and larger unpooler GEMMs; native library layout elsewhere.
- Fused BF16 high/low router with FP32 accumulation and reduction. Small batches use 16x32 tiles with 32 K partitions; larger ones use 64x64 tiles and 16 partitions.
- No output caching, reduced-precision BF16 weights, input-value-based dispatch, or harness modification.

The v17 introductory docstring was inherited from an earlier experiment; its claim of a large-router library fallback is stale. The executable code uses the fused router above; this document describes the actual final implementation.

All 15 remaining official free check runs were used. No paid GPU or API resources were provisioned. These gains apply to the projection benchmark, not full-model serving. Source and result snapshots are retained.

## Research pass: final scored result

- Job: 35d40255820f4df4.
- Speedup: 1.227412x SGLang, 95% interval [1.2090006912699522, 1.24582304734924].
- Correctness: 9/9 tests passed; no flags.
- Projection leaderboard rank: 766 (previous recorded rank 1246; provisional live standings).
- Source: submission.py, exact copy of experiments/v17_gemv_warps.py.
- GPU spend: $0. All 15 remaining free checks used; 0 checks and 7 scored submissions remain today.
- Details: RESEARCH.md. Results: results/research-summary.json.
- This is a microbenchmark improvement, not a measured full-model serving gain.
