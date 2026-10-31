> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# Small-batch uncompressed TinyGEMM experiment — 2026-10-30

## Question

Can a specialized small-M kernel beat the lossless compressed public v18 baseline by avoiding decode instructions? This is a different tradeoff from the failed large-tile decode/WGMMA pipeline. The target 1.667× is not an achieved score. Moving from the upstream published 1.5186× to 1.667× would require 8.9% less aggregate execution time, assuming comparable judge measurements. Our adapted baseline still needs official validation.

## Primary research

- [When and Why to Use TMA](https://arxiv.org/abs/2610.34907), submitted 26 October 2026: fixed transfer overhead, reuse, and workload shape determine whether TMA is advantageous. This motivates measuring small-M alternatives; it does not predict this experiment's result.
- [FlashInfer TinyGEMM source](https://github.com/flashinfer-ai/flashinfer/blob/5d60811431d040ed62f85c7a7906a77a6e72cccc/csrc/tinygemm2.cu): audited and pinned. Uses BF16 operands, FP32 mma.sync accumulation, TMA producer warps, and four compute warps reducing K partitions within a CTA. No weight compression. Output tiles are 16 channels by 8 tokens.
- [Unweight technical report](https://research.cloudflare.com/nikulin2026/): another lossless packing direction, but it introduces decoder and format complexity; defer until a measured implementation can justify it.

TinyGEMM still uses TMA. The hypothesis is lower decode and compute-scheduling overhead, not elimination of TMA. Its cost is moving uncompressed weights and potentially rereading weights across token tiles. The original source and license are retained under research/2026-10-30.

## Adaptation and validation

`modal_lab/tinygemm.cu` retains the upstream device kernel, replaces the TVM binding with a raw-pointer launcher, disables bias and programmatic dependent launch, and exposes 4/8/16 pipeline stages. There is no reduced-precision accumulation, output cache, timing-dependent dispatch, or input-dependent host branch. FP32 router paths are excluded.

The lab compares 16 shapes: M=1/8/16/32 for pooler, QKV, unpooler and O. The unpooler respects the reference's sliced output width. Each stage configuration is checked on outlier input and two fresh graph inputs, including probe rows. Timing uses 128 MiB read-only L2 flushes, CUDA graph internal external events, and 40 alternating repetitions. Ratios are paired reversed-order medians, without an invented judge-overhead correction. This is a screening experiment, not a judge-score estimate. Any winner requires independent weights and full-workload validation before official checks.

Protected sources:

- Official submission: SHA256 `9ea10c81639eb23d6c9eb832465190328664fd655e89e9ff81732b94e10b8a36`.
- Public v18 baseline: SHA256 `6faa2ae8519e3b9c19b27c2fdf47a2a057c7be89c72583ecfd270175c2144ae7`.

No changes to either source are needed to run this lab experiment.

## Result: rejected

All 48 configurations (16 shapes × 3 stage counts) passed eager outlier and two fresh-input graph checks (144 checks). No configuration beat the parent. No official check or submission was used.

| Layer | M | Parent µs | Best TinyGEMM µs | Paired ratio (higher is better) |
|---|---:|---:|---:|---:|
| pooler | 1 | 17.792 | 23.216 | 0.7616 |
| pooler | 8 | 17.984 | 23.072 | 0.7660 |
| pooler | 16 | 17.904 | 39.152 | 0.4598 |
| pooler | 32 | 20.448 | 67.264 | 0.3081 |
| qkv_gate | 1 | 13.600 | 17.808 | 0.7662 |
| qkv_gate | 8 | 13.632 | 17.296 | 0.7870 |
| qkv_gate | 16 | 14.512 | 24.640 | 0.5844 |
| qkv_gate | 32 | 16.192 | 36.176 | 0.4505 |
| unpooler | 1 | 8.240 | 9.056 | 0.9177 |
| unpooler | 8 | 8.576 | 9.136 | 0.9370 |
| unpooler | 16 | 17.888 | 40.384 | 0.4437 |
| unpooler | 32 | 19.680 | 69.632 | 0.2844 |
| o | 1 | 9.264 | 11.888 | 0.7892 |
| o | 8 | 9.312 | 11.136 | 0.8407 |
| o | 16 | 9.664 | 13.536 | 0.7127 |
| o | 32 | 10.016 | 19.216 | 0.5225 |

Raw result: `results/modal/20261030T122046Z_nathen_v18_linear_tinygemm.json`. All numbers are local screening measurements, not official judge scores.

Interpretation: the tested uncompressed, narrow-tile kernel cannot replace the existing compressed paths. Losses increase with token count, consistent with poor weight reuse/too many narrow tiles; this is an inference from the shape trend, not a hardware-counter diagnosis. The next useful direction is reducing decoder instructions and improving weight reuse inside the existing lossless format, preceded by profiling instruction stalls and actual memory traffic. Do not spend more credits tuning this TinyGEMM family without a concrete structural change.

Two discarded setup attempts preceded the valid run: a missing pybind header, then a test wrapper that cached the eager CUDA stream outside graph capture. Fresh-input validation caught the latter; no timing from those attempts was accepted. The final wrapper gets the active stream inside the call. Both failed logs are retained for audit.

Free credits: dashboard decreased from $25.97 to $25.55 during these attempts (~$0.42; billing may lag). Paid charges $0, spending cap $0, and all Modal apps stopped. Protected hashes reverified after testing.
