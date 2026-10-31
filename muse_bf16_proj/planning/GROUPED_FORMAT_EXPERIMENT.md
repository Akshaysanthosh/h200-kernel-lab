> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# Grouped lossless-format experiment — 2026-10-30

## Protected sources

- Official submission: `submission.py`, SHA256 `9ea10c81639eb23d6c9eb832465190328664fd655e89e9ff81732b94e10b8a36`.
- Development parent: `candidates/public/nathen_v18_linear.py`, SHA256 `6faa2ae8519e3b9c19b27c2fdf47a2a057c7be89c72583ecfd270175c2144ae7`.
- Existing champion backups remain unchanged. New code is lab-only, not a submission candidate.

## Local analysis

`tools/analyze_weight_formats.py` calls the public `reference.generate_weights` on CPU for seeds 5151, 82731, and 294817, all four BF16 projections at full size. CPU RNG is not CUDA RNG: these are samples of the specified distribution, not held-out judge tensors. GPU experiments independently generate CUDA weights and measure their actual packed storage.

Results: `results/local/weight-format-analysis.json`.

| Representation, including modeled metadata/padding | Pooler bits/weight |
| --- | ---: |
| Current 12-bit payload + fixed per-warp exception allocation | 12.125 |
| Triple-plane layout, separately 16-byte-padded 64-value fragments | ~13.53 |
| Triple-plane layout, 1024-value groups | ~11.351 |
| Regular 11-bit high-byte delta + sparse 32-bit patches | ~11.438 |
| Adaptive global-rank width, 128-value tiles | ~12.257 |
| Adaptive global-rank width, 256-value tiles | ~11.716 |

These describe explicit layouts, not lower bounds or all possible implementations of the papers. For example, less padding or smaller metadata can improve the adaptive layout. The regular high-byte format is close in size but greatly increases the sequential exception count of the existing decoder. Triple-plane reconstruction offers parallel indexed exceptions instead.

Original versus shuffled adaptive-width results differ by much less than 0.01 bits/weight. There is no practically useful evidence of locality in these independent Gaussian samples. This does not rule out distribution-based global compression.

CPU implementation of the selected layout round-trips all 65,536 BF16 bit patterns, including signed zeros, subnormals, infinities, and NaN payloads. It also round-trips the full 33,554,432-weight CPU pooler. Its actual MMA-permuted packing is 11.351716 bits/weight on seed 5151; permutation changes group padding slightly from the row-major estimate.

## Selected GPU test

`modal_lab/zip_format.py` implements the representation independently, inspired by TCA-TBE (https://arxiv.org/html/2603.17435v1); it does not import ZipServ code or claim to reproduce its kernel.

- Seven consecutive full BF16 exponents plus an escape code in three bit planes.
- Common sign/mantissa bytes and exact BF16 exceptions are compacted separately.
- Each 1024-value MMA-order group carries two 32-bit offsets; payloads are 16-byte aligned.
- Warp prefix counts compute common and exception offsets. Exact BF16 registers feed the protected parent's existing MMA, activation, and reduction schedule.
- No weight approximation, output caching, timing modification, or input-dependent host control flow.
- Compare original 1024-thread geometry plus 256- and 512-thread geometries to account for decoder register pressure.
- Two independent CUDA weight seeds; normal/outlier and M=1/7/8 correctness; changed-activation graph replay; 80 alternating cold-L2 paired samples and A/A control. Report raw ratios and a separate judge-style 4-microsecond-subtracted sensitivity calculation. Neither is an official judge score.
- 95% intervals are paired bootstrap intervals; noise controls and the minimum 3% improvement criterion also apply.

The run uses the challenge CUDA 13 container, a single H200, no persistent volume, no retries, and a 900-second timeout. Modal dashboard verified $24.93 remaining free credits, $0 spend cap and $0 charges before launch. No official free checks remain today, so no official validation or submission is attempted without a later passing check.

## Outcome

**Rejected: correct but substantially slower. No promotion or submission.**

Result: `results/modal/20261030T132340Z_nathen_v18_linear_zip_format.json`.

| CUDA seed | Geometry (WR, WK, KU, S) | Parent graph µs | Candidate graph µs | Parent/candidate ratio |
| --- | --- | ---: | ---: | ---: |
| 82731 | 2,16,8,1 | 17.856 | 55.456 | 0.3231 |
| 294817 | 2,16,8,1 | 17.952 | 55.424 | 0.3257 |
| 82731 | 1,8,16,1 | 17.856 | 65.264 | 0.2742 |
| 294817 | 1,8,16,1 | 17.920 | 65.536 | 0.2740 |
| 82731 | 2,8,16,1 | 17.824 | 65.200 | 0.2745 |
| 294817 | 2,8,16,1 | 17.920 | 65.152 | 0.2764 |

All 36 shape/distribution/configuration cases passed, along with full weight round trips and changed-input graph replay. A/A ratios were 0.99955 and 0.99731. Best candidate paired bootstrap intervals were [0.32123, 0.32490] and [0.32342, 0.33098]; this loss is far larger than noise. No full-workload speedup is inferred from this one-shape experiment.

Measured CUDA packed rates: 11.351842 and 11.352178 bits/weight versus 12.125 for the parent including its exception allocation (50,855,936 bytes). Saving ~6.38% of stored bytes does not compensate for reconstructing compacted common/exception streams. The new loader introduces prefix counts, per-element byte gathers and conditional fallback loads. The original geometry uses 64 registers and 16 local bytes/thread; spill-free 256/512-thread variants still lose (~79/80 registers). These observations reject this implementation; no hardware-counter attribution of all lost time was performed.

The application completed and both protected hashes still match. Modal's immediately refreshed dashboard showed $0 charges and the $0 cap; its credit balance still showed $24.93 and may not yet include this short run. Do not interpret the unchanged balance as zero compute usage.

Next direction: avoid further investment in this compacted-value loader. Any future decoder candidate must preserve vectorized accesses or remove a demonstrated dependency, rather than simply reduce stored bits. Official baseline validation remains pending the free-check quota reset.
