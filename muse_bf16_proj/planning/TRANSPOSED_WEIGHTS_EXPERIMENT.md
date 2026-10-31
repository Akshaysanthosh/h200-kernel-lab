> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# Transposed dense weights experiment — 2026-10-30

## Official validation

Live official quota: 20/20 checks used, zero remaining. No scored attempt substituted. Existing active one-time automation `free-cublaslt-checks-after-quota-reset` already prioritizes the protected public-derived development best at 2026-10-31 05:35 IST, after the 05:30 reset. Its existing gates require correctness, no integrity flags, confirmed improvement, and preserving both best versions. No duplicate automation created. Official score of development best remains unconfirmed.

## Hypothesis and implementation

The best cuBLASLt pooler/unpooler paths use contiguous W.T, whereas the native scheduling experiment used contiguous W. Test that physical orientation difference with exact BF16 weights, same FP32 accumulation, and unchanged logical computation W @ x.T. The lab transposes weights before timing and asserts bit-exact reconstruction. There is no runtime weight transposition, compression, or approximation.

`modal_lab/transposed_schedule.cu` derives from the preserved dense scheduling experiment. Packed variants change A to column-major, weight leading stride to N, and extract N/K from [K,N] storage. CUTLASS selects the matching shared-memory/TMA layout. Token activations remain unchanged. Configurations 0–7 retain the prior tile, stage, cluster and scheduler settings. Configurations 8–9 reproduce original-layout cooperative controls (128x64 and 128x128, automatic stages).

Ten configurations x two swizzle bounds x two raster orders x four shapes = 160 screened cases. Shapes: pooler/unpooler M128/M256. Independent weight seed repeats the selected per-shape setting without retuning. Fresh graph inputs and nearby shapes remain required. Reference comparisons include outliers and one-hot rows. CUDA graphs include required Stream-K workspace initialization, with cold-L2 read flush outside timed events. No official-score inference from raw local timings.

## Budget and protection

Before run: $24.16 free Modal credits, $0 charges, $0 spending cap directly verified. One H200, CUDA13 challenge image, max 900 seconds, no retries or persistent volumes. Both protected hashes match `results/local/format-experiment-protected.json`. No change to either best source.

## Outcome: rejected, no submission

All 160 screen cases passed reference correctness. Eight finalists passed fresh-input graph checks; four selected settings passed independent-weight validation, plus eight nearby-shape checks. No selected setting beat the protected baseline.

| Shape | Selected config | Parent us | Candidate us | Parent/candidate | 95% interval |
|---|---:|---:|---:|---:|---|
| pooler M128 | 7 | 24.896 | 36.656 | 0.6761 | [0.6714, 0.6849] |
| pooler M256 | 7 | 28.000 | 44.992 | 0.6277 | [0.6234, 0.6322] |
| unpooler M128 | 8 | 22.912 | 27.616 | 0.8279 | [0.8207, 0.8368] |
| unpooler M256 | 4 | 27.552 | 33.936 | 0.8131 | [0.8061, 0.8253] |

Config 8 is the original-layout control; it won on unpooler M128. Other selected settings use transposed weights. The native-kernel gap remains; changing orientation alone did not resolve it. A/A controls ranged 0.9885–1.0016. No full-workload promotion test warranted, no official check consumed, no candidate promoted. Both protected hashes and the CUDA source hash match.

Raw result: `results/modal/20261030T140015Z_nathen_v18_linear_transposed_schedule.json`. Official validation of the protected public-derived best is still pending the existing scheduled reset follow-up.

Post-run dashboard: $23.87 free credits remaining, $0 charges and $0 spending cap (reporting may lag).
