> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# QKV control audit and focused retest — 2026-10-31

## Source-level cause of prior discrepancy

The previous experiment wrongly described the restored M128 Lt plan as an identical-route control for the protected parent. `candidate_v14` intercepts QKV M128 and invokes `tt_v14_qkv_run_qkvbig` with route index 0. The saved Lt M128 configuration exists in preparation but is not the production route. M256 does use `tt_lt_run_cublaslt`. The official scored profile independently identifies `qb_4d509bcf5bddaa48::qb_kernel` at M128, versus `nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN` at M256.

Thus the main 6–8% M128 difference is a comparison between two different algorithms, not evidence of an unexplained identical-kernel timing bias. Prior timing data remain measurements of the compared paths; the route attribution and its uncertainty needed correction. No source changes to the champion are required.

## Runtime validation design

- Full parent A/A graphs.
- Full parent versus direct actual route (custom WGMMA at 128, Lt at 256).
- Full parent versus saved Lt plan sharing the parent's graph output.
- Saved Lt versus modified-extension Lt, with exactly the same weight, output and workspace pointers, descriptor fields and algorithm.
- First seed CUDA traces for every diagnostic and retest; cuBLASLt version and saved-plan metadata recorded.
- Repeat diagnostics and selected geometry cases on independent weight seeds 82731 and 294817.

Focused geometry retest restores the best previous configuration for widths 2304/2560/3072/4096 at each M, without new autotuning. Weights are bit-exact, padded rows zero, original 2304 columns returned as a view. Same workspace as protected Lt is used. Also evaluate one already-compiled custom QKV alternative per shape (index 1 for M128; index 2 for M256).

100 paired diagnostic replays; 120 per retest; reversed order, 128MiB cold-L2 read flush outside timing events, fresh-input correctness, outlier and one-hot probes. These are development timings, not official scores. Only repeated gains warrant full-workload/nearby-shape checks and official validation.

Before run: $23.68 free Modal credits, $0 charges, $0 cap. One H200, 900 seconds maximum, no retries/persistent volumes. Both old/new champion hashes and public source verified unchanged. The official best remains 1.5206695978723808x, job 19ffcee8e7894be2.

## Runtime outcome: discrepancy explained, no improvement

cuBLASLt version 130101 matches the protected fixed-plan guard. CUDA traces show the full parent and actual-route controls use the same kernel at each M: custom qb_kernel at 128 and nvjet 72x64 TNN at 256. The saved Lt M128 plan instead launches nvjet 48x64 TNN. Both original and modified Lt wrappers launch that same 48x64 kernel with identical buffers.

| Diagnostic | M | Seed 82731 ratio | Seed 294817 ratio |
|---|---:|---:|---:|
| full_parent_AA | 128 | 0.99556 | 0.99780 |
| actual_route_control | 128 | 0.99157 | 1.00353 |
| saved_lt_same_output | 128 | 0.91793 | 0.90825 |
| extension_same_pointers_vs_saved_lt | 128 | 0.99631 | 0.99470 |
| full_parent_AA | 256 | 1.00334 | 0.99226 |
| actual_route_control | 256 | 0.98980 | 1.00039 |
| saved_lt_same_output | 256 | 0.99889 | 0.99963 |
| extension_same_pointers_vs_saved_lt | 256 | 0.99816 | 1.00598 |

Ratios are first path / second path; larger is faster for the second path. The different M128 Lt algorithm accounts for the main gap. Residual ~1% differences in matching-route controls are small and do not support claims of new gains.

All 16 diagnostic comparisons and 20 retests passed reference correctness. Every retest also passed a fresh-input replay check. Geometry retests cover the selected best configuration at each width, two M values and two weight seeds; native alternatives add four cases.

| M | Variant | Seed 1 ratio | Seed 2 ratio |
|---|---|---:|---:|
| 128 | 2304 | 0.90411 | 0.90064 |
| 128 | 2560 | 0.87113 | 0.89393 |
| 128 | 3072 | 0.85785 | 0.87234 |
| 128 | 4096 | 0.72762 | 0.72554 |
| 128 | native alternative | 0.99428 | 0.99647 |
| 256 | 2304 | 0.98924 | 0.98700 |
| 256 | 2560 | 0.93024 | 0.92151 |
| 256 | 3072 | 0.82148 | 0.81369 |
| 256 | 4096 | 0.71395 | 0.70576 |
| 256 | native alternative | 0.87253 | 0.87141 |

No variant demonstrates a repeated gain. No full-workload promotion test, official check or scored submission spent. Current official 1.5206695978723808x champion and previous 1.2274x version remain unchanged. Raw evidence: results/modal/20261031T055046Z_nathen_v18_linear_qkv_control.json.

Post-run dashboard: $23.52 free credits, $0 charges, $0 spend cap; reporting may lag. All Modal apps stopped.
