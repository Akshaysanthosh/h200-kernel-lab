> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# Dense GEMM bottleneck and weight-stride experiment — 2026-10-30

## Ranking

`results/local/bottleneck-ranking.json` derives per-load shares and first-order log-score sensitivity from two independent parent timing rounds. These are local raw graph timings, not official overhead-subtracted judge times. Shape weights are all equal and cancel within each load.

Pooler is the largest individual cost (roughly 26–31% per load). At M=128/256, pooler plus unpooler account for ~54% of measured load time, and run dense cuBLASLt kernels rather than our small-M compressed decoder. Representative graph durations: pooler M128 ~24.3–24.7us, M256 ~28.3–28.5us; unpooler M128 ~23.0–23.1us, M256 ~27.9–28.1us. This establishes where time is spent, not the limiting hardware resource.

## Different approach

Repack dense BF16 weights with a padded leading dimension. Change only the weight matrix descriptor's physical stride; preserve logical dimensions, transpose flags, FP32 accumulation, BF16 output, and exact weights. Padding is prepared untimed. Input/output access and launch count are unchanged. This tests whether the power-of-two weight stride is harming memory access efficiency or kernel selection. It is a hypothesis, not a claim of demonstrated memory-partition contention.

The protected parent extension assumes tightly packed weights. A separate C++ extension uses `weight.stride(0)` in its two weight descriptors and distinct C++ class names to avoid pybind collisions. Both best source files remain unchanged.

Padding candidates: 0, 8, 16, 32, 64, 128 BF16 elements per stored row. Start from each shape's existing packed orientation and saved algorithm, plus up to 12 supported heuristic alternatives. No further broad tile-parameter sweep. New bytes are filled with NaNs in the lab to expose accidental reads of padding; logical weights compare bit-exactly.

## Measurement

- One H200, challenge CUDA 13 image, maximum 900 seconds, no retries or persistent volume.
- Before-run dashboard: $24.66 free credits, $0 charges, $0 cap.
- Four high-cost shapes, normal/outlier and fresh-input graph correctness, alternating cold-L2 read-flushed timing, A/A controls.
- Screening uses 12 samples; finalists 80; selected configurations repeat with 100 samples on independent CUDA weights without retuning.
- CUDA trace records actual kernel names and durations. Nsight Compute profiles pooler M256 with clock control disabled; replay counters are diagnostic, not substituted for paired graph timings.
- Promotion still requires full-workload independent validation and official passing checks. No unconfirmed submission.

## Outcome

**Rejected: no confirmed improvement; no promotion or submission.**

Raw result: `results/modal/20261030T134123Z_nathen_v18_linear_padded_layout.json`. All 258 supported layout/algorithm screen cases passed reference correctness. Finalists also passed changed-input graph checks, and four selected configurations were rechecked with independent weights without retuning.

| Shape | Selected padding | Second-seed speedup | Paired 95% interval |
| --- | ---: | ---: | --- |
| pooler M128 | 128 elements | 1.01254 | [0.99632, 1.02306] |
| pooler M256 | 0 (control) | 0.99215 | [0.98421, 1.00916] |
| unpooler M128 | 0 (control) | 0.99648 | [0.98250, 1.00910] |
| unpooler M256 | 0 (control) | 0.99942 | [0.99368, 1.00868] |

No padded finalist demonstrated a repeated interval excluding parity. A/A controls ranged 0.99154–1.00343. The small apparent gains are insufficient to justify full-workload promotion testing. This rejects the tested stride-padding variants, not every possible data-layout transformation.

## Hardware finding

Nsight Compute completed successfully for the protected pooler M256 kernel `nvjet_sm90_tst_128x64_64x8_1x1_v_bz_NNN`. Raw diagnostic output: `results/modal/dense-pooler256-ncu.csv`.

- Compute (SM) throughput: 69.92%.
- DRAM throughput: 58.33%; L2 throughput: 69.20%.
- Average 12.83 warp cycles per issued instruction; tool reports 6.9 cycles waiting on L1TEX scoreboard dependencies (53.7% of issue-delay cycles).
- Active threads/warp 31.95, non-predicated-off 27.52.
- Profiled duration 26.528us; separate graph trace kernel duration ~23.516us. Do not substitute profiled timing for paired graph results.

Clocks were not locked, and PC stall sampling was unavailable. These metrics do not establish a single exclusive bottleneck, do not prove partition camping, and do not imply 53.7% of execution time is removable. They suggest mixed compute utilization and memory latency/scheduling constraints. Simple physical-stride padding does not fix them.

Four separate graph traces each showed one dense cuBLAS kernel, so eliminating a separate decode or reduction launch is not an opportunity on these routes. Further work should target execution scheduling/overlap and weight reuse in these dense paths, with a concrete difference from previously rejected WGMMA pipelines. Merely reducing weight bytes or changing alignment has not provided gains.

Both protected source hashes remain unchanged. All work is lab-only; no new submission candidate was promoted.

Post-run verification: Modal shows $24.45 free credits remaining, $0 charges and a $0 spending cap (reporting may lag). CLI confirms no active apps; the experiment app stopped with zero tasks.
