> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# Compressed-kernel profiling and targeted optimization — 2026-10-30

## Scope and protected sources

Profile the best local public-derived baseline and test lower decoder overhead and register operand reuse. Preserve the official 1.2274118693× source (SHA256 `9ea10c81639eb23d6c9eb832465190328664fd655e89e9ff81732b94e10b8a36`) and public parent (SHA256 `6faa2ae8519e3b9c19b27c2fdf47a2a057c7be89c72583ecfd270175c2144ae7`). Neither was edited.

## Profile

Raw: `results/modal/20261030T124102Z_nathen_v18_linear_profile_compressed.json`.

Challenge image, NVIDIA H200, PyTorch 2.14.1+cu130. Cold-L2 graph timing uses 128 MiB read flush and internal external events. The separate CUDA activity trace identifies device kernels; these two timing series must not be conflated or translated directly into an official score.

| Layer | M | Graph median µs | Device kernel trace median µs |
|---|---:|---:|---:|
| pooler | 1 | 17.472 | 13.921 |
| pooler | 8 | 17.744 | 13.536 |
| pooler | 16 | 17.792 | 13.697 |
| pooler | 32 | 20.384 | 16.224 |
| QKV | 1 | 13.760 | 9.633 |
| QKV | 16 | 14.272 | 9.951 |
| QKV | 32 | 16.096 | 11.936 |
| unpooler | 16 | 17.696 | 13.953 |
| unpooler | 32 | 19.616 | 15.295 |
| O | 8 | 9.344 | 5.056 |

Pooler1 uses 58 registers/thread, no local-memory spill, 1024 threads/block, and only one active block/SM. Packed weights occupy 50,331,648 bytes, with at most six exceptions per stream group. O8 uses 48 registers/thread, no spill, five active blocks/SM. Occupancy limits motivate a smaller-block experiment; they do not prove occupancy is the limiting performance factor.

Nsight Compute exists at `/usr/local/cuda/bin/ncu`. Its first run failed because GPU clock locking was unavailable. The targeted experiment retries with `--clock-control none`, without changing the GPU clock configuration.

## Targeted experiments

1. **Arithmetic decoder:** transplant the parent's existing v18d decoder into v18s with matching v18d lossless packing. It replaces two lookup PRMTs and sign assembly with integer rotate/mask/add reconstruction. Escape patching restores exact original bits before BF16×BF16 FP32 MMA. CPU audit covers 15,488 code-position cases and 15,488 escape-high-byte cases.
2. **Scheduling and occupancy:** compare stage depths and delayed activation loading, then smaller pooler blocks. Hold packing/geometry fixed when isolating decoder effects. Diagnostic kernel modes remain zero; all measured candidates compute full outputs.
3. **Pooler32 register operand reuse:** enable the existing q-outer/n-inner HMMA loop for XM7 BF16 only, preserving each accumulator's K order. This is one changed source line. It attempts to reuse weight registers across successive token-group instructions. It does not reduce global-memory bytes.

Audit finding: current small-batch kernels already share each decoded weight across the whole token batch; QKV128 also covers all 128 tokens. Merely increasing token tile size cannot eliminate nonexistent redundant weight reads.

Screening uses outliers and fresh graph inputs, same-GPU cold-L2 paired reversed-order timing, 20 repetitions for the sweep, 80 for finalists, 100 for reuse. Winners require independent weights and full correctness/timing validation. Hardware-counter and SASS findings must be distinguished from source-level hypotheses.

## Hardware counters and screening results

The retry succeeded (`ncu_returncode=0`). CSV saved to `results/modal/compressed-pooler1-ncu.csv`. Pooler1 DRAM throughput was 73.82%, L2 throughput 71.13%, compute throughput 30.52%. Nsight attributed 77.6% of average warp issue delay to scoreboard dependencies on L1TEX operations (20.0 of 25.8 cycles). This supports a memory-dependency bottleneck for this profiled path; it does not mean 77.6% of kernel runtime is removable. Counters used unlocked clocks and kernel replay, distinct from acceptance timings. Optional PC sampling was absent.

Decoder sweep raw: `results/modal/20261030T124358Z_nathen_v18_linear_tune_decode.json`.

- 57 configurations passed output and fresh-graph checks; spilling/unsupported configurations were excluded before timing.
- Pooler1 arithmetic/scheduling finalist: paired ratio 1.01058×; identical-geometry arithmetic OPT6 only 1.00136×. Smaller blocks increased theoretical occupancy but did not improve timing.
- O8 arithmetic, ST1/OPT4: 1.01926×. This small signal is selected for independent full-workload validation, not promotion.
- QKV1 finalist: 1.00528×, insufficient evidence of a gain.
- Pooler32 register reorder: 0.99765× (20.416 µs parent / 20.464 µs candidate median), no gain. Compiled instructions contain weight `.reuse` flags, but candidate-only flags do not establish increased reuse relative to the parent.

Follow-up: two independent weight sets and all 30 scored shapes for the separate `candidates/public/v18_arithmetic_o8.py`, then four weight-load policies (including unchanged control) on a third weight seed. This directly tests whether bypassing L1 or changing L2 prefetch size helps the memory-dependent pooler path.

## Final decision: no promotion or submission

Independent validation raw: `results/modal/20261030T124744Z_v18_arithmetic_o8_validate_decode.json`.

The O8 candidate passed all 63 correctness cases on each of two independent weight sets, with normal/outlier inputs, nearby shapes, retained outputs, and fresh graph inputs. Aggregate candidate/parent speedup was **1.0010316563×** on seed 82731 and **0.9958302072×** on seed 294817. Identical-parent A/A controls were 0.9986396036× and 0.9967281916×. There is no repeatable aggregate improvement. O8 alone measured 1.00833× and 1.01002×; that small signal does not justify promotion.

Memory-policy results on the third weight seed:

| Weight load policy | Paired speedup vs parent |
|---|---:|
| Unchanged no-allocate control | 1.00708× |
| L2 prefetch 128B | 0.97520× |
| L2 prefetch 256B | 0.98513× |
| Global-cache/L1 bypass | 1.00717× |

The L1-bypass result is indistinguishable from the unchanged control. Both prefetch hints lose. All policy checks passed. No new candidate qualified for official validation or submission; preserve the existing scheduled priority for the public parent.

The provisional arithmetic candidate is retained only as a rejected experiment. Both protected best hashes were rechecked. All GPU apps stopped. Dashboard moved from $25.55 to $24.93 remaining free credits (~$0.62 usage; reporting may lag), with $0 charges and $0 spending cap.

Conclusion: on the measured pooler1 path, memory dependencies dominate and these decoder/scheduling/cache changes do not overcome them. The existing kernels already reuse weights across their token tile. A larger improvement would require reducing actual losslessly encoded weight bytes or another structural change, with decode costs measured rather than assumed.
