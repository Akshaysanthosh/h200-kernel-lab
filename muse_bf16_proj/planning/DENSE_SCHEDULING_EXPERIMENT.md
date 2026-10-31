> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# Dense Hopper scheduling experiment — 2026-10-30

## Hypothesis

Pooler and unpooler account for about 54% of execution time in each of the M128/M256 loads. The protected pooler M256 profile measured 69.92% compute throughput, 58.33% DRAM throughput and substantial scoreboard waits. Those measurements suggest mixed compute utilization and memory latency/scheduling constraints; they do not prove that one mechanism is the sole bottleneck.

Test native CUTLASS TMA producer/consumer scheduling without decoding. This differs from the rejected compressed WGMMA pipeline (register decoding overlapped with one outstanding MMA group) and the early cp.async small-M CUTLASS trial. Dense BF16 operands are unchanged; FP32 accumulation and BF16 output are preserved.

Source: local NVIDIA CUTLASS 3.9.2 headers and example 49, plus https://github.com/NVIDIA/cutlass/blob/main/examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu . CUTLASS maintains shared-memory buffer and asynchronous matrix-operand lifetimes.

## Variants

Logical matrix multiplication is W @ x.T, with column-major output viewed as row-major [tokens, outputs]. K tile is 64. Each kernel directly produces BF16 output; Stream-K uses CUTLASS's deterministic FP32 partial accumulation and required workspace initialization inside the timed graph.

| ID | Tile M,N | Stages | Cluster | Schedule |
|---|---|---|---|---|
| 0 | 64,64 | 3 | 1,1 | Persistent ping-pong |
| 1 | 64,64 | auto, epilogue carveout | 1,1 | Persistent ping-pong |
| 2 | 128,64 | 3 | 1,1 | Persistent cooperative |
| 3 | 128,64 | auto | 1,1 | Persistent cooperative |
| 4 | 128,128 | auto | 1,1 | Persistent cooperative |
| 5 | 128,64 | auto | 2,1 | Persistent cooperative + multicast |
| 6 | 128,64 | auto | 1,1 | Cooperative, Stream-K heuristic |
| 7 | 128,128 | auto | 1,1 | Cooperative, Stream-K heuristic |

Each tests two raster orders and swizzle bounds 1 and 4. Four target shapes: pooler/unpooler M128/M256. A heuristic Stream-K scheduler may choose data-parallel decomposition where it estimates that to be preferable; its name alone does not prove use of a split reduction.

## Gates

- Before run: $24.45 free credits, $0 charges, $0 spend cap directly verified.
- One H200, CUDA13 challenge image, 900-second timeout, no retries, no persistent volumes.
- Both protected source hashes verified against results/local/format-experiment-protected.json.
- Reference correctness for every configuration before timing. Fresh-input CUDA graph replay for finalists, independent CUDA weight seed, nearby shape checks.
- Cold-L2 128MiB read flush outside events, alternating order, paired-ratio medians, bootstrap 95% intervals; A/A controls. Raw graph events, not an official judge score.
- Screen 12 samples, top two per shape 80 samples, independent seed 100 samples without retuning.
- Only repeatable gains justify building a submission candidate and full-workload validation. No promotion based on an isolated best sample.

## Outcome: reject, no promotion or submission

All 128 screened configurations passed correctness. Eight finalists passed fresh-input graph correctness; four selected settings were rechecked on independent weights, and all eight nearby-shape checks passed. Compilation took 113.2 seconds.

| Shape | Best config | Parent us | Candidate us | Parent/candidate | 95% interval |
|---|---:|---:|---:|---:|---|
| pooler M128 | 6 | 23.872 | 37.744 | 0.6338 | [0.6318, 0.6400] |
| pooler M256 | 7 | 28.160 | 45.408 | 0.6235 | [0.6203, 0.6283] |
| unpooler M128 | 3 | 23.008 | 27.888 | 0.8196 | [0.8124, 0.8278] |
| unpooler M256 | 4 | 27.712 | 34.016 | 0.8150 | [0.8081, 0.8215] |

None is competitive. Stream-K heuristic variants were the strongest pooler settings, and deep cooperative pipelines were strongest on unpooler, but all lost by large margins. These results reject these tested native scheduling variants, not asynchronous overlap as a general technique. cuBLASLt already uses tuned native kernels; adding warp specialization alone is not sufficient. No claim of a specific hardware cause for the slowdown without candidate counter profiles.

No full-workload promotion validation or official check was spent on these losing kernels. Both protected source hashes remain unchanged. Raw result: `results/modal/20261030T134903Z_nathen_v18_linear_dense_schedule.json`.

A/A ratios ranged 0.9918–1.0282, so small percentage gains would need additional controls; the observed 18–38% throughput losses are much larger. Increasing pipeline depth helped compared with three stages, but did not reach baseline performance.

Post-run dashboard shows $24.16 free credits, $0 charges, and $0 spending cap; billing reporting may lag.
