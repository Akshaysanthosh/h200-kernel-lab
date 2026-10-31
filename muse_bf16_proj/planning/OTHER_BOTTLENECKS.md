> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# Beyond pooler/qkv: research and experiments — October 30, 2026

## Target and physical headroom

The target is 40% less whole-workload execution time than the official 1.2274118693x
champion, equivalent to approximately 2.046x SGLang. Using the official profile's
per-shape times and replacing selected layers by the challenge's stated bandwidth/compute
floors gives these optimistic reductions in geometric mean load time:

| Layers taken to their stated floor | Whole-workload reduction |
|---|---:|
| Router only | 8.04% |
| Unpooler only | 6.56% |
| Output projection only | 4.97% |
| Router + unpooler + output | 19.59% |
| Pooler + qkv only | 19.83% |
| All five layers | 39.53% |

These are counterfactual estimates, not measured speedups or rigorous bounds for compressed
representations. They assume ordinary data movement and omit practical efficiency losses.
Consequently 40% is approximately the ideal-hardware ceiling for uncompressed tuning.
It is not a reasonable promised milestone. End-to-end gains must be calculated per load,
not by averaging individual kernel speedups or adding the percentages above.

## Research conclusions

- CUTLASS explains the occupancy/reuse trade-off and why split-K helps skinny grids:
  https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md
  Removing a reduction launch can hurt if the remaining output grid underfills the GPU.
- The H100 TMA study shows the load mechanism should depend on the operand/workload:
  https://arxiv.org/html/2610.34907v1
  It supports testing ordinary loads, not assuming every custom kernel should use TMA.
- DFloat11 reduces weight storage about 30% without precision loss, but its reported
  large throughput benefits compare against CPU offloading, not our resident H200 baseline:
  https://arxiv.org/abs/2504.11651
- Unweight reconstructs losslessly encoded BF16 tiles before Hopper WGMMA consumption.
  Its engineering results do not establish a ready-made speedup over cuBLAS:
  https://research.cloudflare.com/papers/unweight-2026.pdf
  A codec must save more HBM time than its decoding and pipeline overhead consume.

## Experiment 1: direct loads and complete-output reductions

Original Triton implementations test direct-load matrix multiplication, both operand
orientations, narrow tiles, split-K 1/4/16, and complete-output FP32 vector reductions.
Router retains FP32 weights for vectors and allowed hi+lo BF16 for tensor cores.
The high and low accumulators are separate in the revised version.

The initial search stopped on a router accuracy failure. The revised external tuner records
and excludes every accuracy failure; it never relaxes tolerances. Across 246 configurations
on 15 shapes, three configurations failed strict checks. The best small routers improved
about 4% on batches16/32 but much less overall. Full-reduction single-kernel versions lost:
removing the launch did not compensate for lost parallelism. Unpooler/output did not yield
material improvements, and large custom variants were substantially slower than libraries.

Evidence: results/modal/20261030T094841Z_other_search_other.json; external tester
modal_lab/tune_other.py. No candidate from this sweep replaces either best version.

## Experiment 2: fused lossless weight decoding

A bounded prototype packs each BF16 value's sign/mantissa into eight bits and its exponent
into a four-bit offset within 256-element blocks. Blocks outside the exponent range use
original BF16 values. The consumer reconstructs values inside the GEMV/GEMM, with FP32
accumulation. No full-weight decompression is moved outside timed calls.

The development evaluator independently verifies bit-for-bit reconstruction of every weight.
All four dense matrices passed; packed streams are 75.195% of original weight bytes and
about 1.2–1.5% of blocks require exact raw fallback. The prototype retains original tensors,
so this is a traffic-reduction experiment, NOT a claim of reduced total allocated memory.
Fallback and decoder overhead must be included in timing. No lossy casts or input-result
caching are used. Only speed and correctness-qualified paths may ever be promoted.

### Packed prototype result

72 configurations were attempted. Four output-projection configurations had invalid
K/(splitK*BK) geometry and were rejected by correctness; the external tuner now filters
those configurations before launch. The other 68 passed the trial checks but were slower.
Best confirmed speed ratios versus champion ranged from 0.073 to 0.644 across 12 shapes
(less than 1 means slower). Matrix consumers reached 246–255 registers per thread;
packing is bit-exact but this consumer is not competitive. These results reject this
implementation, not every possible lossless approach. The original weights remain
resident for exact fallback, so allocation savings were never established.

Evidence: results/modal/20261030T095114Z_packed_search_packed.json.

## Decision and next engineering gate

Keep the official champion and the previous Hopper hybrid unchanged. No new candidate
qualifies for promotion or official quota. The scheduled official validation still starts
with candidates/hopper/hybrid_v1.py; do not test these rejected search scaffolds.

To pursue a substantially larger gain, change the execution pipeline rather than continue
this tile search: a native CUDA/CUTLASS producer-consumer pipeline could decode packed
weights into shared memory while consumer warps execute WGMMA, with regular handling
of exceptional blocks. This is a research hypothesis, not a promised speedup. First prove
that decode-plus-consume beats the uncompressed consumer on one dominant shape by at
least 10%, with exact reconstruction and cold-L2 measurements, before expanding the
implementation. The current direct Triton decoder fails that gate by a wide margin.

For ordinary kernels, use a smaller measured milestone such as 5–10% overall. A 40% goal
requires approaching all the reference floors or reducing physical memory traffic beyond
their assumptions. Neither the current results nor the papers establish that we can do it.

Final billing verification: $29.04 free credits remaining; $0.96 credits used to date;
$0 net charges and $0 spend limit. Billing may lag. All apps were confirmed stopped.
