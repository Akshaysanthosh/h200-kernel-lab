> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# Research update after H200 experiments — October 30, 2026

## Decision

The highest-upside next engineering direction is shape-specialized native Hopper dense projection
kernels, with the proven champion as fallback. Complete bounded offline library selection first;
keep lossless compression as a conditional second branch, not the main bet. Winning is unproven.

Live board snapshot: ours 1.2274x rank 773, leader 1.6498x, third 1.6367x. Matching the leader requires
about 25.6% lower weighted time; matching third about 25.0%. Existing tiny hybrid gains are insufficient.
This refers to muse_bf16_proj, not the overall serving competition or prize eligibility.

## Evidence that changes the old plan

1. Six basic Lt configurations passed but lost overall; hybrid v1 improved only 0.3–0.5% in development.
2. New offline search requested up to 32 eligible choices per layout with 32 MiB workspace across 14
   dense shapes. The library offered 8 choices per layout: 224 valid shape/layout/algorithm combinations.
   Every trial passed output/probe checks; top 3 per shape were retested with fresh input. A/A champion
   controls ranged roughly 0.989–1.010, establishing a ~1% noise floor for this session.
3. Best promising choices: qkv 128 original rank 2 (~3.7% in confirmation), qkv 256 original rank 4 (~7.1%),
   pooler 128 original rank 0 (~4.2%), unpooler 256 original rank 0 (~4.3%). Smaller-batch gains were mostly
   inside control noise. These are shape gains, not overall leaderboard gains.
4. This only covers returned heuristics under one workspace cap and allowed reduction modes. It does
   NOT exhaust all cuBLASLt algorithms, tiles, stages or split-K settings. Do not claim a mathematical
   library ceiling. Offline selection is external developer tooling, never benchmark searches in prepare().

## First-principles priorities

Pooler, qkv and unpooler together represent ~78% of measured load time. Router alone cannot close
the gap even if it were free. BF16 weight traffic dominates low batch; intensity is approximately M
FLOP/byte. Under reference peak 989TFLOP/s and 4.8TB/s, crossover is around 206FLOP/byte. Real sustained
bandwidth and occupancy must be measured; hardware-floor estimates are optimistic.

A 64MiB weight matrix takes ~14us at peak bandwidth. A 25% storage reduction saves at most ~3.5us
before decode/escape overhead. Any proposed codec must fit inside that budget, not merely compress.

## Research and relevance

- NVIDIA Hopper tuning guide: https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
  TMA reduces data-movement instruction/register work; producer/consumer warp specialization can
  overlap transfers and tensor-core compute. Larger tiles/clusters can reduce occupancy, so use measured
  resource usage. Native features are mechanisms, not automatic speedups.
- NVIDIA CUTLASS efficient GEMM: https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md
  Small M/N grids underfill SMs; tile size trades parallelism against reuse. This directly motivates
  controlled split-K/Stream-K experiments for skinny projections.
- Stream-K: https://arxiv.org/abs/2301.03598
  Distribute inner-loop work to reduce wave quantization. Its reported maxima across other geometries
  are not predictions for our workload. Apply only if grid/occupancy evidence supports the mechanism.
- ZipServ: https://arxiv.org/html/2603.17435v1 (especially section 7)
  Fused decode into tensor-core consumption avoids a full intermediate tensor. Crucially, authors state
  their kernels can lose to cuBLAS on A100/H800; consumer-GPU headline speedups do not transfer to H200.
- Unweight: https://research.cloudflare.com/papers/unweight-2026.pdf (tables 9/11, sections 8.5/11.6)
  Hopper-specific lossless design still reports ~30–41% end-to-end throughput loss on H100 in this
  intermediate report, with reconstructive matmul slower than cuBLAS. Useful architectural lessons,
  strong evidence against treating compression as a ready-made winning solution.
- DFloat11: https://arxiv.org/abs/2504.11651
  Capacity/offloading benefits do not establish faster resident H200 matmul.
- CUDA event capture: https://docs.pytorch.org/docs/2.9/generated/torch.cuda.Event.html
  New search uses external timing events captured after an in-graph 128MiB cache flush; A/A controls and
  independent finalist measurements. Still a development method, not the official judge protocol.

## Next experimental sequence and gates

1. Validate hybrid v2 (new qkv 256 rank4) against champion across all public tests, graph replay,
   fresh inputs, and 30 timed shapes. Preserve it only as experimental until official confirmation.
2. Build a native uncompressed kernel baseline on pooler/qkv at M=8/16/32 first. Compare narrow
   mma.sync tiles and Hopper WGMMA (which may waste work at tiny M). Match weight layout to coalesced
   vector loads; test bounded split-K and avoid a separate reduction when grid parallelism suffices.
   Record kernel names, registers, shared memory, spills and profiler counters where permitted.
3. For M=128/256, prototype TMA/WGMMA double-buffered pipeline with producer/consumer warps;
   compare static split-K against work-centric scheduling. Do not use Blackwell-only instructions.
4. Gate each branch: repeated >=10% gain on dominant shapes or a credible predicted >=3% total gain
   before spending official quota. First milestone ~1.35x, then 1.45x/1.55x/1.65x; not promised outcomes.
5. Only revisit compression after the uncompressed consumer is competitive and bit-exact packed decode
   is cheap enough. Fixed-width palette + exact fallback, fused reconstruction; no lossy casts, output
   caching, input-value shortcuts, benchmark identification or reference access in submitted code.
6. Final candidate needs different random weight/input seeds, outliers/probes, nearby shapes, retained
   old outputs and correct graph replay. Use official checks after reset, and promote only a confirmed
   improvement over 1.227411869x/current account best. Keep historical sources/results immutable.

Budget: per-run 15-minute hard timeout, no retries, one GPU, $0 spend cap, no persistent volumes.
Reserve at least $5 free credits for independent confirmation. Do not burn all credits on blind searches.

## Follow-through results

Hybrid v2 passed full development validation twice, including independent weights on the second run.
Relative to champion: 1.008205 and 1.005935 (about0.8% and0.6% faster). This is a marginal,
unconfirmed official gain. Preserved in experiments/modal-best-v2-2026-10-30; original champion
remains untouched. Official quota is exhausted until reset, so no scored submission was made.
