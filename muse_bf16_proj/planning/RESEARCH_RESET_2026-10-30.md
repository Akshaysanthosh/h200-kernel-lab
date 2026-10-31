> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# Research reset after the CUTLASS decoder experiment

Recommendation: reproduce a credible, stronger public same-problem baseline,
then optimize measured remaining gaps. Stop treating generic CUTLASS integration
or compression alone as a likely breakthrough. No GPU jobs run in this research turn.

## Target

Live CLI snapshot saved in results/leaderboard-research-20261030.json:
ours 1.2274x, rank 779; leader 1.6498x; third 1.6367x; tenth 1.5915x.
Matching these requires respectively 25.60%, 25.01%, and 22.88% lower aggregate
time than our official champion. Rankings are provisional; the leader's screening
is pending. A 40% reduction is beyond what we need to match today's leader.

## New evidence

Nathen Fernandes publishes an MIT-licensed implementation of this exact problem:
https://github.com/NathenFernandes/muse-bf16-proj
Its 1.5186x result and job 66d8b007f6c641ee match the live board (rank 220).
Read-only source inspection found operand-fragment repacking, pipelined register
loads, byte-permutation decoding, exact exceptional-bit restoration before MMA,
and cluster reductions. This is a much stronger baseline lead than a generic
square-matmul benchmark. It is not the current winning implementation, nor have
we reproduced it. Attribution and license must accompany any adapted code.
Source/provenance/license and published shape table are in research/2026-10-30/.
No downloaded code was imported or executed.

A separate authors' guide explains fragment-order packing and compact decoding:
https://thethings.ai/p/okc/muse-bf16-proj-guide/05-repack
https://thethings.ai/p/okc/muse-bf16-proj-guide/07-format
Its compression consumer uses a different exception-correction scheme; do not
conflate that with exact register restoration in the inspected repository.
Use the latter as the preferred correctness contract.

ZipServ (ASPLOS 2026) explains tensor-core-aware lossless formats and overlapped
register decoding, but explicitly reports that A100/H800 performance may lose
to cuBLAS. Its consumer-GPU headline gains do not predict our H200 score:
https://arxiv.org/html/2603.17435v1
Stream-K offers work balancing, not a guaranteed bandwidth improvement:
https://arxiv.org/abs/2301.03598
FlashInfer now exposes a small-M BF16 TinyGEMM backend with SM90 support; this is
an independent control worth comparing, subject to version/API/precision audit:
https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/gemm/gemm_base.py
TileOPs' own small-M investigation also documents that measured crossovers differ
from initial predictions and peak bandwidth is an optimistic denominator:
https://github.com/tile-ai/TileOPs/issues/1664

## First-principles diagnosis

For BF16 W[N,K], compulsory weight bytes are 2*N*K, work is 2*M*N*K;
weight-only arithmetic intensity is approximately M FLOP/byte. Small batches
therefore need efficient weight streaming and enough parallel work, not maximum
square-GEMM compute throughput. Layout must match the consumption instruction.
For pooler the 64 MiB weights require about 14 us at the stated 4.8 TB/s peak;
12-bit packing saves only about 3.5 us of ideal transfer time, before metadata,
exceptions and decoding. Saving that cannot compensate for our 75.6 us decoder.
That experiment rejects our pipeline, not every lossless representation.

There is a calibration discrepancy: the champion's official pooler M16 time is
20.08 us; recent Modal best is around 29.9 us. These are different runs/methods,
not a controlled hardware comparison. Check clocks/power/software/kernel names,
cache-flush implementation, graph timing and overhead accounting. Never infer
an official score from raw Modal microseconds. Same-session ratios and subsequent
official checks are mandatory. Do not assume GPU clocks are the cause.

## Best next sequence

1. Freeze current best. Audit the public implementation's active dispatch, all
   compressed strings, compilation, precision, prepare behavior, shape fallbacks,
   graph lifetimes and licensing. Reproduce a minimal relevant path first; don't
   blindly submit a large borrowed file or copy its historical tuning scaffolds.
2. Calibrate the official champion, our best hybrid, the audited public baseline,
   and (if feasible) FlashInfer TinyGEMM in the same H200 session. Use all 30 shapes,
   paired cold-L2 graphs, independent weights/inputs and per-load aggregation.
3. Focus small M=1/8/16/32 on fragment-order prepacking, multiple outstanding loads,
   exact register-level decode, and bounded warp/cluster split-K. Compare compressed
   and uncompressed twins. Avoid global partial writes/reduction launch only when
   cluster occupancy and synchronization are actually cheaper.
4. Keep M128/256 library paths until counters justify a replacement. Compare
   pinned Lt choices, narrow WGMMA and Stream-K only on measured weak shapes.
5. Use external diagnostic ablations: read-only checksum, loads+decode checksum,
   full MMA and reduction. Collect bandwidth, long-scoreboard stalls, occupancy,
   register/spill and tensor utilization where profiler permissions permit.
   Diagnostics never enter submission dispatch or alter benchmark behavior.
6. Promote only repeated >=10% reduction on a dominant shape, or >=3% aggregate
   gain outside measured A/A noise. Validate graph freshness, all probes and nearby
   shapes, then official checks after reset. Submit only a confirmed overall gain.

Reproducing 1.5186x would imply about 19.2% less aggregate time than our current
score, but would still need another 8.0% time reduction to match today's leader.
These are score-ratio calculations, not forecasts. If reproduction fails, diagnose
before spending credits on modifications. Budget first reproduction session to
at most ~$1 of free credits, keep a $5 confirmation reserve, and retain the $0 cap.

Decision: highest-confidence next action is audited baseline reproduction and
measurement calibration. The highest-upside design branch is consumption-order
packing + latency-hiding register decode, with per-shape dispatch. There is no
verified path to first place yet; a paper or public score is evidence to test.
