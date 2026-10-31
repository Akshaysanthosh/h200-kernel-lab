> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# Interleaved high/low router accumulation — 2026-10-31

The official router timings are 4.18/4.04/5.05/5.94/7.84 us at M8/16/32/128/256, accounting for roughly 9–11% of their load times. The router is a different opportunity from the failed QKV geometry and compressed decoder work. Reference floors are prioritization aids, not attainable-time predictions.

## Mechanism

The rules explicitly permit FP32 router weights represented by BF16 high + low components with FP32 accumulation. This experiment packs these components alternately along the reduction dimension during preparation. During each timed call, activation loads use the corresponding original K index twice. One BF16 tensor-core dot accumulates both terms into FP32. There is no approximate compression, exception decoding, activation preprocessing outside the timed call, cached output or precision shortcut beyond the permitted high/low representation.

The hypothesis is that one regular matrix product can simplify dual-component scheduling. Costs are duplicated activation accesses, twice the reduction length, and a second reduction kernel when splitting K. Splitting K across 1/4/16 blocks trades parallelism against that overhead. Triton's documented BF16 dot with FP32 accumulator is used (https://triton-lang.org/main/python-api/generated/triton.language.dot.html).

## Experiment

- Protected public v18 source/snapshot SHA256: 6faa2ae8519e3b9c19b27c2fdf47a2a057c7be89c72583ecfd270175c2144ae7.
- Old main/snapshot SHA256: 9ea10c81639eb23d6c9eb832465190328664fd655e89e9ff81732b94e10b8a36.
- Separate lab files only; no champion edits.
- One H200, CUDA13 challenge image, at most 900 seconds, zero retries, no persistent storage.
- Observed billing before launch: $23.51 credits, $0 net charges, $0 spend cap, workspace akshaysanthosh, CLI profile okc-free.
- 12 configurations per shape, five shapes. Two finalists per shape retimed with 120 paired replays, then on an independent weight seed.
- Cold-L2 read flush before every timed graph, alternating order, fresh inputs, normal and outlier distributions including one-hot probes, A/A controls and bootstrap intervals.
- No fixed graph overhead subtraction. Local timings are development evidence, not judge scores.
- Require repeated improvements before full-load/nearby-shape and official checks. No submission unless exact bytes pass official correctness/integrity and convincingly improve above the current account best.

## Results: rejected

60 first-seed configurations passed correctness, including fresh-input replay checks. Ten selected configurations repeated on independent weights also passed. Twenty finalist timing comparisons and ten A/A controls were recorded. No truncation or compilation failures. A/A ratios ranged 0.99061–1.01038.

| M | Best ratio, first seed | Best ratio, second seed | Second-seed parent / trial (raw us) |
|---|---:|---:|---:|
| 8 | 0.54360 | 0.54752 | 8.288 / 15.216 |
| 16 | 0.52348 | 0.52367 | 8.480 / 16.224 |
| 32 | 0.56617 | 0.56623 | 9.312 / 16.448 |
| 128 | 0.55204 | 0.55733 | 10.304 / 18.464 |
| 256 | 0.46795 | 0.46815 | 12.096 / 25.760 |

Ratios are parent / trial: below 1 is worse. All best configurations use split-K 16. Even these take roughly 1.8–2.1 times as long as the protected parent; the single-part variants are much worse. This rejects this implementation/layout, not every possible router optimization. The measurement does not isolate the relative costs of duplicated activation loads, layout conversion, and reduction; attributing all loss to any one factor would require profiling.

No nearby/full-load or official checks were spent because no finalist was competitive. No scored submission and no champion changes. Raw data: results/modal/20261031T055903Z_nathen_v18_linear_router_interleave.json. Modal app ap-CpeLZ6DbOAVuhdv80aKzEQ completed. Post-run dashboard: $23.38 credits, $0 charges, $0 cap, with usage reporting lag.

Live leaderboard at this test: rank 270/1516, top-ten cutoff 1.6003x, requiring 4.97597% less effective aggregate execution time than 1.5206695978723808x. A modest improvement on a single router shape cannot close that gap.
