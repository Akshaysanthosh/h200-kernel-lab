# Benchmark methodology and interpretation

## Official metric

For each of six serving loads, the judge sums weighted times for five layer types. The problem score is the geometric mean of baseline/candidate load-time ratios. Thus a 6% gain in one layer of one load can become less than 1% overall.

The judge captures CUDA graphs, uses fresh inputs, starts timed replays with a cold L2, alternates candidate/baseline order and reports an uncertainty interval. Its fixed graph-overhead correction is approximately 4 microseconds. Read [TASK.md](../muse_bf16_proj/TASK.md) for the exact described procedure.

## Local development comparison

The newer lab evaluators use the same H200 process, weights and software stack for parent and candidate. A 128 MiB read flush precedes timing; external CUDA events bracket computation inside graphs. Paired repetitions reverse order. An identical-parent A/A control measures drift. Full confirmations use two independent weight seeds.

No guessed 4-microsecond constant is subtracted locally. Local raw timing ratios are **not official judge scores** and cannot be directly multiplied into a promised leaderboard result. Earlier evaluators differ; consult each historical report before comparing numbers across phases.

## Correctness beyond one sample

Full confirmation includes normal/outlier weights or inputs, sparse probe rows, nearby token counts, reference comparisons, graph replay after fresh input changes, and retained-output lifetime checks. The parent and candidate must operate on the same activations when attributing an isolated effect.

## Promotion policy

1. Confirm candidate source hash and protected champion hashes.
2. Pass correctness before timing.
3. Confirm gains with reversed order, A/A controls and independent weights.
4. Inspect the full workload, preparation time and neighboring shapes.
5. Run an official free check, review correctness and integrity flags.
6. Submit only a qualified improvement; preserve the earlier best.

Identical-source official requests may be deduplicated. A duplicate response is not independent confirmation. Overlapping intervals are treated conservatively alongside effect size and paired evidence, not as a standalone hypothesis test.

## Known limitations

- The challenge image tag is mutable; historical library metadata matters. CUDA12.8 results did not reliably transfer to CUDA13.
- The public source contains fixed Lt configurations tied to cuBLASLt version 130101.
- Allocator layout, GPU state and measurement overhead matter for microsecond kernels.
- Some early profiles and reports lack the controls introduced later.
- Official no-flags validation is evidence of acceptance for that run, not a formal proof of compliance or a final leaderboard review.
- GPU code was not rerun merely to prepare this portfolio publication.
