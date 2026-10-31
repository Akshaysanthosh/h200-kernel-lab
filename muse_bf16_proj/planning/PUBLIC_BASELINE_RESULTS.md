> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# Public v18 audit and H200 comparison — 2026-10-30

Status: officially validated and submitted on 2026-10-31 at 1.5206695978723808x, rank 270/1515 at completion. See planning/OFFICIAL_PUBLIC_V18_RESULTS.md. Historical development evidence follows.
The official 1.2274118693095961x champion and experimental Hopper hybrid remain unchanged.

## Candidate and provenance

`candidates/public/nathen_v18_linear.py`, SHA256
`6faa2ae8519e3b9c19b27c2fdf47a2a057c7be89c72583ecfd270175c2144ae7`.
MIT NathenFernandes public v18, with attribution and explicit F.linear fallback
replacing the SGLang dispatch import. This is an adaptation of the published
1.5186x baseline, not a reproduction of its official score. Read
`research/2026-10-30/AUDIT.md` for source review, scope and compiler clock helper.
Preserved in `experiments/public-v18-2026-10-30/` with result and hashes.

## Results

One Modal NVIDIA H200, driver 610.57.04, 700W limit; PyTorch 2.14.1+cu130,
CUDA 13.0, cuBLASLt 130101. Container lmsysorg/sglang:dev-muse-spark-open.
No clocks were locked; the hardware metadata clock is an idle snapshot.

| Independent weight seed | Relative speed vs champion | Relative speed vs hybrid | Champion A/A control |
|---|---:|---:|---:|
| 82731 | 1.161254x | 1.162844x | 1.000542x |
| 294817 | 1.163669x | 1.159757x | 1.000190x |

Equivalent aggregate execution-time reductions: 13.89% and 14.06% vs champion.
Each seed passed 63 cases with normal and outlier inputs, exact-column probe
checks, output lifetime checks, plus fresh-input graph replay for all 30 scored
shapes on champion, hybrid and public. These are two weight sets on one allocation,
not independent machines. First public preparation took 90.80s (<300s cap),
second preparation with compiled modules cached took 2.25s.

Paired reversed-order 40 repetitions per shape, with 128MiB read flushing
outside the timed interval but inside the graph. Identical champion A/A control.
Six load ratios geometrically aggregated; all shape weights equal 62 and cancel.
No guessed overhead constant subtracted. This is not an official score or CI.
Do not multiply this local ratio by the old official score to invent a new score.

## Timing discrepancy

Pooler M16 calibration, microseconds:

| Method | Champion | Hybrid | Public adaptation |
|---|---:|---:|---:|
| Read flush, events inside graph | 24.736 | 24.160 | 18.416 |
| Zero-fill flush, events inside graph | 30.432 | 30.112 | 22.672 |
| Read flush outside graph, events inside | 24.880 | 24.064 | 18.544 |
| Historical zero-fill/external-event method | 30.432 | 29.696 | 22.784 |

The zero-fill flush materially changes measured latency. Switching to read
flushing removes roughly 5.7us from the champion measurement. Official timing
also subtracts fixed graph replay overhead (~4us); this explains much of the
remaining gap to the previous ~20.08us official measurement, but matching that
old number exactly remains unverified. Profiler CUDA event durations are saved
as diagnostics, not substituted for graph timings.

The old CUDA12.8 lab cannot run the public Lt extension (requires lib13), and
its algorithm choices are not transferable to CUDA13. The challenge-stack run
showed no repeatable overall advantage for our prior hybrid vs champion.
Use the challenge image and read-flush comparison for subsequent experiments.

## Next official gate

Today's free checks are exhausted. Existing Oct31 05:35 IST follow-up should
prioritize this exact public candidate, then repeat a successful free check for
confirmation before any scored submission. Inspect integrity flags, correctness,
profiles, prepare time and timing intervals; no submission on merely local evidence.
Submit only the exact bytes that pass with no flags and demonstrably exceed the
current account best (at least 1.2274118693095961x). Do not overwrite the champion
unless the scored result improves it. Maximum eight free checks; no paid GPU.

All three Modal apps stopped. Latest dashboard: $27.17 displayed free credits,
$0 total spend, $0 spending limit. Usage panels are temporarily inconsistent due
to billing lag; approximately $0.4 credits used since the pre-run observation.

Raw result: results/modal/20261030T111404Z_nathen_v18_linear_public.json.
Successful log: results/modal/public-v18-isolated-binding.log.
Setup failure logs: public-v18.log (CUDA library), public-v18-challenge-image.log
(global pybind LtPlan type collision). Lab hybrid copy renames only its host
binding type to avoid collision; saved hybrid source unchanged.
