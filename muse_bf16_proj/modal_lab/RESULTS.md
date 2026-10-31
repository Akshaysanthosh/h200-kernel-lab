> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# Modal H200 results — October 30, 2026

CLI connected to workspace akshaysanthosh through profile okc-free (1-day token).
Dashboard verified Starter $30 compute credits, $0 net spend cap. No paid spend authorized.
No persistent volumes, deployments, or endpoints created. All experiments are ephemeral apps.

PyTorch 2.9.0, CUDA 12.8, H200. The C++ cuBLASLt extension fully compiled, linked and ran.
Six initial candidates all passed 24 correctness cases, fresh-input/output-lifetime checks,
and graph replay correctness across 30 timed shapes. Native plans were observed in logs.
Relative speed vs preserved champion (higher is better):

| Candidate | Development relative speed |
|---|---:|
| original h0 | 0.992004 |
| transposed h0 | 0.996058 |
| original h1 | 0.980570 |
| transposed h1 | 0.981382 |
| original h2 | 0.972229 |
| transposed h2 | 0.974132 |
| hybrid v1, first measurement | 1.003016 |
| hybrid v1, independent repeat | 1.004950 |

The hybrid preserves champion dispatch except qkv m128/256 (original layout, eligible rank2),
pooler m128 (original rank0), unpooler m256 (original rank0). The strongest repeated gain is
qkv m128, around 5–6%; other gains are small. It remains a marginal experimental improvement,
not a proven official leaderboard win. Candidate and two results are preserved separately in
experiments/modal-best-2026-10-30. Main submission.py and champion snapshot are unchanged.

These are paired CUDA-event graph replay development measurements with 128MiB cache flushes,
not official timing: overhead is not subtracted, no formal confidence interval was computed,
and library/hardware scheduling may differ from the judge. No claims of 1.23x official score.

Two setup issues were fixed: entry point made importable instead of serializing code across
Python versions; result transfer changed to JSON to avoid a TorchVersion deserialization dependency
on the Mac. The initial result-transfer-failed run was not used as evidence. Logs retained.

## Next official run

Official quota confirmed 0/20 checks left on October30. No official checks or scored submissions
were used during this Modal session. After October31 00:00UTC reset, first check:

```
python3 tools/run_lt_check.py candidates/lt/lt_hybrid_v1.py --execute
```

Do not waste official quota rechecking the six uniformly slower variants. Review the hybrid's
correctness, no-flags status, official score and confidence interval. Only submit when the exact
candidate convincingly beats 1.2274118693095961 (and current account best if higher). Preserve the
champion if the apparent gain is noise. This is not a large enough improvement for top-three ambitions.

Further Modal work must reverify credits and the server-side $0 spend cap, verify CLI workspace,
and remain bounded. CLI token expires after one day. Never upload .modal.toml or challenge tokens.

Final dashboard reading: $0.42 credits applied, $29.58 credits remaining, $0.00 net charges. Billing may lag.

## Subsequent research pass

Read planning/WINNING_APPROACH.md. A bounded search tested224 valid native plans. Hybrid v2
changes qkv256 to original-layout eligible rank4; passed twice at1.008205 and1.005935 relative
to champion, including fresh weights. First official candidate is now candidates/lt/lt_hybrid_v2.py.
The earlier v1 evidence remains preserved. No official improvement is claimed.

## Custom Hopper follow-up

Read planning/HOPPER_RESULTS.md. Two bounded configuration sweeps yielded a combined
Hopper/cuBLASLt candidate: candidates/hopper/hybrid_v1.py. It passed twice with fresh
weights (34 correctness cases plus graph checks), measuring 1.013241x and 1.014003x
relative to the champion. This supersedes LT hybrid v2 as the first official-check
candidate. Larger custom-kernel paths were slower and excluded. No official submission
was made; the official champion remains byte-for-byte preserved. The 40% goal is unmet.

## Other bottlenecks and packed-weight research

Read planning/OTHER_BOTTLENECKS.md. A246-configuration direct-load/vector sweep and
a72-configuration lossless packed-weight prototype produced no material new winner.
Rejected search scaffolds must not consume official quota. The best experimental and
official candidates remain unchanged. All Modal jobs stopped after the bounded tests.

## Native producer/consumer pipeline

Read native_pipeline/README.md. Three native CUDA revisions passed correctness on
pooler M16 but failed the >=10% faster shape gate. The best packed result was72.672us
versus30.048us for the saved hybrid. All native search scaffolds are rejected for
promotion; preserve the existing official champion and Hopper hybrid. No official
quota or scored submissions were used.

## CUTLASS uncompressed Hopper consumer

Read cutlass_lab/README.md. Three successful sweeps tested70 configurations and six
independent-weight confirmations. Best TMA/WGMMA result30.336us versus29.824us for
the saved hybrid: near parity but still slower. cp.async/WGMMA did not improve it.
No decoding added, promotion made or official quota spent. Existing best versions
remain unchanged; do not submit the CUTLASS development scaffolds.

## CUTLASS fused lossless decode (2026-10-30)

See `cutlass_decode/README.md`: 24 configurations and four independent confirmations passed, but best confirmed 75.616 us vs saved best 29.936 us. No promotion, no submission. Free credits $27.59, charges $0; all apps stopped.

## Public v18 comparison, challenge CUDA13 stack

See `planning/PUBLIC_BASELINE_RESULTS.md` for the full audit and timing correction.
The public-derived candidate passed two independent-weight correctness suites
and measured 1.161254x / 1.163669x vs preserved champion (13.9–14.1% less time).
Use it first in the next official free check. This is not an official score.
Earlier CUDA12.8/zero-fill measurements must not be mixed with these results.
Champion and previous hybrid remain preserved; no scored submission was made.

## Further public-baseline optimization — no promotion

See `planning/PUBLIC_OPTIMIZATION_RESULTS.md` before spending more checks.
2,147 large-shape Lt schedules and 480 compressed WGMMA shape/config combinations
were evaluated on free Modal H200. A two-descriptor Lt refinement passed full
correctness twice but measured 1.001410x / 0.998251x vs public parent: rejected.
Larger decoded WGMMA tiles improved the experiment but still lost to parent.
The unchanged public parent remains the next official free-check candidate.
Do not repeat these rejected sweeps without a new hypothesis. All four apps stopped.

## Decode/WGMMA overlap pipeline — rejected

See `native_pipeline/WGMMA_PIPELINE.md` before revisiting this approach.
V1/v2 failed correctness due to live A-register access before asynchronous
completion, identified in a short-reduction diagnostic and emitted PTX.
V3 keeps issue, next-fragment decode and wait in one inline PTX block. It passed
288 production-shape/config tests plus finalist fresh-input graph checks, but
all shape finalists were slower than the public parent (ratios 0.611–0.744x).
No promotion or submission; scheduled public-parent validation is unchanged.
Five bounded ephemeral H200 jobs including two short diagnostics; all stopped.
Final dashboard: $25.99 free credits, $0 total spend, $0 spend limit (billing may lag).
