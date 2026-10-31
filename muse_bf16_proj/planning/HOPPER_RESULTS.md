> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# Hopper custom-kernel experiments — October 30, 2026

Target: reduce whole-workload time by 40% from the official 1.2274118693x champion.
No such gain has been achieved. All compute uses Modal included credits with a verified
$0 spend cap. Official checks remain exhausted until October 31 00:00 UTC.

## First-principles limit

Using the champion's official profile and replacing only pooler/qkv times by TASK.md's
hardware floors, while retaining every other layer's measured time, reduces the geometric
mean load time by approximately 19.83%. This is an optimistic model, not a measured result;
the stated floors assume ordinary uncompressed data movement. A 40% whole-workload reduction
requires improvements beyond these two layers, or a lossless representation reducing traffic.

## Implementation and evidence

Original Triton kernels use Hopper tensor descriptors for TMA loads, FP32 accumulation,
WGMMA matrix instructions and an FP32 split-K reduction. No runtime benchmarking, input
result caching, reduced-precision weights or harness dependencies exist in submission code.
External tuning records generated instructions, shared memory, registers and spills.

The first default configuration passed correctness but achieved only 0.90958x relative
to the champion. Two subsequent sweeps tested orientation, tile size, K chunk size,
split-K and pipeline depth on all ten pooler/qkv batch combinations. Generated PTX
confirmed TMA and WGMMA and no register spills for the selected configurations.

Small qkv batches showed roughly 3–8% isolated gains; small pooler batches roughly 1–2%.
Large-batch custom paths lost to the library and are excluded from the combined candidate.
These are development measurements, not official leaderboard improvements.

Sources of evidence:
- results/modal/20261030T093523Z_tma_v1_benchmark.json
- results/modal/20261030T093911Z_tma_v2_hopper.json
- results/modal/20261030T094023Z_tma_v2_hopper.json
- modal_lab/tune_hopper_round1.py and modal_lab/tune_hopper.py (external tuners)
- tools/build_hopper_hybrid.py (reproducible candidate builder)

The combined candidate applies custom kernels over small-batch ranges, retains the
cuBLASLt hybrid v2 for selected large shapes, and uses champion fallbacks elsewhere.
Nearby shapes, new input values, output lifetime, graph replay and independent weights
must pass before considering an official check. Never submit solely on Modal timings.

## Combined validation

Hybrid v1 passed 34 correctness cases, fresh-input/output-lifetime checks and graph
replay checks across all 30 scored shapes in two independent-weight runs. Relative
speedups were 1.0132408196 and 1.0140025210: approximately 1.31% and 1.38% less time.
This is the best experimental candidate, but not an official leaderboard improvement.
Candidate and both result files are preserved with SHA256SUMS under
experiments/modal-best-hopper-2026-10-30. The 1.2274118693x official champion is unchanged.

At the next reset, first check candidates/hopper/hybrid_v1.py with the guarded runner.
Only submit after official passing improvement; modest gains may overlap timing noise.

Final dashboard: $29.23 included credits remaining, $0.77 used, $0 net charges,
$0 spend limit. Billing can lag. All Modal apps confirmed stopped.
Scheduled official validation updated to October 31, 05:35 IST, with this candidate first.
