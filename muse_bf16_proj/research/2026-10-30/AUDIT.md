> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# Public v18 baseline audit

Upstream: NathenFernandes/muse-bf16-proj, solution branch, retrieved 2026-10-30.
Exact source hash is in provenance.json. MIT license retained. This is a manual
source review supported by static inventory, not a formal correctness proof.

`tools/audit_public_baseline.py` parses the source without importing it, expands
68 assigned CUDA/C++ literals, and preserves a readable Python control-flow copy.
Inspection included final prepare/kernel chains, fallback, fixed cuBLASLt
configuration restoration, packing/inverse packing, register exception patches,
cluster reductions, and fresh output allocation. No network, subprocess launch,
harness/reference imports, event-timing modifications, or cached input outputs
were found. PyTorch load_inline is the standard compilation facility.

Preparation builds many historical modules in four worker threads and validates
weight packing with synthetic inputs. It does not use actual inference inputs.
The final preparation warms a fixed route list; the Lt path restores fixed
algorithm descriptors, it does not benchmark/select algorithms during prepare.
Unused diagnostic template parameters exist; selected production routes use full
computation. Independent test coverage is required before promotion.

Important nuance: several native helpers read the GPU `%clock` register to create
an opaque compiler control-flow boundary. The trap condition `(clock & m) ==
0x7fffffff` cannot be true for supported small token counts. This is not used to
measure a run, distinguish benchmark inputs, or choose outputs. Retain this in
the audit trail and require official no-flags validation; don't describe the
source as having no clock reads. Likewise, resource-query APIs are not timers.

Dense tensor-core paths use BF16 operands and FP32 accumulators; router retains
FP32 or allowed hi+lo. Compressed paths patch original bits before multiplication,
with fallback when escape lists exceed capacity. These claims are based on
inspected paths and need runtime probes across independent weights and inputs.

## Standalone adaptation

candidates/public/nathen_v18_linear.py differs from upstream only by license/
attribution comments and replacing the SGLang dispatch import with explicit
`F.linear(x, weight, bias)`. This avoids installing a large serving dependency in
the lab. It is an adapted public baseline, NOT the byte-identical 1.5186x submission.
The challenge baseline documents its H200 dense fallback as cuBLAS; the pinned
SGLang implementation has F.linear as its library branch, but also optional
specialized branches. Hence equivalence is limited to that library branch and
must not be assumed for every SGLang environment.

The adapted bytes are the candidate to validate and eventually submit, if they
qualify; do not test one source and submit another. The upstream result is
provenance, not independent validation of this adaptation.

## Timing method

The new evaluator captures a 128 MiB read-and-checksum flush followed by external
CUDA timing events around kernel_fn inside the same graph. A paired reversed
order comparison uses identical GPU, weights, inputs and libraries. An identical
champion graph is an A/A control. The flush is excluded from the timing interval.
A separate pooler calibration compares read flushing, historical zero-fill
flushing, and external-event timing around graph submission. No arbitrary 4 us
constant is subtracted. Official TASK.md describes a read flush and fixed replay
overhead subtraction; local measurements remain development estimates.

The evaluator uses two independent weight seeds, all public test shapes, all 30
scored shapes, outliers, probe rows, nearby shape boundaries, retained outputs,
and graph replay after fresh input changes. Original champion and hybrid source
hashes are preserved. Official free checks are currently 20/20 used; no scored
attempt is to be used as a replacement for a free validation check.

## Environment finding during first launch

The CUDA 12.8/torch 2.9 lab could not link the upstream Lt extension because it
explicitly requires libcublasLt.so.13. Its fixed algorithm table is guarded by
cuBLASLt version 130101. This is material: previous development timings did not
use this library stack. The failed build log is results/modal/public-v18.log.
The comparison is being rerun with the upstream-recommended development image
lmsysorg/sglang:dev-muse-spark-open. That tag is mutable; the image build log and
runtime version metadata must be preserved. None of this justifies changing the
numerical/timing rules or pretending an older-stack score transfers exactly.

## Same-process binding isolation

The challenge-image launch passed CUDA linking but exposed a global pybind type
collision: both our hybrid and upstream register a C++ `LtPlan` class. For the
comparison only, modal_lab/hybrid_comparison.py renames our host C++ type to
OkcHybridLabPlan. It is generated by one textual substitution from the saved
hybrid; the original file remains untouched. This does not alter CUDA kernels,
math, dispatch, or algorithm settings. The public candidate remains unchanged.
The prior setup failure is recorded in public-v18-challenge-image.log.

## Runtime outcome

Both weight seeds passed the full lab suite. Public cold preparation 90.80s;
cuBLASLt 130101 with all seven expected public fixed-plan configs available.
See planning/PUBLIC_BASELINE_RESULTS.md and the raw JSON for timings and limits.
Official integrity review remains pending; a local audit cannot guarantee acceptance.
