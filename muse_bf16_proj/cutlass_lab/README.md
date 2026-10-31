> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# Uncompressed Hopper CUTLASS pooler experiment

Target: pooler M=16, N=4096, K=8192, compared with the preserved Hopper hybrid
(best experimental candidate) and official champion, on the same H200.

CUTLASS v3.9.2 is pinned at ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e.
It is downloaded from https://github.com/NVIDIA/cutlass and kept in ignored vendor/cutlass.
Its BSD-3-Clause license and copyright notices remain in that checkout. The original
PyTorch wrapper in pooler.cu uses public collective-builder APIs; no sample executable
or reference computation is included in the timed kernel.

The problem is oriented as W[4096,8192] times X^T[8192,16]. Hopper SM90a TMA and
warp-specialized WGMMA operate on BF16 operands with FP32 accumulators. Four tile shapes
and five split-K values are selected externally. Split-K is represented as independent
K-partition batches with explicit strides; partial outputs are FP32 and their reduction
and BF16 cast are included in every measurement. Each call uses fresh output/workspace.
There is no weight decoding or precision reduction in this experiment.

Correctness gates: outlier/probe checks for each configuration, normal/fresh-input checks,
retained-output checks and graph replay with changed values. Timing uses alternating
orders and 48 paired cold-L2 graph measurements. The two fastest correct configurations
are confirmed with independent weights and new graph allocations. These are development
measurements, not official scores. Generated SASS/resources are captured to verify actual
WGMMA and inspect register/spill use.

Only a competitive uncompressed result justifies adding decoding. The prior official
and experimental best source files must remain unchanged throughout the experiment.
Candidate code currently depends on development-mounted CUTLASS headers and retains
old kernel_fn for comparison; it is not a deployable new competition submission.

Reproduce locally with tools/build_cutlass_candidate.py, then use the guarded Modal
runner in cutlass mode after verifying free credits and the $0 spending cap. The
external evaluator is modal_lab/test_cutlass.py. No official checks are spent here.

API reference:
https://github.com/NVIDIA/cutlass/tree/v3.9.2/examples/49_hopper_gemm_with_collective_builder

## First sweep

All 20 configurations passed. Best tile 64x16x128, three stages, split-K 4 measured
31.216us versus 29.936us for the saved hybrid in independent-weight confirmation.
A 64x32x128 tile confirmed similarly. SASS contains HGMMA.64 instructions, establishing
that the test uses Hopper warpgroup matrix instructions. This is close but slower;
no decoding was added. JSON: results/modal/20261030T101915Z_cutlass_v1_cutlass.json.

The second sweep varies two/three/four pipeline stages and compares direct global
stores against the TMA/shared-memory epilogue. This tests whether small partial-output
epilogue costs and occupancy account for the remaining gap. v1 source is preserved in
pooler_v1.cu and candidates/hopper/cutlass_v1.py; the v2 source is separate.

## Second sweep

All 30 configurations passed. Direct global stores reduced the gap: three-stage and
four-stage 64x16x128, split-K 4 confirmed at ratios 0.9815/0.9830 to the saved best,
with independent weights. The best confirmed absolute time was 30.336us versus 29.824us.
This remains slower; no source promotion or decoding was performed.
Evidence: results/modal/20261030T102309Z_cutlass_v2_cutlass.json.

The final comparison changes only the uncompressed loading/scheduling family to
CUTLASS KernelCpAsyncWarpSpecialized, keeping Hopper WGMMA and direct output stores.
It tests 64/128-wide weight tiles, 64/128-element K tiles and three/four stages. The
same shape, precision, output checks, paired timing and independent confirmation apply.

The first cp.async compile rejected the 64x16x64 tile: its 256 copy threads require
at least 32 activation rows at K64/alignment8. No GPU timing was produced for that
attempt. The retry changes that tile to 64x32x64, leaving the other three unchanged.
The failed source/log remain preserved for diagnosis; no tolerance was changed.


## Final result and decision

Three successful sweeps tested 70 configurations plus six independent-weight
confirmations. Every evaluated configuration passed output, fresh-input/output-lifetime
and graph replay checks. One initial cp.async build failed before testing and was fixed
by using a supported tile; that attempt is not counted as a passing configuration.

The strongest confirmed uncompressed native result remains v2, config 2 / split 4:
30.336us versus 29.824us for the preserved hybrid, paired ratio 0.9830 (about 1.7% slower).
The cp.async family confirmed at ratios 0.9244/0.9347; generated code contains both
LDGSTS and HGMMA instructions. TMA/WGMMA is the better tested family for this shape.
Static resource reports show no local-memory allocation; these are not Nsight stall
counter measurements and do not establish the exact remaining bottleneck.

No new winner was promoted, no decoding added, and no official quota or scored
submission used. The official 1.2274118693x champion and the saved 1.4% experimental
hybrid are byte-for-byte unchanged. Scheduled validation should continue to use that
existing hybrid. Do not run official checks on these control-dispatch scaffolds.

The best CUTLASS shape configuration, source, dependency pin, result and hashes are
preserved separately under experiments/cutlass-best-shape-2026-10-30. It is a useful
near-parity native baseline, not a demonstrated faster submission. Further decoder
work would need to offset its own overhead and this remaining gap; the current tests
do not justify claiming that outcome or adding decoding as an improvement.

Final billing: $28.09 included credits remaining, $1.91 used cumulatively, $0 net
charges and a $0 spend cap. Billing may lag. All GPU apps confirmed stopped.
