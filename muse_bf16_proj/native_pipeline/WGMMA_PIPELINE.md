> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# Decode / WGMMA pipeline experiment

Preserved best candidates are untouched. This directory contains experimental
CUDA, not an approved submission. `modal_lab/pipeline_source.py` reproduces it
from the attributed public baseline; `pipeline_v1.diff` shows the scheduling
change against the synchronous wide-tile version.

## Original v1/v2 contract and schedule (rejected)

The public lossless BF16 packing and exception restoration remain unchanged.
BF16 tensor-core operands, FP32 accumulation, and the original final reduction
are preserved. One committed WGMMA group can be outstanding at a time.

1. Decode tile i into a next-fragment register set while tile i-1 is in flight.
2. Drain the previous group before touching its accumulator or A registers.
3. Refill the released TMA slot and copy the decoded fragment into live A.
4. Wait for the current TMA tile, fence registers, issue four K16 MMAs, commit.
5. Drain the final group before barriers are invalidated or shared memory reused.

The drain asm uses read/write constraints for live A and accumulator registers,
and input constraints for the next fragment. This keeps the live registers
reserved while requiring the next decode before the drain. Q=2 separates TMA
slots. Warpgroup control conditions are uniform. Variants with local-memory
spills are not timed. These source-level invariants are not a substitute for
runtime correctness testing or inspection of the compiler's final scheduling.

Source: NVIDIA PTX ISA, wgmma.wait_group register-access requirements:
https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group

## Test plan

Same H200, CUDA13 challenge image, public parent kernel as reference competitor.
Cold 128MiB read flush outside the event timing interval inside CUDA graphs.
Reversed-order paired timings; full numerical checks before screening, fresh
input graph replay for finalists. BN64/128/256, WK1/2, prefetch1/2 and cluster
split1/2/4/8. No timing or tuning code enters a submission.

Promote only after independent-weight confirmation. Submit only after official
free checks pass without integrity flags and demonstrate a genuine improvement.

## Revision history

- v1 compiled but immediately failed outlier correctness on unpooler m128,
  BN64/WK1/PF1/split1, with NaN/Inf. No performance accepted. Log:
  results/modal/pipeline-v1.log. v1 source and generator are preserved.
- v2 makes the WGMMA A operands read/write in the issue asm (opaque to compiler
  copy propagation), and adds a named 128-thread barrier after the previous
  group drains and before its shared slot is refilled. Both changes strengthen
  lifetime/synchronization constraints; they do not establish which caused v1's
  failure. Source/diff reproduced by tools/render_pipeline.py.

## Diagnostic and revision 3

The short-reduction diagnostic established that v2 passes K64 but produces
nonfinite outputs from K128 onward, while both synchronous original and
immediate-wait control pass K64/128/256/512/4096. Preserved PTX shows generic
`mov.u32` copies of the matrix-A registers after the async issue and before
completion, followed by next-iteration writes to those registers before wait.
This violates the documented register-access lifetime, despite C++ liveness
constraints. Merely naming separate C++ arrays was insufficient.

Revision 3 puts the current tile's four WGMMA instructions, commit, decoding of
the next raw tile into early-clobber output registers, and wait in one inline
PTX block. No group is outstanding across the C++ loop boundary. Original
TMA refill timing is restored. Exception patches still restore original BF16
bits before multiplication. This overlaps the base decode; exception processing
remains outside the in-flight interval. The last tile's unused next decode is
harmless and does not consume a new input. Original prologue initializes raw
buffers, so no uninitialized future weights are read.

The v3 diagnostic passed all K64/128/256/512/4096 cases, with the same reported
FP64-relative errors as the synchronous control. Diagnostic evidence:
results/modal/20261030T120122Z_nathen_v18_linear_pipeline_debug.json.
Full production-shape timing is a separate gate; this diagnostic is not a win.

Generated CUDA derives from NathenFernandes public v18 (MIT), whose attribution
and license are retained in candidates/public/nathen_v18_linear.py and
research/2026-10-30/nathen-LICENSE. Scheduling changes are local experimental work.

## Final result — reject for performance

V3 passed 288 shape/config correctness checks and fresh-input graph replay
checks on finalists. All six shape winners lost to the preserved public parent.
These were development measurements on one H200, not official scores.

| Shape | Parent us | Pipeline us | Parent/pipeline |
|---|---:|---:|---:|
| unpooler m128 | 22.784 | 30.880 | 0.7351x |
| unpooler m256 | 27.296 | 40.032 | 0.6856x |
| pooler m128 | 24.176 | 32.336 | 0.7439x |
| pooler m256 | 27.584 | 41.232 | 0.6755x |
| qkv_gate m128 | 17.728 | 24.544 | 0.7186x |
| qkv_gate m256 | 21.344 | 34.848 | 0.6113x |

No promotion, no scored submission, no official free check spent on this
rejected kernel. Both the official champion and public-derived best remain
unchanged. Existing next-reset official validation still targets the public
parent. Do not spend checks on this pipeline without a new measured hypothesis.

Final source SHA256 matches the source generated locally and recorded in the
result JSON. SASS excerpt and register/local-memory resource counts are also
recorded. No hardware stall counters were collected; extra live registers and
restricted scheduling are plausible costs, not proven exclusive causes.

Evidence: results/modal/20261030T120212Z_nathen_v18_linear_pipeline.json.
