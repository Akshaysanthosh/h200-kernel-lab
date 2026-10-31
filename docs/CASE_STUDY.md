# Engineering case study

## Problem

The workload computes `out = x @ W.T` for attention projections, pooler/unpooler projections and an expert router on H200. Token counts range from small decode batches to larger verification batches. Weight storage must retain its original precision; only lossless repacking is allowed, apart from the explicitly permitted router hi+lo representation.

The engineering challenge is that the best implementation changes with matrix shape. Small batches are heavily affected by weight traffic and launch costs; larger batches need efficient tensor-core utilization and careful memory scheduling. A smaller representation can lose if reconstructing it consumes too many instructions, registers or synchronization steps.

## Outcome

The current officially scored artifact achieves 1.5206696× the SGLang baseline, with all nine correctness tests passing and no reported integrity flags. Its interval is [1.4978596, 1.5434796]. It is a public-v18 adaptation, explicitly credited to Nathen Fernandes, rather than a newly invented algorithm.

The earlier scored result was 1.2274119×. Comparing those scored point estimates gives about 19.28% less time for the later version relative to the earlier one. These were separate official runs, so this is a historical comparison rather than a paired causal estimate.

## Work carried out in this project

1. Established an initial correct submission and preserved exact-byte scored snapshots.
2. Built local and cloud evaluation tooling for candidate comparison, correctness, fresh inputs and retained outputs.
3. Investigated cuBLASLt layouts and schedules, native CUDA pipelines, CUTLASS, lossless formats and decoding strategies.
4. Audited and adapted public implementations, retaining license notices and recording their source hashes.
5. Reconciled timing differences between CUDA 12.8 development tests and the CUDA 13 challenge stack.
6. Used official free checks to validate promising artifacts, and separated those results from leaderboard submissions.
7. Rejected candidates with non-repeating improvements, including ones with attractive short-screen results.

## Three important engineering decisions

### A benchmark control must execute the same implementation

QKV at 128 tokens already used a custom WGMMA route. Comparing an apparently identical stored cuBLASLt configuration to that route was not an identical-code control. Tracing the final dispatch resolved the discrepancy and changed how later experiments were interpreted. [Audit](../muse_bf16_proj/planning/QKV_CONTROL_AUDIT.md)

### Compression is a pipeline tradeoff

A grouped lossless format reduced storage but introduced prefix/gather and exception work. Wider compressed WGMMA tiles improved experimental variants without beating the established baseline. Those results justified rejecting the tested formats, not declaring compression universally ineffective. [Format study](../muse_bf16_proj/planning/GROUPED_FORMAT_EXPERIMENT.md)

### A higher point estimate is insufficient for promotion

The SK2 hybrid passed an official free check at 1.53485×. Its interval overlapped the 1.52067× champion and its full-workload local margin was small. It was retained as a development artifact rather than submitted as a demonstrated improvement. Interval overlap alone is not a formal significance test; the decision also considered paired controls, repeatability and effect size.

## Collaboration and ownership

This work was performed with coding agents and research assistants. Upstream kernel authors supplied major algorithmic contributions. The portfolio claim is the documented process of investigation, integration, validation and decision-making, with source available for inspection. It does not imply sole authorship of upstream CUDA or a top-10 finish.

## Lessons

- Inspect active dispatch and actual library versions before trusting comparisons.
- Optimize a meaningful share of the scored workload, not only the easiest individual shape.
- Use controls and independent data to distinguish gains from allocator, compilation and device-state effects.
- Keep failed experiments: they narrow future work and explain why a plausible design was abandoned.
- Preserve exact bytes so the code being discussed is the code that was checked.
