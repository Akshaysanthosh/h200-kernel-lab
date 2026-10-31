> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# Native lossless pooler pipeline

Scope: pooler M16, N4096, K8192. Original CUDA C++ implementation using CUDA
partitioned pipelines and BF16 WMMA with FP32 accumulation. This first native
prototype uses WMMA, not Hopper WGMMA assembly. No performance claim is implied.

Producer warps decode losslessly packed weights and copy current activations into
shared-memory tiles. Consumer warps execute tensor-core matrix products. Pipeline
acquire/commit and wait/release protect ring-buffer reuse. Stage1 serializes producer
and consumer phases; stage2/3 allow decode of future tiles to overlap consumption.
The library controls use the same inputs and cold-L2 graph replay timing.

Eight fixed configurations vary N/K tile widths, producer count and buffer count.
Split-K2/4/8 provides grid parallelism; an FP32 partial reduction and BF16 output
conversion are included in timed calls. Uncompressed native variants isolate the
packing cost. Prepare contains weight packing/extension compilation only; fresh
activations are consumed during every timed invocation. No output caching occurs.

The first shape must pass exact storage reconstruction, strict output checks,
fresh-input and retained-output tests and graph replay checks. The target is at least
10% less time than the saved experimental best on this shape (ratio >=1/0.9).
An isolated shape win is not an overall leaderboard score or permission to overwrite
either preserved best version.

API references:
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html
- https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/pipeline.html
- https://github.com/NVIDIA/cuda-samples/tree/master/cpp/3_CUDA_Features/bf16TensorCoreGemm

Builder: tools/build_native_pipeline.py. External evaluator: modal_lab/test_native.py.
The candidate retains the previous kernel_fn as a control and exposes _native_run
only for the development evaluator. It must not be submitted as a new optimized entry.

## Revisions under test

- v1: CUDA partitioned pipeline, scalar producer loads. 48 combinations passed trial
  correctness; best packed result ~398.6us versus ~29.9us saved-best control. No win.
  Exact source is preserved in packed_pooler_v1.cu and native_pipeline_v1.py.
- v2: explicit ready/free shared-memory CUDA barriers, independent weight seed,
  outlier input restored for every trial.36 combinations; best packed result ~338.6us
  versus ~29.7us control. No win. Source: packed_pooler_v2.cu and native_pipeline_v2.py.
- v3: same barrier structure, but producers load/store eight BF16 values per vector
  and decode packed values from one 64-bit sign/mantissa load and one 32-bit exponent
  load. This removes scalar global-load loops. The best packed result is 72.672 us versus 30.048 us for the saved best; no win.

The first sweep changed inputs to normal after its first trial; subsequent evaluators
restore outliers for every configuration and separately verify normal inputs, output
lifetime and fresh graph replay. Do not treat the first sweep as a full outlier suite.
All timing includes partial reduction and BF16 output conversion.


## Final outcome

The sweeps ran 48 + 36 + 36 configurations. All 120 passed the trial output checks;
revisions 2/3 also restored outliers independently for each trial. Source and all JSON
results remain preserved, including generated-code resource summaries. The best v3
packed path used 128 output columns, 64 K elements, 128 producer threads, two buffers,
and split-K 16. Its median paired ratio to the saved best was 0.4134 (slower).

Vectorization improved this prototype substantially, but it does not meet the required
10% shape-time reduction. No official checks or submissions were used. Neither
submission.py nor candidates/hopper/hybrid_v1.py changed. Scheduled validation should
continue using the existing hybrid, not any native search scaffold.

Controls point to remaining producer and consumer overhead: the native uncompressed
paths are also slower than the library/Triton best. Matched double-buffer variants did
not outperform their single-buffer controls. SASS confirms vector global loads/stores,
BF16 HMMA instructions and hardware barrier instructions, with no local-memory spills
in the resource reports. These are static instruction/resource observations, not Nsight
stall-counter measurements. They do not establish that more buffering will improve speed.

Further work would require a stronger uncompressed native consumer (e.g. a tuned
Hopper WGMMA/CUTLASS schedule) before reintegrating decode. Continuing to tune this
WMMA prototype is not supported by the current performance evidence.

Final dashboard: $28.65 free credits remaining, $1.35 credits used cumulatively,
$0 net charges and $0 spend cap. Billing may lag. All Modal apps stopped. Both best
source hashes verified unchanged.
