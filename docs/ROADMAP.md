# Next hypothesis: hardware-assisted weight traffic reduction

**Status: research proposal only. No GPU test or measured gain.**

The next inexpensive feasibility experiment is Hopper inline compression for the dense BF16 allocations used by pooler and unpooler at 128/256 tokens. Keep the GEMM algorithm fixed; change only the allocation's compression request. The premise is avoiding software reconstruction in the inner loop while preserving exact values.

NVIDIA documents per-allocation compression and potential bandwidth benefits. A device support query is necessary, followed by a query of the granted allocation properties: a request does not guarantee compressed allocation. Hardware compression may not exploit our actual weight distribution.

## Bounded experiment

1. Verify free-credit balance, server spending cap and device support.
2. Compare ordinary allocation, VMM allocation with compression disabled, and VMM allocation with compression enabled.
3. Start with batch-256 pooler, identical weights and identical cuBLASLt algorithm; preserve correct lifetime and alignment.
4. Measure exact values, fresh-input correctness, paired cold-L2 graph timing and traffic counters across independent weights.
5. Stop if unsupported, no meaningful traffic reduction, or no repeatable timing benefit. Expand to the other three cases only after a repeatable win.
6. Require full-workload and official validation before any submission.

Pooler and unpooler account for about 58–60% of the respective 128/256-token load times in the saved official profile. Hypothetical 15%, 20% and 25% time reductions across those four kernels, with everything else fixed, yield approximate overall scores of 1.568×, 1.585× and 1.603×. These are sensitivity calculations, **not forecasts**. The top-10 cutoff last observed was 1.6007× and can change.

A secondary investigation is weight sharing across token tiles using TMA multicast. It should only proceed if profiling demonstrates useful redundant traffic: synchronization can offset savings and hardware may already coalesce duplicate requests.

Sources:

- [NVIDIA Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [CUDA VMM compressible memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#compressible-memory)
- [When and Why to Use TMA](https://arxiv.org/html/2610.34907v1), a workload-specific study; its results are not predictions for this GEMM.
