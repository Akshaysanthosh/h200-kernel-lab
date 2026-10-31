# Experiment ledger

This ledger organizes the investigation by hypothesis. Reports retain detailed configurations, measurements and caveats; they are historical, so earlier “next steps” may already have been completed.

| Approach | What was learned | Evidence |
|---|---|---|
| Early Triton / custom kernels | Established a correct 1.22741× scored checkpoint | [Earlier snapshot](../muse_bf16_proj/champions/2026-10-30_1.227412/) |
| cuBLASLt tuning | Older-stack leads were not consistently transferable; active dispatch must be verified | [Hopper results](../muse_bf16_proj/planning/HOPPER_RESULTS.md) |
| Native overlapped pipeline | Correct implementations did not automatically beat established dense kernels | [Native pipeline](../muse_bf16_proj/native_pipeline/README.md) |
| CUTLASS dense then decoding | First establish competitive dense performance before paying for decoding | [CUTLASS lab](../muse_bf16_proj/cutlass_lab/README.md), [decode lab](../muse_bf16_proj/cutlass_decode/README.md) |
| Public v18 audit/adaptation | Independent H200 validation followed by official 1.52067× scored result | [Audit](../muse_bf16_proj/research/2026-10-30/AUDIT.md), [results](../muse_bf16_proj/planning/OFFICIAL_PUBLIC_V18_RESULTS.md) |
| 2,147 dense configurations | Short-screen leads did not survive full independent confirmation | [Optimization report](../muse_bf16_proj/planning/PUBLIC_OPTIMIZATION_RESULTS.md) |
| 480 compressed WGMMA shape/config evaluations | Wider tiles improved experimental variants but did not beat parent | [Optimization report](../muse_bf16_proj/planning/PUBLIC_OPTIMIZATION_RESULTS.md) |
| Grouped lossless format | Storage savings were outweighed by reconstruction overhead | [Grouped format](../muse_bf16_proj/planning/GROUPED_FORMAT_EXPERIMENT.md) |
| Decoder / exception strategies | Lower apparent instruction or byte counts did not establish an aggregate win | [Vector decoder](../muse_bf16_proj/planning/VECTORIZED_DECODER_EXPERIMENT.md), [exception patch](../muse_bf16_proj/planning/EXCEPTION_PATCH_EXPERIMENT.md) |
| Dense scheduling | Tested persistent/ping-pong/cooperative/Stream-K variants lost substantially to parent | [Scheduling report](../muse_bf16_proj/planning/DENSE_SCHEDULING_EXPERIMENT.md) |
| Transposed weights / padding | Tested alternative layouts did not deliver a confirmed improvement | [Transpose](../muse_bf16_proj/planning/TRANSPOSED_WEIGHTS_EXPERIMENT.md), [padding](../muse_bf16_proj/planning/DENSE_LAYOUT_EXPERIMENT.md) |
| QKV control and geometry | Resolved a dispatch mismatch; geometric changes did not beat the correct control | [Control audit](../muse_bf16_proj/planning/QKV_CONTROL_AUDIT.md), [geometry](../muse_bf16_proj/planning/QKV_GEOMETRY_EXPERIMENT.md) |
| Router interleaving | Tested hi/lo interleaving was slower | [Router report](../muse_bf16_proj/planning/ROUTER_INTERLEAVE_EXPERIMENT.md) |
| Public SK2 specialized correction | Repeatable shape gains, but modest aggregate gain and overlapping official intervals | [SK2 audit](../muse_bf16_proj/research/2026-10-31/vjeux/AUDIT.md) |
| Narrow batch-32 SK2 | 21 distinct schedules across two seeds; no better configuration than the narrow baseline | [Refinement](../muse_bf16_proj/planning/SK2_M32_REFINEMENT.md) |
| Hardware inline compression | Proposed only; no experiment or measured gain yet | [Roadmap](ROADMAP.md) |

## Reading the archive

Start from a report, follow its raw result filename into `muse_bf16_proj/results/modal`, and inspect the matching evaluator in `modal_lab`. Do not combine counts from different reports as if all were independent complete workload runs. Some reports describe screens, others confirmations, and some include overlapping phases.

Raw account inventories, billing records and console logs are omitted. The code, selected JSON measurements, official scored snapshot and source audit records form the public evidence set. `export-manifest.json` records source and exported hashes for the initial curated file set; prose and JSON may have redacted local paths, while candidate code remains exact.
