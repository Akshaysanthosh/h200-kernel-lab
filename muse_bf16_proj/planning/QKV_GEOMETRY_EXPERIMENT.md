> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# Official-profile-guided QKV geometry experiment — 2026-10-31

## Protected baseline and target

Official best: 1.5206695978723808x, job 19ffcee8e7894be2; snapshot champions/2026-10-31_1.520670/submission.py, SHA256 6faa2ae8519e3b9c19b27c2fdf47a2a057c7be89c72583ecfd270175c2144ae7. Previous 1.227412x snapshot and original main submission.py remain unchanged.

Live top-ten cutoff at experiment planning: 1.6003x, approximately 4.98% lower effective execution time required. Positions are provisional. Official profile source: candidates/public/profiles/nathen_v18_linear.19ffcee8e7894be2.txt; structured measurements saved in results/local/official-opportunity-ranking.json.

Hypothetical aggregate multiplier if one layer alone reached the reference throughput floors across six workloads: QKV 1.0718, router 1.0614, unpooler 1.0512, O 1.0458, pooler 1.0381. These are rough prioritization indicators, not predictions or strict lower bounds (compressed kernels can transfer fewer bytes than the dense bound assumes). They do not account for achievable algorithmic overhead. QKV M256 takes 18.34us vs reference 9.77us; M128 takes 13.98us vs 8.42us. These are promising gaps after multiple unsuccessful pooler/unpooler pipeline trials.

## New hypothesis

Change the logical GEMM's output width from 2304 to 2560, 3072, or 4096, with exact original BF16 weights and extra zero rows prepared untimed. Return a view of the first 2304 outputs. This adds real work and bytes; the hypothesis is that an improved cuBLASLt tiling/occupancy choice can outweigh that cost. It is not the previous leading-stride-only padding test. No runtime output copy is required. Output need not be contiguous under TASK.md, but it must preserve shape, precision and fresh-input correctness.

Test both ordinary and pretransposed weights with all four existing Lt orientations. Include N2304 controls and the parent's restored algorithm where supported, plus up to 12 heuristic choices per layout. H200 CUDA13 challenge image, one 900-second allocation, no retries or persistent volumes. Before run: $23.85 free credits, $0 charges, $0 spending cap directly verified.

All supported cases must pass reference correctness before timing. Cold-L2 read flush inside graph and outside timing events; paired reversed-order timings, A/A controls, top three 100-replay finalists, independent weight seed 120-replay recheck without retuning. Fresh-input replay and outliers/probes included. These are raw local events, not judge scores.

A candidate would require full-workload/nearby-shape validation and official free checks before submission. Changes to these two QKV shapes alone cannot reasonably be claimed to close the top-ten gap: reaching both reference bounds yields only about 3.3% overall score gain. A portfolio of wins is likely necessary.

## Outcome: no promotion

384 supported configurations passed reference correctness; six finalists passed fresh-input graph checks. Both selected configurations on independent weights had the original 2304 output width.

| M | Width | Parent us | Candidate us | Parent/candidate | 95% interval |
|---|---:|---:|---:|---:|---|
| 128 | 2304 | 18.080 | 19.264 | 0.9367 | [0.9316, 0.9423] |
| 256 | 2304 | 21.440 | 21.312 | 1.0034 | [0.9964, 1.0102] |

Measurement caveat: M128 original-width/same-algorithm control ran about 6–8% slower than the protected parent. Independent allocations, weight pointers and extension contexts differ; the cause is unresolved. Algorithm/descriptor source review found no clearing of restored algorithms during enumeration. No claim that all geometry variants are ruled out is supported. No tested padded candidate beat the protected baseline. M256 original-width control was at parity on independent weights.

No candidate promoted, no official check or scored submission spent. New 1.52067x champion, public source, old 1.2274x champion and main submission.py hashes remain unchanged. Raw result: results/modal/20261031T054457Z_nathen_v18_linear_qkv_geometry.json.

Post-run dashboard: $23.70 credits remaining, $0 charges, $0 spending cap; reporting may lag. All GPU apps stopped.

Follow-up correction: source inspection identifies a custom WGMMA route for QKV M128, so the restored Lt plan was not an identical-route control. See QKV_CONTROL_AUDIT.md for the controlled validation and retest.
