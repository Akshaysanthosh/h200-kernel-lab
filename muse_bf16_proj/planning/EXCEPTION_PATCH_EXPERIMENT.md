> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# Predicated exception repair — 2026-10-30

## Difference from previous experiments

Keep the protected public parent's exact packing, unrestricted exponent window, exception count, and vector loads. Replace the lane-divergent register repair helper with explicit PTX predicates selecting the destination lane and register. The correction remains the original XOR of exact BF16 high-byte bits. The existing iteration-ordered exception list and warp shuffle traversal remain intact.

This isolates control flow for sparse corrections. It introduces neither a new format nor the additional exceptions responsible for the aligned-prefix decoder regression. Register selection is unrolled for 16/32 registers; no dynamic register array indexing or approximate weights.

## Checks and budget

- Local semantic audit covers 12,288 exception patterns, with every destination lane/register and both BF16 halves checked.
- H200 comparison uses the challenge CUDA 13 image, two independent CUDA seeds, normal/outlier inputs at M=1/7/8, and fresh graph activations.
- Pooler M=1 and O M=8 are compared against their unchanged protected route. Eight stage/block variants, alternating read-flushed graph timing, 80 samples, paired bootstrap intervals and A/A controls.
- Both best source hashes must match `results/local/format-experiment-protected.json` before and after.
- Before-run Modal dashboard: $24.85 free credits, $0 charges, $0 spending cap. One H200, 900-second maximum, no retries or persistent storage.
- Lab files are not submission candidates. Promotion requires independent aggregate validation and official correctness/timing checks.

## Outcome

Small-shape run completed: `results/modal/20261030T133352Z_nathen_v18_linear_exception_patch.json`. All 96 correctness cases passed; exception counts exactly match the original on both shapes and both seeds.

- Pooler original geometry: 1.00137× and 1.01190×. The first interval overlaps parity, and A/A ratios are ~1.003. Do not promote pooler.
- O M=8, original stage-2 geometry: 1.01422× [1.00441,1.02248] and 1.02149× [1.01149,1.03039]. This warrants a full-workload check but is not an aggregate gain.
- Other configurations lose or overlap parity.

Built a separate provisional candidate `candidates/public/v18_predicated_o8.py` changing only the O M=8 route. The protected parent and official champion remain unchanged. Full independent validation is pending; do not submit this candidate.


## Final independent validation: rejected

`results/modal/20261030T133456Z_v18_predicated_o8_public_refine.json`:

| Seed | Aggregate candidate/parent speedup | Identical-parent A/A | O M=8 alone |
| --- | ---: | ---: | ---: |
| 82731 | 1.000434 | 0.998514 | 0.996700 |
| 294817 | 0.999309 | 0.998894 | 1.006617 |

The early O signal does not reproduce consistently, and aggregate differences are within noise. All 63 correctness configurations passed on each seed, with normal/outlier inputs, nearby shapes, retained output checks and changed-activation graphs. First candidate preparation took 7.2 seconds with compiled parent kernels already cached; the parent's cold preparation took 88.6 seconds in this lab run.

No promotion or submission. The provisional candidate is explicitly marked REJECTED in the manifest. Both protected sources remain unchanged. Two bounded free-credit applications completed.

Post-run dashboard: $24.67 free credits remain, $0 charges, $0 cap. Displayed credit usage increased $0.18; final accounting may lag. No active Modal applications remain.
