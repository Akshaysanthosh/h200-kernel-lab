> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# Vectorized aligned-prefix decoder experiment — 2026-10-30

## Hypothesis

The prior compact three-plane layout reduced stored bytes but replaced vector accesses with gathers and lost substantially. This experiment retains the parent's three 128-bit weight loads per 32 weights, stream order, sparse XOR exception representation, tensor-core accumulation, activation accesses, and reduction.

The already-tested arithmetic decoder is not simply rerun. A new packer selects an eight-aligned window of the BF16 high-byte exponent bits. That alignment makes addition equivalent to bitwise OR on common weights, allowing the common decoder to use two fused mask/OR operations, a rotate, and four byte permutations. This is independently implemented; no NF12 code is imported and no NF12 performance claim is assumed.

The trade-off is more exceptions. The existing best eight-value window is unrestricted, while the new one must start at a multiple of eight. Full BF16 precision is retained through the same high-byte XOR patches. A lab copy of the stream builder allocates 64 exception records per warp rather than 32; the decoder refills its exception registers at the 32-entry boundary. Overflow is rejected, not silently truncated.

## Protection and validation

Neither `submission.py` nor `candidates/public/nathen_v18_linear.py` is edited. Expected hashes remain those in `results/local/format-experiment-protected.json`.

- Exhaustive proof over 65,536 BF16 bit patterns × 16 aligned bases: 1,048,576 cases.
- 65,536 randomized packed-word checks, including cross-byte rotation and signs.
- Full-size CPU pack/unpack round trips for pooler and O on two seeds.
- GPU: independent CUDA seeds 82731 and 294817, normal and outlier inputs at M=1,7,8, changed-input graph replay.
- Timing: two original protected v18s routes (pooler M=1 and O M=8), alternating read-flushed CUDA graphs, 80 paired samples, paired bootstrap intervals, A/A controls.
- Pooler native and two smaller-block geometries, stages 1/2; O native geometry, stages 1/2.
- SASS inspection records vector load and integer operation counts per instantiated kernel. Whole-kernel static counts are not per-weight costs or a runtime profile.

Local sampled weights select high-byte base 56. Pooler has 73,719/73,744 exceptions on the two CPU seeds; O has 6,478/6,417. CPU and CUDA random streams differ, so GPU packing is independently checked.

## Budget

Modal dashboard before execution: $24.88 free credits, $0 charges, $0 spend cap. One H200, no persistent volume, no retries, 900-second timeout. No promotion without an independent aggregate improvement and passing official checks.

## Outcome

**Rejected: no improvement on either independently generated CUDA weight seed.** Lab-only files are not submission candidates. Neither protected best source was changed and nothing was submitted.

Result: `results/modal/20261030T132903Z_nathen_v18_linear_vector_decoder.json`.

| Seed | Shape | Best measured variant | Parent µs | Candidate µs | Paired ratio (parent/candidate) |
| --- | --- | --- | ---: | ---: | ---: |
| 82731 | pooler M=1 | original geometry, stage 1 | 17.824 | 25.024 | 0.70634 |
| 294817 | pooler M=1 | original geometry, stage 1 | 17.792 | 25.664 | 0.69130 |
| 82731 | O M=8 | original geometry, stage 2 | 9.280 | 10.656 | 0.87434 |
| 294817 | O M=8 | original geometry, stage 2 | 9.312 | 10.592 | 0.87964 |

All 96 normal/outlier/nearby-shape test cases passed (eight variants × two seeds × six cases), along with exact packing checks and fresh-input graph replay. The 1,048,576-case scalar and 65,536-case packed-word CPU audits also passed on the remote runtime. All paired intervals were below parity; no aggregate promotion test was justified.

Pooler exceptions increased from 4,599 to 74,065 on seed 82731 and from 4,661 to 74,425 on seed 294817. O exceptions increased from 955 to 6,463 and from 947 to 6,466. The 64-entry handling was exercised: pooler lists reached 33 entries on the original geometry and up to 60 in larger warp tiles.

SASS confirms retained LDG.128 accesses and reduced PRMT counts: corresponding stage-1 small kernels show 24→16 static PRMT instructions. The best pooler configuration remained at 58 registers and zero local-memory spill. Some stage-2 configurations spilled, but spill-free configurations also lost. More exceptions are the main structural regression consistent with the timing; this run did not collect counters establishing an exclusive causal attribution.

The result narrows the next hypothesis: cheaper common decoding must not substantially worsen exception coverage. Keep the unrestricted exponent window and investigate exception handling or instruction scheduling only when there is a concrete new mechanism. This negative result does not establish that every vectorized decoder would lose.

The single bounded application completed. Before-run credits were $24.88 with $0 charges and a $0 cap; post-run billing can lag.

Post-run dashboard: $24.85 remaining free credits, $0 charges, $0 spend cap. Displayed credit use increased by $0.03; reporting can lag. CLI verified the experiment stopped and no active apps remained.
