> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# Lossless decoding inside CUTLASS SM90a

Experimental pooler M=16, N=4096, K=8192 only. Not an official submission.
The official champion and hybrid_v1 remain unchanged.

The overlay preserves CUTLASS 3.9.2 BSD notices and modifies its TMA collective:
A weights use 8-bit sign/mantissa plus 4-bit block-relative exponent; blocks
that cannot represent every original BF16 value fall back to the original A.
The producer reconstructs vector groups directly into the GMMA-swizzled shared
buffer. B still uses TMA. A proxy fence and transaction-barrier completion
publish each decoded tile to WGMMA consumers. FP32 partials are reduced to BF16.
Packing is input-independent preparation; all decoding is timed.

v1 uses one producer warp. Its archived collective is in v1/; to reproduce,
use that file in the overlay and omit the kernel overlay. v2 uses all four
producer warps and a dedicated named barrier (user slot 0, hardware slot 8).
Current include/ contains v2. Dependency and overlay hashes: dependency.json.

Tests: full-matrix bitwise reconstruction, outliers and exact probe rows,
fresh inputs, retained output lifetime, fresh-input CUDA graph replay, cold-L2
paired timings, and top-two confirmation with independent weights. Timing
includes the final split-K reduction.
The same binary's ordinary TMA path (config 2, split 4) and unchanged hybrid_v1
are controls. Packing allocation is larger than dense storage because original
weights remain for fallback; only read traffic is potentially reduced.

Build: python3 tools/build_decode_candidate.py
Run: modal_lab/run_candidate.py candidates/hopper/cutlass_decode_v2.py --mode decode --execute
The runner requires a freshly verified $0 spend limit and free credit balance.

## Results (2026-10-30, H200)

24 configurations plus four independent-weight confirmations passed. Full
packed matrix reconstruction was bit-exact on both seeds; fallback block
fractions were 1.5503% and 1.5022%. No 10% improvement was achieved.

| Confirmed configuration | Fused decode + GEMM + reduction | Saved best |
|---|---:|---:|
| v1, 1 producer warp, 2 stages, split 16 | 213.504 us | 29.888 us |
| v2, 4 producer warps, 2 stages, split 16 | 75.616 us | 29.936 us |

Same-binary uncompressed control: 30.240 us in the v2 confirmation.
Required 10% time-reduction target: about 26.94 us. Thus v2 still takes 2.53x
as long as the saved best. Four-way parallel decoding helped substantially,
but did not overcome producer load/decode/synchronization overhead. This is a
mechanistic inference from the paired experiment, not a hardware-counter profile.
Neither revision spills registers (reported LOCAL:0); WGMMA is present.

Evidence: results/modal/20261030T103859Z_cutlass_decode_v1_decode.json and
results/modal/20261030T104210Z_cutlass_decode_v2_decode.json. One initial compile
failed on a fence namespace; a v2 launch was stopped to correct a named-barrier
slot before the final run. That stop raced with completion and produced an
additional exploratory result (20261030T104156Z). It reports 74.544 us, but is
excluded from the comparison because the overlay was being corrected during
that launch; the subsequent clean launch is the evidence used above. The 24
configurations/four confirmations count excludes that ambiguous run. All Modal
apps are stopped. Dashboard credits
went from $28.09 to $27.59; charges and spending limit remain $0.

Decision: do not promote or submit. Preserve both existing best versions and
scheduled official validation. Further work would need a substantially different
producer pipeline (e.g. asynchronously stage compressed streams before decoding),
not another small tile sweep. Such an approach remains untested.
