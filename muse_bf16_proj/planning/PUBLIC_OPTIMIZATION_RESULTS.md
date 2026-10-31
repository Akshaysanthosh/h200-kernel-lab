> Historical lab record. Status and future plans describe the time of writing; see the repository README for the current verified result. Local paths may be redacted.

# Further optimization — October 30, 2026

Goal: 40% less execution time than the original champion. The public-derived
baseline is about 14% lower, so a further ~30% reduction is required.

## Measured bottlenecks

Graph timings below include measurement overhead; do not equate these with
kernel-only hardware roofline estimates or official corrected timings.

| Tokens | QKV us | Pooler us | Unpooler us | Three-layer share of load |
|---:|---:|---:|---:|---:|
| 1 | 14.05 | 18.17 | 8.76 | 69.2% |
| 8 | 14.19 | 18.58 | 9.09 | 69.3% |
| 16 | 14.91 | 18.67 | 18.32 | 72.7% |
| 32 | 16.58 | 20.79 | 20.37 | 73.7% |
| 128 | 18.75 | 25.13 | 23.26 | 75.1% |
| 256 | 22.92 | 28.45 | 28.11 | 75.5% |

## Bounded experiments

1. Dense large-batch cuBLASLt: four layouts, heuristic plans and bounded split-K,
   stage/custom/cluster refinements. BF16 storage and FP32 accumulation required.
2. Lossless large-batch register decoding: reuse the audited decoder with WGMMA
   32/64-token tiles, 1/2/4 warp groups, 1/2 prefetch slots and 1/2/4/8 split clusters.
   Repack and validate exact weight bits. Compare against the public baseline on
   the same H200, challenge CUDA13 stack, cold read-flushed paired graph timing.

Search is lab-only. Any winner needs independent-weight confirmation and then
official free validation before submission. Preserve both existing best files.

## Dense schedule search result

Tested 2,147 valid configurations across six shapes. Four finalists per shape
received 60 paired repetitions and full correctness checks. Short-screen large
gains did not survive confirmation. Best ratios to parent:

- pooler m128: 1.011333x.
- pooler m256: 0.961942x.
- unpooler m128: 0.999645x.
- unpooler m256: 1.024419x.
- qkv_gate m128: 0.922312x.
- qkv_gate m256: 1.000754x.

Two small leads were copied into candidates/public/v18_lt_refined.py for
independent-weight confirmation only. The public parent is unchanged.
Same-config differences (e.g. pooler256) expose allocation/timing sensitivity;
these single-shape search numbers alone are not sufficient for promotion.

## Larger compressed WGMMA tiles, first trial

288 shape/config combinations passed reference checks. All lost to parent.
12 compiled variants: BN32/64, WK1/2/4, prefetch1/2, Q2, R64; four cluster splits.
Longer confirmation winners per shape (largest ratio among finalists):

- pooler m128: 0.8184x; 29.94 vs 24.50 us.
- pooler m256: 0.5549x; 51.17 vs 28.21 us.
- unpooler m128: 0.7819x; 29.36 vs 23.02 us.
- unpooler m256: 0.5494x; 50.86 vs 27.78 us.
- qkv_gate m128: 0.7120x; 25.14 vs 17.95 us.
- qkv_gate m256: 0.5944x; 36.51 vs 21.63 us.

Rejected. Source inspection shows weights decoded repeatedly across token tiles;
the poor m256 scaling is consistent with this overhead, though hardware counters
were not collected and do not establish its exclusive cause. Follow-up tests
BN128/256 instructions to amortize decoding, with larger register/shared-memory
requirements. Do not promote this failed experiment.

## Independent confirmation of dense refinements

Both weight seeds passed 63 normal/outlier/probe/lifetime cases and all 30
graph fresh-input checks. Aggregate ratios versus public parent:

- Seed82731: 1.001410x; A/A control 1.000288x.
- Seed294817: 0.998251x; A/A control 0.999719x.

Rejected: improvement does not repeat. No promotion, no submission, no change
to the scheduled public-parent official check. Result:
results/modal/20261030T112359Z_v18_lt_refined_public_refine.json.

## Dispatch clarification

Tracing the final dispatch chain shows QKV128 already uses the custom `qkvbig`
route, overriding its stored cuBLASLt fallback plan. The initial description
that all six large shapes use dense Lt was too broad. The QKV128 Lt sweep
therefore tested alternatives to that custom kernel and lost; its same-descriptor
comparison is not an A/A test. Pooler128/256, unpooler128/256 and QKV256 use Lt.

## Wider decoded WGMMA follow-up

192 more shape/config combinations passed correctness. Added genuine m64n128
and m64n256 WGMMA instructions to the existing audited generator. All variants
still lost to public parent; no submission candidate was promoted.

| Shape | Parent us | Wide compressed us | Parent / candidate | Registers/thread |
|---|---:|---:|---:|---:|
| pooler m128 | 24.45 | 29.07 | 0.8396x | 140 |
| pooler m256 | 28.19 | 36.93 | 0.7679x | 202 |
| unpooler m128 | 23.01 | 25.31 | 0.9076x | 127 |
| unpooler m256 | 27.55 | 35.90 | 0.7686x | 202 |
| qkv_gate m128 | 17.78 | 23.10 | 0.7682x | 140 |
| qkv_gate m256 | 21.50 | 32.69 | 0.6620x | 140 |

## Decision and next hypothesis

No confirmed improvement this round. Preserve the official 1.227412x champion
and the public-derived baseline (~14% less time in prior paired H200 tests).
The 40% target remains unmet. Existing scheduled official validation still
prioritizes the unchanged public parent; no scored submission made here.

2,147 dense schedule evaluations; 480 compressed shape/config evaluations;
two independent-weight full confirmations of the dense finalist. Four bounded
Modal H200 apps, all on the CUDA13 challenge stack and free credits only.

The raw compressed kernel's generated MMA helper commits and immediately waits
for every four-instruction K64 group, before the next iteration's decode.
A targeted next experiment is a correctly synchronized pipeline that overlaps
decoding/TMA with outstanding WGMMA, with separate live register/shared buffers.
Larger tiles alone improved the experimental kernel but did not win; more
blind tile or Lt sweeps have weak evidence. Hardware stall counters were not
collected, so the serialization/occupancy explanation is a hypothesis grounded
in source and scaling, not a measured hardware-counter diagnosis.

Raw JSON/log files are under results/modal with public_tune, public_compressed,
public_refine and public_wide suffixes. Lab source is saved under modal_lab.
The final successful result was returned before the client-disconnect stop
message; inspect the saved JSON, not just that lifecycle message.
