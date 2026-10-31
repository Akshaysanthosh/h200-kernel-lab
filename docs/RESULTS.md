# Results and evidence

All values below are historical observations, not live leaderboard queries.

| Version | Run type | Speedup | 95% interval | Decision |
|---|---|---:|---|---|
| Earlier champion | Scored, `35d40255820f4df4` | 1.2274118693× | Consult saved result | Preserved |
| Public-v18 adaptation | Free check, `c264b6bc9cd44119` | 1.5169394860× | [1.4941853937, 1.5396935783] | Validated |
| Public-v18 adaptation | Scored, `19ffcee8e7894be2` | 1.5206695979× | [1.4978595539, 1.5434796418] | Current scored best |
| SK2 hybrid | Free check, `baeb0b4f6b824ec2` | 1.5348531307× | [1.5118303338, 1.5578759277] | Not promoted |

The current champion passed 9/9 tests with no reported integrity flags. Rank at submission was 270/1515; subsequent ranks and final placements may differ.

## Exact source identity

| Artifact | SHA256 |
|---|---|
| Current scored champion | `6faa2ae8519e3b9c19b27c2fdf47a2a057c7be89c72583ecfd270175c2144ae7` |
| Earlier champion | `9ea10c81639eb23d6c9eb832465190328664fd655e89e9ff81732b94e10b8a36` |
| SK2 hybrid | `4ee96b46c21a7fe096c456684519b32353ce472a4e369b4aa08023c2bb83f2c9` |
| Narrow SK2 | `11822cbfddd312660603b72e21d4f2c87591995bd4829f11d30b86a6c2fb4f7b` |

## Primary evidence

- [Champion scored result](../muse_bf16_proj/champions/2026-10-31_1.520670/result.json)
- [Champion metadata](../muse_bf16_proj/champions/2026-10-31_1.520670/metadata.json)
- [Champion source](../muse_bf16_proj/champions/2026-10-31_1.520670/submission.py)
- [Official profile](../muse_bf16_proj/candidates/public/profiles/nathen_v18_linear.19ffcee8e7894be2.txt)
- [Official validation narrative](../muse_bf16_proj/planning/OFFICIAL_PUBLIC_V18_RESULTS.md)
- [SK2 official check](../muse_bf16_proj/candidates/public/profiles/v18_sk2_hybrid.baeb0b4f6b824ec2.json)
- [Narrow candidate full confirmation](../muse_bf16_proj/results/modal/20261031T063152Z_v18_sk2_m32_public_refine.json)

## Read the percentages correctly

`1 - 1 / 1.5206695979 = 34.24%` lower time versus the challenge baseline. This does not mean 52.07% lower latency. Nor does it imply a 34.24% whole-model speedup: the challenge only covers these projections and its specified load mixture.

The narrow SK2 candidate's local parent/candidate ratios were 1.0085636 and 1.0041470, with A/A controls 0.9996521 and 0.9958261. This modest margin did not justify a new scored submission.
