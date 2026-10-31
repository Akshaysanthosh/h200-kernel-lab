# Provenance, attribution and publication boundaries

## Major upstream components

| Component | Origin | License / record |
|---|---|---|
| Scored v18 adaptation and derivatives | NathenFernandes/muse-bf16-proj, solution branch | MIT; [retained license](../licenses/Nathen-MIT.txt), [source provenance](../muse_bf16_proj/research/2026-10-30/provenance.json) |
| SK2 implementation | vjeux/okc-2026, revision `f35a41ee1c13e316dabbf772830d47b23b453393` | Apache-2.0; [license](../licenses/vjeux-Apache-2.0.txt), [audit](../muse_bf16_proj/research/2026-10-31/vjeux/AUDIT.md) |
| CUTLASS experimental headers | NVIDIA CUTLASS v3.9.2, `ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e` | BSD-3-Clause; [license](../licenses/CUTLASS-BSD-3-Clause.txt), [pin](../muse_bf16_proj/cutlass_lab/dependency.json) |
| FlashInfer / tinygemm research material | Recorded upstream source in research archive | Apache-2.0; [license](../licenses/FlashInfer-Apache-2.0.txt) |
| Task, baseline, reference and checks | Open Kernel Challenge starter files | Challenge-authored material; preserved notices and challenge terms apply; no new license is asserted here |

The upstream v18 downloaded source hash is `4fe91632d071d64dec8b54085a95ca5ebc8b7235a166460861ee2051cdb08ecc`. The standalone adapted source hash is `6faa2ae8519e3b9c19b27c2fdf47a2a057c7be89c72583ecfd270175c2144ae7`. They are distinct artifacts. This project does not claim byte-identical reproduction of the upstream author's reported score.

## Audit transparency

The parent contains GPU clock-reading helpers used as compiler control-flow boundaries. Those were explicitly discussed in the source audit and were not removed or disguised for this publication. The official scored artifact passed with no reported flags, but that is not a formal proof or a final challenge integrity review.

The upstream SK2 research source also contains other kernel families and debug code. The integrated hybrid extracts the required SK2 dependency closure; its audit explains what is and is not invoked. Read the audit rather than assuming every routine in the archived upstream file is part of the submitted candidate.

## Snapshot policy

The portfolio starts a fresh Git history so workstation history, credentials and unrelated files are not inadvertently published. Protected candidate bytes match the originals. Historical prose/results have local path redactions and archive banners. Raw logs, account inventories, private research attachments, dashboard screenshots, billing gates, virtual environments and downloaded dependency trees are not included.

The archive preserves failed and superseded experiments for inspection. Its contents do not imply that every archived approach is recommended, accepted by the judge, or portable across library versions.
