# Licensing review — third-party assets (before any commercial release)

Reviewed 2026-09-29 (tasks.md backlog item "Licensing review"). Sources are the official licence / terms pages quoted
below, read on that date. **Not legal advice** — the blockers need a lawyer's sign-off before a store release.

## Summary

| Asset | Where | Licence | Commercial use | Status |
|---|---|---|---|---|
| MATT avatar (Avaturn) | `SourceArt/test_MATT_Avaturn.glb`, `Assets/PoOlympic/Art/MATT.glb` | Avaturn Terms of Use (26 Jan 2023) | conditional | **action needed** |
| ZOMBIE + GRANDMA meshes (likely Hunyuan3D output) | `Assets/PoOlympic/Art/Zombie.glb`, `test_GRANDMA_riggedTenCent.glb` | Tencent Hunyuan 3D Community License | not in EU / UK / South Korea | **blocker for a worldwide release** |
| ZOMBIE rig (ActorCore AccuRIG) | `test_ZOMBIE_RiggedAccurig.fbx` | AccuRIG FAQ: free for personal and commercial use | yes | ok (the mesh's own licence applies) |
| Stadium textures (Poly Haven) | `SourceArt/Stadium/fetch_textures.py` | CC0 | yes, no credit needed | ok |
| Inter font | `Assets/PoOlympic/UI/Fonts/` | SIL OFL 1.1 | yes | ship `Inter-LICENSE.txt` |
| MuJoCo 3.11 + Unity plug-in | `Packages/org.mujoco`, `Assets/Plugins/MuJoCo/` | Apache-2.0 | yes | ship licence + note our patches (`POOLYMPIC_PATCHES.md`) |
| Unity Inference Engine | `com.unity.ai.inference` | Unity ToS + its Third Party Notices | yes | include its notices |
| glTFast | package | Apache-2.0 | yes | ship licence |
| Input System | package | Unity Companion License | yes | ok |
| "Olympic" rings / cauldron / name | `SourceArt/Stadium/build_dressing.py`, stadium signage, "POOLYMPICS" | protected marks (e.g. 36 U.S.C. §220506) | **no** without consent | **blocker** |

## Details

**Avaturn (MATT).** Terms: "You may use the avatars for commercial purposes only after notifying us at
hello@avaturn.me with link to the project"; attribution required ("give appropriate credit, provide a link to Avaturn
service, and indicate if changes were made"); commercial use with revenue ≥ USD 1M needs an enterprise licence;
ownership of the avatar stays with Avaturn, who may ask for use to stop. Modification is only allowed "as expressly
set forth" — re-rigging MATT into a physics body should be confirmed in writing. If MATT was made from a real person's
selfie, a **likeness release** from that person is needed (not covered by the terms). The outfit (`M_Military_Uniform`)
may carry its own third-party licence.
Source: Avaturn Terms of Use (linked from avaturn.me), https://avaturn.me/pricing/

**Hunyuan3D.** Tencent Hunyuan 3D 2.0 / 2.1 Community License: "Territory" = worldwide **excluding the European Union,
United Kingdom and South Korea**; §5(c) forbids using, distributing or displaying the works *or Output* outside the
Territory. Tencent claims no rights in Outputs (§4(d)); Output may not be used to improve other AI models (§5(b)). The
1M-MAU clause is measured at the model's release date (not an ongoing cap). The repo does not record which model /
version / service generated GRANDMA and ZOMBIE (image names `texture_pbr_*`, single `node_0` mesh point to Hunyuan);
hosted-service generations (web studio, Tencent Cloud API) fall under a different service agreement — **unverified**.
Sources: https://huggingface.co/tencent/Hunyuan3D-2/blob/main/LICENSE, https://huggingface.co/tencent/Hunyuan3D-2.1/blob/main/LICENSE

**Olympic marks.** The five interlocking rings, "Olympic" and "Olympiad" are reserved in the US (36 U.S.C. §220506) and
protected elsewhere; the stadium dressing builds a five-ring emblem in the official colours, an Olympic cauldron and
"POOLYMPICS" signage. Store review / takedown risk.

## Checklist before a commercial release

1. Remove the five-ring emblem from `build_dressing.py` (re-export the stadium) and get legal advice on the
   "PoOlympic / POOLYMPICS" name.
2. Hunyuan3D: confirm tool + version for ZOMBIE and GRANDMA; replace them with owned / licensed meshes, **or** exclude
   the EU, UK and South Korea from Play Store and Windows distribution.
3. Avaturn: notify hello@avaturn.me with the store links; get written OK for commercial use and re-rigging; check the
   outfit's licence.
4. Likeness release, if MATT is based on a real person.
5. In-game credits screen: Avaturn credit + link + "modified"; optionally "Powered by Tencent Hunyuan".
6. `THIRD_PARTY_NOTICES` in the build: MuJoCo (Apache-2.0 + its bundled deps: qhull, lodepng, tinyxml2, …), Inter
   (OFL 1.1), glTFast (Apache-2.0), Unity Inference Engine third-party notices, Hunyuan Notice if those assets ship.
7. Check that the Unity plan's revenue cap fits the business.

Could not verify: the ZOMBIE mesh's origin; the Hunyuan version / channel and hosted-service terms; MuJoCo's full native
dependency notice list; Avaturn free-tier limits (pricing page is script-rendered).
