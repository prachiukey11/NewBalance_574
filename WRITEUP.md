# Footwear TechPack — Write-up

## (a) Which 3 capabilities I picked and why

| # | Capability | Why this one |
|---|---|---|
| 1 | **Geometry analysis** (Bill of Materials + dimensions) | The deepest 3D-specific work in the brief. It is what justifies starting from a `.glb` instead of a photo. Once components are segmented, length, width, height, sole stack, heel height, toe spring, instep girth, ankle opening, and the US/UK/EU/CM size all fall out of the same axis-detection layer. Every downstream page builds on this. |
| 2 | **Material / colorway extraction** | What a sourcing manager opens first. Per-component dominant colour with Pantone TCX + RAL + CMYK matches is near-free once the mesh is segmented. The visual payoff (Fabric Anatomy + Color Anatomy infographics) is high for the work it costs. |
| 3 | **2D technical drawings + dimension callouts** | The page a pattern engineer actually sends to the factory. Blender Freestyle produces clean black-on-white line-art once the cameras exist; ReportLab draws the dimension lines as plain 2D primitives, so the labels stay sharp at any zoom. |

Multi-view rendering became the shared foundation rather than a fourth pick — every capability above needs orthographic + three-quarter renders, so it had to be solid before any of the three were possible. That gives **four PDFs** in the output set (`01_views`, `02_bom_measurements`, `03_colorway`, `04_techdrawings`) but really three capabilities of work.

I cut **LLM-grounded construction notes** and **pattern unfolding**. Construction notes is a thin wrapper on top of the geometry + material output — useful but not what distinguishes a 3D pipeline from a templated word doc. Pattern unfolding is a real research problem (UV-based flatten + seam detection + grading); bolting it on superficially would have hurt the rest of the work. Both are described in §(c) with how I would actually approach them.

## (b) Overall approach

The pipeline is a small linear DAG. Each stage is deterministic and reproducible — same input + same seed = byte-identical PDFs.

```
.glb ─► Blender import + axis detection
         ├─► photo renders (7 views)
         ├─► line-art renders (lateral, top)
         └─► id-mask renders (per-component flat colours)
                  │
                  ▼
        mesh segmentation (spatial + k-means + SAM 2 refinement)
                  │           ──► tongue, collar, eyestay sub-components
                  ▼
        per-component analysis
          ├─► dimensions (geometry.analyze)
          ├─► palette  (OKLab k-means, Pantone/RAL/CMYK match)
          └─► material (OpenCLIP zero-shot + PBR heuristics)
                  │
                  ▼
        ReportLab → 4 PDFs
```

Two design choices worth flagging:

- **Mesh-first, ML-optional.** The BOM and colorway anatomy pages always render from mesh-derived data (id-mask centroids, CLIP materials, per-component palettes). A second optional pass with GroundingDINO + SAM 2 finds parts that aren't separable as 3D mesh components — eyelets, the New Balance "N" logo, tongue label, foxing — and adds them as fine callouts. The pipeline runs end-to-end without those ML packages installed; you just see fewer parts on the legend.
- **The "vibrant" palette.** A component that mixes blue mesh with gray suede averages to gray when you area-weight the colour. So instead of the area-weighted dominant, the colorway page uses the **most-saturated** swatch from each component's 3-5 colour mini-palette. That is how blue is recovered from the toe-box mesh and the N logo even though they are small regions.

## (c) What I would do with another week

- **Better callout positioning.** Right now most callouts (1, 2, 3, 4, etc.) land on the correct part, but a few — typically the tongue, heel collar, and N logo — drift a few mm above the visible silhouette because the mesh's UV island for those parts wraps inside the throat opening. With another week I would: (a) train a small YOLOv8-seg model on labelled lateral / medial shoe photos (~200 images is enough for one shoe family) so the callout anchor comes from a learned 2D mask instead of a projected mesh centroid; (b) add a simple "snap-to-nearest-edge" pass so a marker that lands in white space is pulled to the closest visible silhouette pixel.
- **LLM-grounded construction notes** (the capability I skipped). I have the structured input — per-component geometry, materials, and area shares. I would prompt Claude with that JSON, force it to cite every claim with a component name (`[heel-counter]`) or a measurement (`[heel_height=41.8mm]`), and gate the output on a regex check so untraceable claims are rejected. Two days of work given the inputs already exist.
- **Pattern unfolding** (the other capability I skipped). The right approach is per-component UV-unwrap with `xatlas` (open-source LSCM), then a seam-detection pass on the boundary curves to merge UV islands that share a stitch line. Output: one DXF per component at 1:1 scale, ready to feed a Gerber or Lectra cutting plotter. This is a week of work on its own — that is why I chose not to fake it.
- **Custom-trained material classifier.** Today's CLIP zero-shot occasionally calls mesh "canvas" because it has no fashion-specific fine-tuning. A LoRA on ~500 labelled patches from footwear datasets would push confidence well above 0.9 on the common materials (suede, mesh, leather, EVA, rubber).
- **Auto size detection.** Right now `--target-length-mm` is required to get correct millimetres (default 270 = US-9). Given a reference object in the scene or a foot-pose prior, the pipeline could auto-detect.

## (d) Where I cut corners and what I'd harden for production

| Corner cut | Production hardening |
|---|---|
| Hardcoded canonical-name → display-name maps in `pipeline.py` and `pdf.py` | Move to a `vocab.yaml` so other footwear types (boots, sandals, heels) drop in without code changes. |
| Single-threaded Cycles rendering for byte-determinism | Replace `--fast` with proper tile-level seeding so determinism survives multi-threading. ~4× pipeline speed-up. |
| `_compute_centroids_from_id_mask` does largest-connected-component, but ties between two near-equal regions are not handled | Add area-weighted centre-of-mass across the top-K largest regions, weighted by inverse distance from the component's expected anatomical position. |
| No unit tests on `geometry.py` / `colorway.py` — only the integration test (the four PDFs) covers them | Add pytest coverage on the pure functions: axis detection, K-means determinism, Pantone-match ΔE thresholds. CI on every push. |
| Stale-PDF cleanup is a simple `glob + unlink` at the top of `run()` | Add a `--keep-stale` flag for callers that diff outputs across pipeline versions. |
| Single shoe model in / single PDF set out | Batch mode that walks a folder of `.glb` files, writes a master index PDF, and skips re-renders whose source hash matches. |
| The optional GroundingDINO + SAM 2 path adds ~2 GB of weights for ~4 extra callouts | Distil a small task-specific head on top of a frozen DINOv2 backbone — same recall at ~80 MB. |

## (e) Assumptions about input and the factory consumer

**Input.** A single `.glb` or `.gltf` with a UV-mapped diffuse texture, sitting roughly on the ground plane with the foot pointed along one axis. The pipeline does run `axis_map` detection so the orientation does not need to be exact, but a left vs. right shoe is not auto-flipped — the lateral side is whichever face has the New Balance logo if one exists, otherwise the side with the higher mesh density. The model is assumed to be a single shoe (not a stereo pair or a posed figure wearing the shoe).

**Scale.** glTF carries no enforced physical units, so the pipeline scales the longest axis to `--target-length-mm` (default 270 mm = US-9). Every measurement in the BOM PDF derives from this number. The consumer is expected to know the shoe's real outsole length and pass it.

**Factory consumer.** Three personas drive the four PDFs:

- **Pattern engineer** — opens `04_techdrawings.pdf` for lateral profile, plan, exploded view, sagittal section. Wants line-art at 1:2 scale with dimension labels in millimetres and tolerances stated explicitly (`±1 mm`).
- **Sourcing manager** — opens `03_colorway.pdf` page 1-2 for fabric + colour anatomy and page 3 for the Parts List table (every component with material + function). Wants Pantone TCX and RAL Classic references so factories on different colour systems can both quote.
- **Product / merchandising** — opens `02_bom_measurements.pdf` for the headline anatomy infographic and the full dimensions table, and `01_views.pdf` for hero shots for the line sheet. Wants something printable and self-explanatory without reading any code.

No persona should have to learn the pipeline's internals to consume the output — that is why the PDFs are kept small (one purpose per file), consistently styled, and never reference the model id or git hash on a page the factory sees.
