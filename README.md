# Footwear TechPack Pipeline

Turn a 3D shoe model (`.glb`) into a four-PDF factory-ready tech pack:
multi-view renders, a Bill-of-Materials with dimensions, a colorway sheet
with materials and palette, and a 2D technical drawing set.

```
shoe.glb  →  python main.py shoe.glb  →  output/<shoe-name>/
                                          ├─ 01_views.pdf
                                          ├─ 02_bom_measurements.pdf
                                          ├─ 03_colorway.pdf
                                          └─ 04_techdrawings.pdf
```

---

## What you get

| PDF | Content |
|-----|---------|
| **01 — Multi-view rendering** | Photo-realistic Cycles renders of the shoe from 7+ camera angles (lateral, medial, top, bottom, front, back, three-quarter) on a clean white background. |
| **02 — 3D geometry analysis** | Parts-anatomy infographic (numbered callouts on lateral + medial photos), a full dimensions table (length, width, height, sole stack, heel height, toe spring, ankle opening, sizes US/UK/EU/CM), and a per-component Bill-of-Materials table. |
| **03 — Material / colorway** | Three pages — (1) fabric anatomy with materials labelled per component, (2) color anatomy with hex codes + dominant palette, (3) reference sheet with fabric swatches, parts list, and color palette. |
| **04 — 2D technical drawings** | Black-and-white Freestyle line-art views with dimension callouts (lateral profile, top plan, exploded view, sagittal section). |

---

## Quick start

```bash
# 1. Python 3.11 venv (bpy 4.x is 3.11-only)
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install
pip install -r requirements.txt

# 3. Run
python main.py input/used_new_balance_574_classic______free.glb

# 4. Open the PDFs
open output/used_new_balance_574_classic_free/
```

For a shoe of a different size, pass the actual outsole length in mm:

```bash
python main.py input/your_shoe.glb --target-length-mm 266   # US 8.5
```

`glTF` carries no physical units, so the pipeline scales the model so its
longest axis matches `--target-length-mm` (default `270 mm` = US-9). Every
dimension on the BOM page derives from this number.

---

## How long does it take?

**~5–7 minutes** on Apple Silicon (M1/M2/M3), longer on Intel.

Where the time goes:

| Stage | Time | Why |
|-------|------|-----|
| Blender Cycles renders (7+ photo views + 2 line-art + 2 id-masks + exploded + section) | ~2–3 min | Single-threaded CPU rendering at 1100×800 with 48 samples per pixel — this is what gives byte-identical reruns. Pass `--fast` for multi-threaded (~4× faster, but PNGs are not bit-identical). |
| Mesh segmentation + SAM 2 refinement | ~30 s | SAM 2 (`hiera-small`, ~155 MB) runs once per side view to find tongue / collar / eyestay sub-components inside the upper. |
| CLIP material classifier | ~10 s | OpenCLIP `ViT-B/32` (~150 MB) zero-shot classifies each component into one of 12 material classes. |
| (Optional) GroundingDINO fine callouts | ~30 s | Adds 3–4 extra callouts (eyelets, N logo, tongue label, foxing) that aren't separable as 3D mesh components. Only runs if `transformers` and `torch` are installed. |
| PDF assembly | ~5 s | ReportLab. |

First run downloads model weights into `~/.cache/huggingface/` — budget an
extra minute or two for that, then it's cached.

---

## Requirements

- **Python 3.11** (`bpy` 4.x wheels are 3.11-only — `bpy` is Blender bundled as a Python module, no GUI install required)
- `reportlab`, `numpy`, `Pillow`, `scipy`, `scikit-learn`, `trimesh`, `pygltflib`
- `open_clip_torch` for material classification
- `transformers` + `torch` **optional** — only needed for the fine-callout enricher (eyelets, N logo, tongue label, foxing). The four PDFs render without them.

All in [`requirements.txt`](requirements.txt).

---

## Project layout

```
NewBalance_574/
├─ main.py                 # entry point: `python main.py path/to/shoe.glb`
├─ requirements.txt
├─ input/                  # sample .glb models
└─ src/
   ├─ pipeline.py          # end-to-end pipeline
   ├─ render_blender.py    # Blender Cycles renders (photo views + id-masks)
   ├─ geometry.py          # axis detection, mesh segmentation, dimensions
   ├─ colorway.py          # palette extraction (OKLab K-means) + Pantone/RAL/CMYK matching
   ├─ material.py          # CLIP zero-shot + PBR-heuristic material classifier
   ├─ segmentation_sam.py  # SAM 2 mesh refinement (tongue / collar / eyestay)
   ├─ anatomy_detect.py    # optional GroundingDINO fine-callout enricher
   ├─ techdrawings.py      # Freestyle line-art views + dimension callouts
   └─ pdf.py               # ReportLab layouts for all four PDFs
```

---

## CLI flags

```
main.py shoe.glb [--target-length-mm F]      # the simple entry point

python -m src.pipeline --glb PATH --out DIR \
    --model-id ID --model-name "Name"        # full CLI
    [--designer "..." --factory "..." --season "..."]
    [--seed N] [--upper-clusters K] [--palette-colors K]
    [--target-length-mm F]                   # default 270 = US 9
    [--render-w 1100 --render-h 800]
    [--samples 48]                           # Cycles samples / pixel
    [--fast]                                 # multi-threaded Cycles (~4× faster)
    [-q]                                     # quiet
```

---

## Determinism

Same input + same seed + same output path = byte-identical PDFs across
reruns (default mode). Pass `--fast` to trade determinism for ~4× speed.

The output directory is auto-cleaned at the start of each run — stale
`*.pdf` files from earlier runs are deleted before the new four are
written, so the output set is always exactly four files.

