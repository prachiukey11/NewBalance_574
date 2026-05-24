"""Zero-shot fine-detail anatomy enricher.

Operates as an OPTIONAL second pass on top of the mesh-derived component
list, finding the few parts that aren't separable as 3D mesh components:

  * eyelets       (holes in the eyestay, no surface area)
  * n-logo        (surface decoration painted on the quarter)
  * tongue-label  (small woven badge on the tongue)
  * foxing        (rand strip between upper and midsole)

Pipeline: GroundingDINO (open-vocabulary box detector) -> SAM 2 image
predictor (box-prompted segmentation) -> per-part {bbox, mask, centroid}.

Both models lazy-load and cache for the process lifetime; CPU only for
determinism. The whole module is import-guarded: if `transformers` /
`torch` aren't installed, `pipeline.py` skips this enricher silently and
ships the mesh-only callouts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


_DINO_CACHE: dict = {}
_SAM_PREDICTOR_CACHE: dict = {}

DEFAULT_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
DEFAULT_SAM_MODEL = "facebook/sam2-hiera-small"


# Each tuple: (label_id, prompt_phrase, display_name, description).
# Only the fine-detail parts that mesh segmentation can't separate.
PART_VOCAB: List[Tuple[str, str, str, str]] = [
    ("eyelets",      "lace eyelets",
     "Eyelets",       "Holes for the laces."),
    ("n-logo",       "New Balance N logo",
     "N Logo (Brand Mark)", "The New Balance “N” on the side."),
    ("tongue-label", "tongue brand label",
     "Tongue Label",  "Branding / size label on the tongue."),
    ("foxing",       "foxing strip between upper and midsole",
     "Foxing/Rand",   "Strip between upper and midsole."),
]


@dataclass
class PartDetection:
    label: str               # canonical id from PART_VOCAB
    score: float             # GroundingDINO confidence in [0, 1]
    bbox_xyxy: Tuple[float, float, float, float]
    centroid_xy: Tuple[float, float]  # pixel (x, y), mask centroid if SAM ran
    image_w: int             # source image width (px), for normalisation
    image_h: int             # source image height (px), for normalisation
    mask: Optional[np.ndarray] = None  # HxW bool, None if SAM skipped


def _load_dino(model_name: str = DEFAULT_DINO_MODEL, device: str = "cpu"):
    if "model" in _DINO_CACHE:
        return _DINO_CACHE["processor"], _DINO_CACHE["model"]
    from transformers import (AutoModelForZeroShotObjectDetection,
                              AutoProcessor)
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device)
    model.eval()
    _DINO_CACHE["processor"] = processor
    _DINO_CACHE["model"] = model
    _DINO_CACHE["device"] = device
    return processor, model


def _load_sam_predictor(model_name: str = DEFAULT_SAM_MODEL, device: str = "cpu"):
    if "predictor" in _SAM_PREDICTOR_CACHE:
        return _SAM_PREDICTOR_CACHE["predictor"]
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2_hf(model_name, device=device)
    _SAM_PREDICTOR_CACHE["predictor"] = SAM2ImagePredictor(model)
    return _SAM_PREDICTOR_CACHE["predictor"]


def _run_dino(image: Image.Image, prompts: List[str],
              box_threshold: float, text_threshold: float,
              device: str = "cpu") -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
    import inspect

    import torch
    processor, model = _load_dino(device=device)
    text = ". ".join(prompts) + "."
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])  # (h, w)
    # API moved between transformers versions: older releases used
    # `box_threshold=`, newer releases renamed it to `threshold=`.
    # Pass whichever keyword the installed version actually accepts; if
    # neither, fall through with no threshold kwarg and filter ourselves.
    post = processor.post_process_grounded_object_detection
    sig = inspect.signature(post).parameters
    kwargs = {"target_sizes": target_sizes, "text_threshold": text_threshold}
    if "box_threshold" in sig:
        kwargs["box_threshold"] = box_threshold
    elif "threshold" in sig:
        kwargs["threshold"] = box_threshold
    results = post(outputs, inputs.input_ids, **kwargs)[0]
    out = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if float(score) < box_threshold:
            continue
        out.append((str(label), float(score), tuple(float(x) for x in box.tolist())))
    return out


def _phrase_to_label(phrase: str) -> Optional[str]:
    """Map a GroundingDINO returned phrase back to a canonical label id."""
    p = phrase.lower().strip()
    if not p:
        return None
    for label_id, prompt, _, _ in PART_VOCAB:
        if p == prompt.lower():
            return label_id
    for label_id, prompt, _, _ in PART_VOCAB:
        pl = prompt.lower()
        if p in pl or pl in p:
            return label_id
        words_p = set(p.split())
        words_pl = set(pl.split())
        if words_p & words_pl - {"of", "a", "the", "shoe", "sneaker", "between", "and"}:
            return label_id
    return None


def _mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0.0, 0.0)
    return (float(xs.mean()), float(ys.mean()))


def _bbox_centroid(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def detect_parts(
    image_path: str,
    prompts: Optional[List[Tuple[str, str]]] = None,
    box_threshold: float = 0.28,
    text_threshold: float = 0.22,
    use_sam: bool = True,
    device: str = "cpu",
    verbose: bool = True,
) -> List[PartDetection]:
    """Detect the fine-detail parts on a single rendered view."""
    log = print if verbose else (lambda *a, **k: None)
    if prompts is None:
        prompts = [(pid, prompt) for pid, prompt, _, _ in PART_VOCAB]
    prompt_strs = [p for _, p in prompts]

    image = Image.open(image_path).convert("RGB")
    img_np = np.array(image)
    iw, ih = image.size

    t0 = time.time()
    raw = _run_dino(image, prompt_strs, box_threshold, text_threshold, device)
    log(f"    GroundingDINO: {len(raw)} raw in {time.time() - t0:.1f}s")

    best: Dict[str, Tuple[float, Tuple[float, float, float, float]]] = {}
    for phrase, score, bbox in raw:
        label_id = _phrase_to_label(phrase)
        if label_id is None:
            continue
        if label_id not in best or score > best[label_id][0]:
            best[label_id] = (score, bbox)

    if not best:
        return []

    detections: List[PartDetection] = []
    if use_sam:
        predictor = _load_sam_predictor(device=device)
        predictor.set_image(img_np)
        for label_id, (score, bbox) in best.items():
            try:
                masks, _scores, _ = predictor.predict(
                    box=np.array(bbox), multimask_output=False,
                )
                mask = masks[0].astype(bool)
                centroid = _mask_centroid(mask)
            except Exception as e:
                log(f"    SAM mask failed for {label_id}: {e}")
                mask = None
                centroid = _bbox_centroid(bbox)
            detections.append(PartDetection(
                label=label_id, score=score, bbox_xyxy=bbox,
                centroid_xy=centroid, image_w=iw, image_h=ih, mask=mask,
            ))
    else:
        for label_id, (score, bbox) in best.items():
            detections.append(PartDetection(
                label=label_id, score=score, bbox_xyxy=bbox,
                centroid_xy=_bbox_centroid(bbox),
                image_w=iw, image_h=ih, mask=None,
            ))

    detections.sort(key=lambda d: -d.score)
    log(f"    fine parts: " + ", ".join(d.label for d in detections))
    return detections


def detect_anatomy(
    lateral_path: str,
    medial_path: str,
    box_threshold: float = 0.28,
    text_threshold: float = 0.22,
    use_sam: bool = True,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, List[PartDetection]]:
    log = print if verbose else (lambda *a, **k: None)
    log(f"  ML fine-callout enricher (GroundingDINO + SAM 2, {device})")
    lateral = detect_parts(lateral_path, box_threshold=box_threshold,
                            text_threshold=text_threshold, use_sam=use_sam,
                            device=device, verbose=verbose)
    medial = detect_parts(medial_path, box_threshold=box_threshold,
                           text_threshold=text_threshold, use_sam=use_sam,
                           device=device, verbose=verbose)
    return {"side-lateral": lateral, "side-medial": medial}


def detection_to_callout(det: PartDetection) -> dict:
    """Convert a PartDetection to a normalised centroid frac (cx, cy in [0,1])."""
    cx = det.centroid_xy[0] / max(det.image_w, 1)
    cy = det.centroid_xy[1] / max(det.image_h, 1)
    return {
        "label": det.label,
        "centroid_frac": (float(cx), float(cy)),
        "score": float(det.score),
    }


def display_for(label_id: str) -> Tuple[str, str]:
    """Return (display_name, description) for a canonical label id."""
    for lid, _prompt, disp, desc in PART_VOCAB:
        if lid == label_id:
            return disp, desc
    return label_id, ""
