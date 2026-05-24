#!/usr/bin/env python3
"""Single-arg entry point.

    python main.py path/to/shoe.glb [--target-length-mm 266]

Writes the four PDFs to output/<shoe-name>/:
  01_views.pdf            multi-view photo renders
  02_bom_measurements.pdf 3D geometry analysis (BOM + dimensions)
  03_colorway.pdf         materials + colorway
  04_techdrawings.pdf     2D technical drawings

Use `python -m src.pipeline` for the full CLI (custom model id, --fast,
render size, etc.).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from src.pipeline import run


def _shoe_name(glb_path: Path) -> str:
    """Clean a .glb filename into a safe directory slug + display name."""
    stem = glb_path.stem
    # Collapse runs of underscores and strip leading/trailing.
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem


def _display_name(slug: str) -> str:
    """Make a slug like 'used_new_balance_574_classic_free' into a title."""
    return slug.replace("_", " ").title()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("glb", help="path to the input .glb file")
    p.add_argument(
        "--target-length-mm", type=float, default=270.0,
        help="shoe last length in mm (default 270 = US-9; pass 266 for US-8.5)",
    )
    args = p.parse_args(argv)

    glb = Path(args.glb).expanduser().resolve()
    if not glb.exists():
        print(f"error: file not found: {glb}", file=sys.stderr)
        return 2

    slug = _shoe_name(glb)
    out_dir = Path(__file__).parent / "output" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"input  : {glb}")
    print(f"output : {out_dir}")
    print(f"shoe   : {_display_name(slug)} (target length {args.target_length_mm} mm)")
    print()

    run(
        glb_path=str(glb),
        output_dir=str(out_dir),
        model_id=slug.upper().replace("_", "-"),
        model_name=_display_name(slug),
        designer=f"Auto-generated from {glb.name}",
        factory="TBD",
        season="Reference",
        seed=0,
        target_length_mm=args.target_length_mm,
        render_width=1100,
        render_height=800,
        samples=32,
        deterministic=True,        # byte-identical reruns
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
