"""ReportLab page assembly for 01_cover, 02_views, 03_bom, 04_colorway,
05_construction. Engineering-drawing style: large readable tables, no
marketing flourishes."""
from __future__ import annotations

import datetime as _dt
import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (
    Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table,
    TableStyle, PageBreak,
)

from .colorway import PaletteEntry
from .geometry import GeometryAnalysis
from .ingest import NormalizedScene


# Common header / footer

@dataclass
class TechPackHeader:
    model_id: str
    model_name: str
    date_iso: str
    source_file: str
    source_hash: str
    designer: str = "—"
    factory: str = "—"
    season: str = "—"


def _header_footer(c: rl_canvas.Canvas, doc, header: TechPackHeader, page_label: str) -> None:
    width, height = doc.pagesize
    margin = 15 * mm
    c.saveState()
    # Top rule
    c.setLineWidth(0.5)
    c.setStrokeColor(colors.HexColor("#222222"))
    c.line(margin, height - margin + 4, width - margin, height - margin + 4)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin, height - margin + 7, f"{header.model_id} · {header.model_name}")
    c.drawRightString(width - margin, height - margin + 7,
                      f"{page_label}   {header.date_iso}")
    # Bottom rule
    c.line(margin, margin - 4, width - margin, margin - 4)
    c.setFont("Helvetica", 7)
    c.setFillColor(colors.HexColor("#555555"))
    c.drawString(margin, margin - 12,
                 f"Source: {os.path.basename(header.source_file)}  ·  SHA256:{header.source_hash[:12]}")
    c.drawRightString(width - margin, margin - 12, f"Page {doc.page}")
    c.restoreState()


def _make_doc(path: str, header: TechPackHeader, page_label: str, pagesize=A4) -> SimpleDocTemplate:
    return SimpleDocTemplate(
        path,
        pagesize=pagesize,
        leftMargin=15 * mm, rightMargin=15 * mm,
        topMargin=22 * mm, bottomMargin=22 * mm,
        title=f"{header.model_id} · {page_label}",
        author="techpack-pipeline",
    )


# Styles

def _styles() -> dict:
    base = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle("title", base["Title"], fontName="Helvetica-Bold",
                                fontSize=22, leading=26, spaceAfter=8)
    s["subtitle"] = ParagraphStyle("subtitle", base["Title"], fontName="Helvetica",
                                   fontSize=12, leading=14, spaceAfter=12)
    s["h2"] = ParagraphStyle("h2", base["Heading2"], fontName="Helvetica-Bold",
                             fontSize=12, leading=15, spaceBefore=8, spaceAfter=4)
    s["body"] = ParagraphStyle("body", base["BodyText"], fontName="Helvetica",
                               fontSize=9.5, leading=12.5)
    s["small"] = ParagraphStyle("small", base["BodyText"], fontName="Helvetica",
                                fontSize=8, leading=10.5,
                                textColor=colors.HexColor("#555555"))
    s["mono"] = ParagraphStyle("mono", base["BodyText"], fontName="Courier",
                               fontSize=8.5, leading=11)
    return s


# View-name → human-readable label and description, used by write_views
# to caption each render cell on the multi-view PDF.
VIEW_LABEL = {
    "side-lateral":             "Lateral (Outboard)",
    "side-medial":              "Medial (Inboard)",
    "top":                      "Top (Plan)",
    "bottom":                   "Bottom (Outsole)",
    "front":                    "Front (Toe-on)",
    "back":                     "Back (Heel-on)",
    "three-quarter":            "3/4 Lateral-Front",
    "three-quarter-med":        "3/4 Medial-Front",
    "three-quarter-lat-back":   "3/4 Lateral-Back",
    "three-quarter-med-back":   "3/4 Medial-Back",
}

VIEW_DESC = {
    "side-lateral":             "Outer side of the shoe.",
    "side-medial":              "Inner side of the shoe (against the foot).",
    "top":                      "Looking down on the upper.",
    "bottom":                   "Outsole tread pattern.",
    "front":                    "Looking toward the toe box.",
    "back":                     "Looking at the heel counter.",
    "three-quarter":            "Standard product hero angle.",
    "three-quarter-med":        "Medial three-quarter perspective.",
    "three-quarter-lat-back":   "Lateral-back three-quarter perspective.",
    "three-quarter-med-back":   "Medial-back three-quarter perspective.",
}


def write_views(
    output_path: str,
    header: TechPackHeader,
    view_paths: Dict[str, str],
) -> str:
    """Multi-view rendering of the 3D model.

    Layout: landscape A4, two renders per page, side-by-side.
    Order: six standard reference views (orthographic) first, then four
    3/4 perspective views at the end so the reader meets the rigorous
    drawings before the "hero" angles.
    """
    from PIL import Image as _PILImage  # avoid hard dep at module load

    s = _styles()
    doc = _make_doc(output_path, header, "Views", pagesize=landscape(A4))
    story = []

    # Orthographic (the six standard reference views), then perspectives.
    ordered = [
        "side-lateral", "side-medial",
        "front",        "back",
        "top",          "bottom",
        # Additional 3/4 perspectives at the end:
        "three-quarter",          "three-quarter-med",
        "three-quarter-lat-back", "three-quarter-med-back",
    ]
    available = [v for v in ordered
                 if v in view_paths and os.path.exists(view_paths[v])]

    if not available:
        doc.build(story,
                  onFirstPage=lambda c, d: _header_footer(c, d, header, "VIEWS"),
                  onLaterPages=lambda c, d: _header_footer(c, d, header, "VIEWS"))
        return output_path

    # Cover header at the top of the first page only.
    story.append(Paragraph(
        "Multi-view rendering of the 3D model", s["title"]))
    story.append(Paragraph(
        "Six standard reference views (orthographic) followed by four 3/4 "
        "perspectives covering both forward and rearward corners on the "
        "inboard and outboard sides. Orthographic views are scale-true "
        "(1 mm on the model = 0.42 mm on the page); the 3/4 views are "
        "perspective references — do not scale.",
        s["small"],
    ))
    story.append(Spacer(1, 4 * mm))

    # Two renders per page, side-by-side. Each cell ≈ 128×95 mm with the
    # image aspect-preserved inside.
    CELL_W = 128 * mm
    CELL_IMG_H = 92 * mm  # image area inside the cell (label sits below)
    GAP = 8 * mm

    def _make_cell(view_key: str):
        path = view_paths[view_key]
        with _PILImage.open(path) as im:
            iw, ih = im.size
        aspect = iw / ih if ih else 1.0
        if CELL_W / aspect <= CELL_IMG_H:
            draw_w, draw_h = CELL_W, CELL_W / aspect
        else:
            draw_w, draw_h = CELL_IMG_H * aspect, CELL_IMG_H
        img = RLImage(path, width=draw_w, height=draw_h)
        img_frame = Table([[img]], colWidths=[CELL_W], rowHeights=[CELL_IMG_H])
        img_frame.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOX", (0, 0), (-1, -1), 0.3, colors.HexColor("#888888")),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        label_html = (
            f"<b>{VIEW_LABEL.get(view_key, view_key)}</b><br/>"
            f"<font size='7.5' color='#555555'>{VIEW_DESC.get(view_key, '')}</font>"
        )
        cell = Table(
            [[img_frame], [Paragraph(label_html, s["body"])]],
            colWidths=[CELL_W],
        )
        cell.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))
        return cell

    # Layout two cells per row. We may have an even or odd count; for an
    # odd tail view, the right cell stays empty.
    pairs = [available[i:i + 2] for i in range(0, len(available), 2)]
    for pi, pair in enumerate(pairs):
        if pi > 0:
            story.append(PageBreak())
            # Section header for the perspective-views run.
            if pair[0].startswith("three-quarter") and (
                    pi == 0 or not pairs[pi - 1][0].startswith("three-quarter")):
                story.append(Paragraph(
                    "Additional perspectives — 3/4 angles", s["h2"]))
                story.append(Spacer(1, 2 * mm))
        left = _make_cell(pair[0])
        right = _make_cell(pair[1]) if len(pair) == 2 else Spacer(1, 1)
        row = Table([[left, right]], colWidths=[CELL_W, CELL_W])
        row.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        # Centre the row on the page width.
        centerer = Table([[row]], colWidths=[267 * mm])
        centerer.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(centerer)

    doc.build(
        story,
        onFirstPage=lambda c, d: _header_footer(c, d, header, "VIEWS"),
        onLaterPages=lambda c, d: _header_footer(c, d, header, "VIEWS"),
    )
    return output_path


# BOM + measurements PDF

def write_bom(
    output_path: str,
    header: TechPackHeader,
    ga: GeometryAnalysis,
    scene: NormalizedScene,
    component_overlay_path: Optional[str] = None,
    spec_drawing_path: Optional[str] = None,
    lineart_path: Optional[str] = None,
    id_mask_path: Optional[str] = None,
    id_palette: Optional[List[Tuple[float, float, float]]] = None,
    materials: Optional[List] = None,
    anatomy: Optional[dict] = None,
) -> str:
    """Write 03_bom_measurements.pdf with the reference-style layout:

    * Page 1 (landscape): annotated lateral diagram — line-art shoe, red
      dots at component centroids, bold labels stacked on the LEFT.
    * Page 2 (portrait): dimensions table.
    * Page 3 (portrait): Bill of Materials table.

    Mixed-orientation pages are handled by a custom BaseDocTemplate with
    two PageTemplates (Landscape / Portrait) and NextPageTemplate
    flowables to switch between them.
    """
    from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame,
                                     NextPageTemplate)
    s = _styles()

    # Build the two page templates.
    L_W, L_H = landscape(A4)
    P_W, P_H = A4
    margin = 15 * mm
    top_margin = 22 * mm
    bottom_margin = 22 * mm

    landscape_frame = Frame(
        margin, bottom_margin,
        L_W - 2 * margin, L_H - top_margin - bottom_margin,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        id="landscape_frame",
    )
    portrait_frame = Frame(
        margin, bottom_margin,
        P_W - 2 * margin, P_H - top_margin - bottom_margin,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        id="portrait_frame",
    )

    def _make_hf(label):
        def _fn(c, d):
            _header_footer(c, d, header, label)
        return _fn

    page_landscape = PageTemplate(
        id="landscape", frames=[landscape_frame],
        pagesize=landscape(A4),
        onPage=_make_hf("BOM · COMPONENT MAP"),
    )
    page_portrait = PageTemplate(
        id="portrait", frames=[portrait_frame],
        pagesize=A4,
        onPage=_make_hf("BOM · DIMENSIONS & COMPONENTS"),
    )

    doc = BaseDocTemplate(
        output_path,
        pageTemplates=[page_landscape, page_portrait],
        title=f"{header.model_id} · BOM",
        author="techpack-pipeline",
    )

    story: list = []

    # === Page 1 (landscape): parts-anatomy infographic ===
    # Always uses the new layout when lateral+medial photo renders exist.
    # The callouts list comes from the unified mesh+ML payload (callouts
    # key) — see pipeline._build_anatomy_payload. The legacy id-mask
    # diagram is the last-resort fallback for callers that don't supply
    # a photo-render path pair.
    have_anatomy = (
        anatomy is not None
        and anatomy.get("lateral_image_path")
        and anatomy.get("medial_image_path")
        and os.path.exists(anatomy["lateral_image_path"])
        and os.path.exists(anatomy["medial_image_path"])
    )
    if have_anatomy:
        ai = _AnatomyInfographic(
            lateral_image_path=anatomy["lateral_image_path"],
            medial_image_path=anatomy["medial_image_path"],
            callouts=anatomy.get("callouts", []),
            title_main=anatomy.get("title_main",
                                    f"{header.model_name.upper()} — PARTS ANATOMY"),
            title_sub=anatomy.get("title_sub", ""),
            box_w=L_W - 2 * margin,
            box_h=L_H - top_margin - bottom_margin,
        )
        story.append(ai.as_flowable())
        story.append(NextPageTemplate("portrait"))
        story.append(PageBreak())

    have_diagram = (not have_anatomy and lineart_path and id_mask_path
                    and id_palette and os.path.exists(lineart_path)
                    and os.path.exists(id_mask_path))
    if have_diagram:
        labels = [c.name for c in ga.components]
        # Use the full landscape frame for the diagram. Title sits in the
        # page header via _make_hf; we drop the in-flow Paragraph title so
        # the flowable has all 166 mm of vertical space to itself.
        from reportlab.platypus import Flowable as _Flowable
        cb = _LateralAnnotationCallout(
            lineart_path=lineart_path,
            id_mask_path=id_mask_path,
            id_palette=id_palette,
            labels=labels,
            box_w=L_W - 2 * margin,
            box_h=L_H - top_margin - bottom_margin,
        )
        story.append(cb.as_flowable())
        story.append(NextPageTemplate("portrait"))
        story.append(PageBreak())

    # === Page 2 (portrait): dimensions ===
    story.append(Paragraph("Dimensions", s["title"]))
    story.append(Paragraph(
        "All dimensions extracted from the 3D mesh after axis canonicalisation "
        "and scale calibration. ± values are 2σ bootstrap-resample ranges.",
        s["small"],
    ))
    story.append(Spacer(1, 4 * mm))
    m = ga.measurements
    tol = getattr(m, "tolerances", {}) or {}

    def _fmt(value, key=None, unit="mm"):
        if value is None:
            return "—"
        t = tol.get(key)
        if t and t > 0:
            return f"{value:.1f} ± {t:.1f} {unit}".strip()
        return f"{value:.1f} {unit}".strip()

    def _fmt_opt(key, unit="mm"):
        v = m.extras.get(key)
        if v is None:
            return "—"
        return f"{v} {unit}".strip()

    dims = [
        ["Dimension", "Value", "Notes"],
        ["Overall length",       _fmt(m.length_mm, "length_mm"),
            "Toe-to-heel bounding box. ± = 2σ bootstrap range."],
        ["Overall width",        _fmt(m.width_mm, "width_mm"),
            "Maximum lateral-medial bbox"],
        ["Overall height",       _fmt(m.height_mm, "height_mm"),
            "Sole to collar"],
        ["Outsole thickness",    _fmt(m.sole_thickness_mm),
            "Top of outsole component above ground"],
        ["Midsole thickness",    _fmt(m.midsole_thickness_mm),
            "Between outsole top and upper start"],
        ["Heel stack height",    _fmt_opt("heel_stack_mm"),
            "Sole+midsole height at 20% of length (under heel)"],
        ["Forefoot stack height", _fmt_opt("forefoot_stack_mm"),
            "Sole+midsole height at 80% of length (under metatarsals)"],
        ["Drop (heel − forefoot)", _fmt_opt("drop_mm"),
            "Headline runner-shoe metric. ~10 mm typical for trainers."],
        ["Heel height",          _fmt(m.heel_height_mm, "heel_height_mm"),
            "Sole top at the rear 30%"],
        ["Toe spring",           _fmt(m.toe_spring_mm),
            "Lowest point of frontmost 5% of upper above ground"],
        ["Sole flex point",      _fmt_opt("sole_flex_x_pct", "%"),
            "X-position (% of length) where sole curvature is thinnest"],
        ["Insole length",        _fmt(m.insole_length_mm),
            "Footbed length at the level of the midsole top"],
        ["Forefoot girth",       _fmt(m.forefoot_girth_mm, "forefoot_girth_mm"),
            "Alpha-shape perimeter of YZ section at x = +0.65 L"],
        ["Instep girth",         _fmt(m.instep_girth_mm, "instep_girth_mm"),
            "Alpha-shape perimeter of YZ section at midfoot (concave-aware)"],
        ["Throat opening",       _fmt_opt("throat_opening_mm"),
            "Lacing throat width — clearance between lateral & medial uppers"],
        ["Ankle opening (L × W)",
            (f"{m.extras.get('ankle_opening_l_mm','?')} × "
             f"{m.extras.get('ankle_opening_w_mm','?')} mm"),
            "Collar opening dimensions at the topmost slice"],
        ["Size: CM (shoe length)", f"{m.extras.get('size_cm_shoe', '?')} cm",
            "Last length / 10 — matches NB/Adidas chart 'CM' column"],
        ["Size: US men's", f"{m.extras.get('size_us_men', '?')}",
            "1 US = 10 mm of last length"],
        ["Size: UK men's", f"{m.extras.get('size_uk_men', '?')}", "US men's − 0.5"],
        ["Size: EU", f"{m.extras.get('size_eu', '?')}", "US men's + 33.5"],
        ["Foot length (est.)", f"{m.extras.get('foot_length_mm', '?')} mm",
            "Last length − 12 mm toe room"],
    ]
    t = Table(dims, colWidths=[55 * mm, 35 * mm, 90 * mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#222222")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ALIGN", (1, 1), (1, -1), "RIGHT"),
    ]))
    story.append(t)

    # === Page 3 (portrait): BOM table ===
    story.append(PageBreak())
    story.append(Paragraph("Components (Bill of Materials)", s["title"]))
    story.append(Paragraph(
        "Canonical-vocabulary component list. Confidence ≥ 0.5 means the "
        "cluster is a single, real component (high face count, low colour "
        "variance, sensible area). Rows below 0.5 are flagged for human "
        "review. Perim = cutting-line length used directly for pattern grading.",
        s["small"],
    ))
    story.append(Spacer(1, 4 * mm))
    rows = [["#", "Component", "Material", "Area (mm²)",
             "Perim (mm)", "Color", "Conf", "Notes"]]
    total_area = sum(c.area_mm2 for c in ga.components) or 1.0
    for i, c in enumerate(ga.components, start=1):
        area_pct = 100.0 * c.area_mm2 / total_area
        rgb = c.dominant_color_rgb
        swatch = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
        conf = getattr(c, "confidence", 1.0)
        perim = getattr(c, "perimeter_mm", 0.0)
        conf_str = f"{conf:.2f}"
        if conf < 0.5:
            conf_str += " ⚠"
        note = c.note or "—"
        if conf < 0.5:
            note = ("REVIEW: " + note) if note != "—" else "REVIEW — low confidence"
        rows.append([
            str(i), c.name, c.inferred_material,
            f"{c.area_mm2:,.0f}  ({area_pct:.0f}%)",
            f"{perim:,.0f}",
            swatch, conf_str, note,
        ])
    bt = Table(rows, colWidths=[7 * mm, 30 * mm, 36 * mm, 26 * mm,
                                18 * mm, 18 * mm, 14 * mm, 31 * mm])
    style_cmds = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.0),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#222222")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("ALIGN", (3, 1), (4, -1), "RIGHT"),
        ("ALIGN", (6, 1), (6, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    # Paint the swatch cell with the actual dominant color.
    for ri, c in enumerate(ga.components, start=1):
        rgb = c.dominant_color_rgb
        hexcol = colors.Color(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
        style_cmds.append(("BACKGROUND", (5, ri), (5, ri), hexcol))
        lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        fg = colors.white if lum < 130 else colors.black
        style_cmds.append(("TEXTCOLOR", (5, ri), (5, ri), fg))
        # Highlight low-confidence rows with a faint warning tint.
        if getattr(c, "confidence", 1.0) < 0.5:
            style_cmds.append(
                ("BACKGROUND", (0, ri), (4, ri), colors.HexColor("#FFF4E0")))
            style_cmds.append(
                ("BACKGROUND", (6, ri), (-1, ri), colors.HexColor("#FFF4E0")))
    bt.setStyle(TableStyle(style_cmds))
    story.append(bt)

    doc.build(story)
    return output_path


# Colorway callout (lateral shoe with arrows to swatch labels)

class _CalloutDiagram:
    """Base flowable: overlay image with arrows pointing from per-component
    label boxes (left/right columns) to each component's centroid on the
    image. Subclasses override `_draw_label_content` to control what
    appears inside each label box. Defaults (label sizes, padding) target
    the colorway callout; subclasses may override.
    """
    label_h_mm: float = 14.0
    label_col_w_mm: float = 58.0
    swatch_w_mm: float = 8.0

    def __init__(self, image_path: str, id_mask_path: str,
                 id_palette: List[Tuple[float, float, float]],
                 components: list,
                 box_w: float, box_h: float):
        from reportlab.platypus import Flowable
        self._Flowable = Flowable
        self.image_path = image_path
        self.id_mask_path = id_mask_path
        self.id_palette = id_palette
        self.components = components  # subclass-dependent shape
        self.box_w = box_w
        self.box_h = box_h


class _ColorwayCallout(_CalloutDiagram):
    """Lateral overlay with arrows from colorway swatches (one per component).
    `components` is a list of (name, PaletteEntry) tuples."""

    def _render_label(self, c, idx, lx, ly, w, h, swatch_w):
        name, pe = self.components[idx]
        r, g, b = pe.rgb[0] / 255, pe.rgb[1] / 255, pe.rgb[2] / 255
        c.setFillColor(colors.white)
        c.setStrokeColor(colors.HexColor("#888888"))
        c.setLineWidth(0.4)
        c.rect(lx, ly, w, h, fill=1, stroke=1)
        c.setFillColorRGB(r, g, b)
        c.setStrokeColor(colors.HexColor("#555555"))
        c.setLineWidth(0.3)
        sw_x = lx + 2 * mm
        sw_y = ly + (h - swatch_w) / 2
        c.rect(sw_x, sw_y, swatch_w, swatch_w, fill=1, stroke=1)
        tx = sw_x + swatch_w + 2 * mm
        c.setFillColor(colors.HexColor("#101010"))
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(tx, ly + h - 5 * mm, name)
        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.HexColor("#444444"))
        c.drawString(tx, ly + h - 9 * mm,
                     f"{pe.hex}  ·  {pe.pantone}")

    def as_flowable(self):
        outer = self
        Flowable = self._Flowable

        class _F(Flowable):
            def wrap(self, aw, ah):
                return outer.box_w, outer.box_h

            def draw(self):
                outer._draw(self.canv)

        return _F()

    def _draw(self, c):
        from PIL import Image as _PIL
        import numpy as _np

        with _PIL.open(self.image_path) as im:
            iw, ih = im.size
        with _PIL.open(self.id_mask_path) as im:
            mask_arr = _np.array(im.convert("RGB"))

        # Compute pixel centroid for each component by exact-color match on
        # the id-mask render. Returns (cx_frac, cy_frac) in [0,1] of the
        # mask image, or None if the component is not visible from this angle.
        centroids: List[Optional[Tuple[float, float]]] = []
        h, w = mask_arr.shape[:2]
        for i in range(len(self.components)):
            if i >= len(self.id_palette):
                centroids.append(None)
                continue
            rgb01 = self.id_palette[i]
            r, g, b = (int(round(rgb01[0] * 255)),
                       int(round(rgb01[1] * 255)),
                       int(round(rgb01[2] * 255)))
            # Render PNG round-trip may shift by ±1; match within tolerance.
            d = _np.abs(mask_arr.astype(_np.int16) - _np.array([r, g, b], dtype=_np.int16))
            m = (d.max(axis=2) <= 2)
            cnt = int(m.sum())
            if cnt < 50:
                centroids.append(None)
                continue
            ys, xs = _np.where(m)
            cx = float(xs.mean()) / w
            cy = float(ys.mean()) / h
            centroids.append((cx, cy))

        # Image rect: fit image preserving aspect inside an inner box centred
        # in our flowable, with margins reserved for label columns.
        LABEL_COL_W = self.label_col_w_mm * mm
        GAP = 6 * mm
        img_box_w = self.box_w - 2 * (LABEL_COL_W + GAP)
        img_box_h = self.box_h
        aspect = iw / ih if ih else 1.0
        if img_box_w / aspect <= img_box_h:
            draw_w, draw_h = img_box_w, img_box_w / aspect
        else:
            draw_w, draw_h = img_box_h * aspect, img_box_h
        img_x = LABEL_COL_W + GAP + (img_box_w - draw_w) / 2
        img_y = (self.box_h - draw_h) / 2

        c.drawImage(self.image_path, img_x, img_y, draw_w, draw_h,
                    preserveAspectRatio=False, anchor="sw", mask="auto")

        # Partition components into left/right columns by centroid x.
        left_items: List[Tuple[int, float]] = []   # (idx, centroid_y_in_page)
        right_items: List[Tuple[int, float]] = []
        anchors: List[Optional[Tuple[float, float]]] = []
        for i, cent in enumerate(centroids):
            if cent is None:
                anchors.append(None)
                continue
            cx_frac, cy_frac = cent
            ax = img_x + cx_frac * draw_w
            ay = img_y + (1.0 - cy_frac) * draw_h  # flip y: image origin top-left → page bottom-left
            anchors.append((ax, ay))
            if cx_frac < 0.5:
                left_items.append((i, ay))
            else:
                right_items.append((i, ay))

        # Sort each column top-to-bottom (high y to low y on page).
        left_items.sort(key=lambda t: -t[1])
        right_items.sort(key=lambda t: -t[1])

        # Label box dimensions; stack with consistent spacing.
        LABEL_H = self.label_h_mm * mm
        SWATCH_W = self.swatch_w_mm * mm

        def _layout_column(items, col_x_left, col_x_right):
            """Distribute labels evenly within the flowable height; return
            list of (idx, lx, ly_bottom)."""
            n = len(items)
            if n == 0:
                return []
            total_h = n * LABEL_H + (n - 1) * 4 * mm
            if total_h > self.box_h:
                # Compress spacing if there are many labels.
                spacing = max(0.0, (self.box_h - n * LABEL_H) / max(n - 1, 1))
            else:
                spacing = 4 * mm
            content_h = n * LABEL_H + (n - 1) * spacing
            y0 = (self.box_h + content_h) / 2  # top of first label
            placements = []
            for k, (idx, _ay) in enumerate(items):
                ly_top = y0 - k * (LABEL_H + spacing)
                ly_bottom = ly_top - LABEL_H
                placements.append((idx, col_x_left, ly_bottom))
            return placements

        left_placements = _layout_column(left_items, 0, LABEL_COL_W)
        right_placements = _layout_column(right_items,
                                          self.box_w - LABEL_COL_W, self.box_w)

        def _draw_label(idx: int, lx: float, ly: float, side: str):
            self._render_label(c, idx, lx, ly, LABEL_COL_W, LABEL_H, SWATCH_W)

        def _draw_arrow(p1, p2):
            c.setStrokeColor(colors.HexColor("#202020"))
            c.setLineWidth(0.6)
            c.line(p1[0], p1[1], p2[0], p2[1])
            # Arrowhead at p2
            import math as _math
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            L = (dx * dx + dy * dy) ** 0.5 or 1.0
            ux, uy = dx / L, dy / L
            ah = 2.0 * mm
            aw = 1.2 * mm
            bx, by = p2[0] - ux * ah, p2[1] - uy * ah
            # Perpendicular
            px, py = -uy, ux
            c.setFillColor(colors.HexColor("#202020"))
            from reportlab.graphics.shapes import Polygon  # noqa: F401  (just to ensure import)
            p = c.beginPath()
            p.moveTo(p2[0], p2[1])
            p.lineTo(bx + px * aw, by + py * aw)
            p.lineTo(bx - px * aw, by - py * aw)
            p.close()
            c.drawPath(p, stroke=0, fill=1)

        # Draw arrows first (so labels sit on top), then label boxes.
        for placements, side in ((left_placements, "left"), (right_placements, "right")):
            for idx, lx, ly in placements:
                anc = anchors[idx]
                if anc is None:
                    continue
                if side == "left":
                    p1 = (lx + LABEL_COL_W, ly + LABEL_H / 2)
                else:
                    p1 = (lx, ly + LABEL_H / 2)
                _draw_arrow(p1, anc)
        for placements, side in ((left_placements, "left"), (right_placements, "right")):
            for idx, lx, ly in placements:
                _draw_label(idx, lx, ly, side)


def _compute_centroids_from_id_mask(
    id_mask_path: str,
    id_palette: list,
    n_components: int,
    bg_min_value: int = 220,
    min_component_pixels: int = 50,
    max_color_distance: int = 30,
) -> list:
    """For each component index, find its pixel centroid on the id-mask
    PNG. Returns a list of (cx_frac, cy_frac) in [0, 1] of the mask image
    (origin top-left), or None where the component is not visible.

    Pixels are assigned to the nearest palette colour by L2 distance.
    `max_color_distance` (L2, sum of squared channel diffs ≤ 30² = 900)
    rejects pixels that aren't actually close to any palette colour —
    important when the id-mask is rendered with anti-aliasing at
    component boundaries (the AA pixels are halfway between two colours
    and would otherwise pull the centroid toward the seam).

    Background pixels (all RGB > bg_min_value) are excluded before
    assignment so the background doesn't 'win' the nearest-colour vote.
    """
    from PIL import Image as _PIL
    import numpy as _np
    with _PIL.open(id_mask_path) as im:
        mask_arr = _np.array(im.convert("RGB"))
    h, w = mask_arr.shape[:2]
    if n_components == 0:
        return []

    # Apply the sRGB OETF (linear → display-encoded) to the input palette
    # so it matches the renderer's output. The Blender renderer takes
    # linear floats in [0, 1] and writes sRGB-encoded uint8 to the PNG;
    # without this step the input (linear) palette is darker than the
    # rendered pixels and no match is found.
    def _linear_to_srgb_u8(x: float) -> int:
        x = max(0.0, min(1.0, x))
        if x <= 0.0031308:
            y = 12.92 * x
        else:
            y = 1.055 * (x ** (1.0 / 2.4)) - 0.055
        return int(round(y * 255))

    palette = _np.array(
        [(_linear_to_srgb_u8(id_palette[i][0]),
          _linear_to_srgb_u8(id_palette[i][1]),
          _linear_to_srgb_u8(id_palette[i][2]))
         if i < len(id_palette) else (255, 255, 255)
         for i in range(n_components)],
        dtype=_np.int32,
    )
    flat = mask_arr.reshape(-1, 3).astype(_np.int32)
    bg = (flat.min(axis=1) >= bg_min_value)
    diff = flat[:, None, :] - palette[None, :, :]
    dist2 = (diff * diff).sum(axis=2)
    assign = dist2.argmin(axis=1)
    min_d2 = dist2.min(axis=1)
    too_far = (min_d2 > max_color_distance * max_color_distance)
    assign[bg | too_far] = -1
    assign = assign.reshape(h, w)

    # Largest-connected-component centroid. Arithmetic mean over ALL
    # pixels of a component lands BETWEEN clusters when the component
    # has disconnected pixel regions (e.g. tongue mesh visible through
    # the throat opening + a small patch above), placing the marker in
    # empty space. Picking the centroid of just the LARGEST connected
    # region keeps the marker on the visible part. 8-connectivity so
    # AA-bleed pixels don't fragment a real component.
    try:
        from scipy.ndimage import label as _label
        _structure = _np.ones((3, 3), dtype=bool)
        _have_scipy = True
    except Exception:
        _have_scipy = False

    out: list = []
    for i in range(n_components):
        comp_mask = (assign == i)
        if comp_mask.sum() < min_component_pixels:
            out.append(None)
            continue
        if _have_scipy:
            labelled, n_lab = _label(comp_mask, structure=_structure)
            if n_lab == 0:
                out.append(None)
                continue
            sizes = _np.bincount(labelled.ravel())
            # sizes[0] is background; pick the largest non-background label.
            largest = int(sizes[1:].argmax()) + 1 if n_lab >= 1 else 0
            ys, xs = _np.where(labelled == largest)
            if len(xs) < min_component_pixels:
                # Largest region too small; fall back to mean of all.
                ys, xs = _np.where(comp_mask)
        else:
            ys, xs = _np.where(comp_mask)
        out.append((float(xs.mean()) / w, float(ys.mean()) / h))
    return out


class _LateralAnnotationCallout:
    """Reference-style annotated BOM diagram: line-art lateral shoe on the
    right ~70 % of the flowable, with red dots at each component's pixel
    centroid and bold black labels stacked on the LEFT, connected by thin
    red leader lines. Matches the industry-standard designer-sketch
    aesthetic shown in the user's reference image.

    Parameters:
      lineart_path: path to a black-on-white Freestyle line-art PNG of
                    the lateral view (e.g. linework_side-lateral.png).
      id_mask_path: distinct-palette flat-colour render used to locate
                    each component's pixel centroid.
      id_palette: list of (r, g, b) floats in [0, 1]; index i matches
                  face_component value i.
      labels: list of label strings parallel to id_palette.
      box_w, box_h: flowable size in page units (points).
    """

    DOT_COLOR = colors.HexColor("#D9342E")
    LEADER_COLOR = colors.HexColor("#D9342E")
    LABEL_COLOR = colors.HexColor("#101010")

    def __init__(self, lineart_path: str, id_mask_path: str,
                 id_palette: list, labels: list,
                 box_w: float, box_h: float):
        from reportlab.platypus import Flowable
        self._Flowable = Flowable
        self.lineart_path = lineart_path
        self.id_mask_path = id_mask_path
        self.id_palette = id_palette
        self.labels = labels
        self.box_w = box_w
        self.box_h = box_h

    def as_flowable(self):
        outer = self
        Flowable = self._Flowable

        class _F(Flowable):
            def wrap(self, aw, ah):
                return outer.box_w, outer.box_h

            def draw(self):
                outer._draw(self.canv)

        return _F()

    def _draw(self, c):
        from PIL import Image as _PIL
        with _PIL.open(self.lineart_path) as im:
            iw, ih = im.size
        aspect = iw / ih if ih else 1.0

        centroids = _compute_centroids_from_id_mask(
            self.id_mask_path, self.id_palette, len(self.labels),
        )

        # Layout: image centred, label columns on BOTH sides.
        LABEL_COL_W = 42 * mm
        GAP = 6 * mm
        img_box_w = self.box_w - 2 * (LABEL_COL_W + GAP)
        img_box_h = self.box_h
        if img_box_w / aspect <= img_box_h:
            draw_w, draw_h = img_box_w, img_box_w / aspect
        else:
            draw_w, draw_h = img_box_h * aspect, img_box_h
        img_x = (self.box_w - draw_w) / 2
        img_y = (self.box_h - draw_h) / 2

        c.drawImage(self.lineart_path, img_x, img_y, draw_w, draw_h,
                    preserveAspectRatio=False, anchor="sw", mask="auto")

        # Resolve each visible component's anchor on the page.
        anchors = []
        for i, cent in enumerate(centroids):
            if cent is None or not self.labels[i]:
                anchors.append(None)
                continue
            cx_frac, cy_frac = cent
            ax = img_x + cx_frac * draw_w
            ay = img_y + (1.0 - cy_frac) * draw_h
            anchors.append((ax, ay))

        # Partition components into left/right columns by centroid x.
        # Components on the left half of the image → left column;
        # right half → right column. Then sort each column top-to-bottom.
        left_items: list = []
        right_items: list = []
        img_centre_x = img_x + draw_w / 2
        for i, anc in enumerate(anchors):
            if anc is None:
                continue
            if anc[0] < img_centre_x:
                left_items.append((i, anc))
            else:
                right_items.append((i, anc))
        left_items.sort(key=lambda t: -t[1][1])
        right_items.sort(key=lambda t: -t[1][1])

        def _slot_ys(n):
            top_y = self.box_h - 4 * mm
            bottom_y = 4 * mm
            if n == 0:
                return []
            if n == 1:
                return [(top_y + bottom_y) / 2]
            step = (top_y - bottom_y) / (n - 1)
            return [top_y - i * step for i in range(n)]

        c.setFont("Helvetica-Bold", 9)

        # LEFT column: labels right-aligned at left_x_right; leader extends
        # rightward to the dot.
        left_x_right = LABEL_COL_W - 4 * mm
        for slot_y, (i, anc) in zip(_slot_ys(len(left_items)), left_items):
            label = self.labels[i]
            c.setFillColor(self.LABEL_COLOR)
            c.drawRightString(left_x_right, slot_y - 2.4, label)
            c.setStrokeColor(self.LEADER_COLOR)
            c.setLineWidth(0.45)
            c.line(left_x_right + 1.5 * mm, slot_y - 0.5, anc[0], anc[1])
            c.setFillColor(self.DOT_COLOR)
            c.setStrokeColor(self.DOT_COLOR)
            c.circle(anc[0], anc[1], 1.6 * mm, stroke=0, fill=1)

        # RIGHT column: labels left-aligned at right_x_left.
        right_x_left = self.box_w - LABEL_COL_W + 4 * mm
        for slot_y, (i, anc) in zip(_slot_ys(len(right_items)), right_items):
            label = self.labels[i]
            c.setFillColor(self.LABEL_COLOR)
            c.drawString(right_x_left, slot_y - 2.4, label)
            c.setStrokeColor(self.LEADER_COLOR)
            c.setLineWidth(0.45)
            c.line(right_x_left - 1.5 * mm, slot_y - 0.5, anc[0], anc[1])
            c.setFillColor(self.DOT_COLOR)
            c.setStrokeColor(self.DOT_COLOR)
            c.circle(anc[0], anc[1], 1.6 * mm, stroke=0, fill=1)


@dataclass
class Callout:
    """Unified callout produced by both the mesh-derived path and the ML
    enricher. Centroid fractions are in [0, 1] image coords with origin
    top-left (None if the part isn't visible in that view).

    Used by _AnatomyInfographic (description below name), by
    _FabricAnatomyInfographic (material below name), and by
    _ColorAnatomyInfographic (hex_color shown as swatch + code).
    """
    label: str                        # canonical id, e.g. "vamp", "n-logo"
    display_name: str                 # "Toe Box (Vamp)"
    description: str = ""             # "Front upper that covers the toes."
    material: Optional[str] = None    # "Suede", "EVA Foam", ...
    hex_color: Optional[str] = None   # "#4F6D9A"
    lateral_centroid: Optional[Tuple[float, float]] = None  # (cx, cy) in [0,1]
    medial_centroid: Optional[Tuple[float, float]] = None


def _build_callouts_from_mesh(
    ga: GeometryAnalysis,
    lateral_id_mask_path: Optional[str],
    medial_id_mask_path: Optional[str],
    id_palette: List[Tuple[float, float, float]],
    materials: Optional[List] = None,
    display_name_overrides: Optional[Dict[str, str]] = None,
    description_overrides: Optional[Dict[str, str]] = None,
    component_palettes: Optional[List[List[PaletteEntry]]] = None,
) -> List[Callout]:
    """Build a list of Callouts directly from the existing mesh segmentation.

    Centroid fractions for each view are looked up via the existing
    `_compute_centroids_from_id_mask` helper on each view's id-mask render.
    materials[i].label feeds the Callout.material slot;
    component.dominant_color_rgb feeds the hex_color slot.

    display_name_overrides / description_overrides let the caller plug in
    the reference-image friendly names ("Toe Box (Vamp)" instead of "vamp"),
    keyed by canonical component name.
    """
    n = len(ga.components)
    lateral_cents = (_compute_centroids_from_id_mask(
        lateral_id_mask_path, id_palette, n)
        if lateral_id_mask_path else [None] * n)
    medial_cents = (_compute_centroids_from_id_mask(
        medial_id_mask_path, id_palette, n)
        if medial_id_mask_path else [None] * n)

    display_name_overrides = display_name_overrides or {}
    description_overrides = description_overrides or {}

    # Prefer the per-component MOST-SATURATED swatch (from the OKLab
    # mini-palette) over the area-weighted dominant color. The dominant
    # color averages e.g. a blue-mesh + gray-suede component to gray
    # and erases the vibrant accent the user actually wants to see.
    try:
        from .colorway import rgb_to_hsv_s
    except Exception:
        rgb_to_hsv_s = None

    def _vibrant_hex(idx: int, fallback_rgb) -> Optional[str]:
        if (component_palettes is not None and idx < len(component_palettes)
                and rgb_to_hsv_s is not None):
            entries = component_palettes[idx] or []
            # Drop tiny noise clusters and pick the most-saturated remaining.
            candidates = [e for e in entries if getattr(e, "fraction", 0.0) >= 0.05]
            if not candidates and entries:
                candidates = list(entries)
            if candidates:
                best = max(candidates, key=lambda e: rgb_to_hsv_s(e.rgb))
                return best.hex
        if fallback_rgb is not None:
            return (f"#{int(fallback_rgb[0]):02X}"
                    f"{int(fallback_rgb[1]):02X}"
                    f"{int(fallback_rgb[2]):02X}")
        return None

    out: List[Callout] = []
    for i, comp in enumerate(ga.components):
        name = comp.name
        display = display_name_overrides.get(name, name)
        desc = description_overrides.get(name, "")
        mat = None
        if materials is not None and i < len(materials) and materials[i] is not None:
            mat = getattr(materials[i], "label", None) or getattr(
                materials[i], "material_class", None)
        rgb = getattr(comp, "dominant_color_rgb", None)
        hex_color = _vibrant_hex(i, rgb)
        out.append(Callout(
            label=name,
            display_name=display,
            description=desc,
            material=mat,
            hex_color=hex_color,
            lateral_centroid=lateral_cents[i] if i < len(lateral_cents) else None,
            medial_centroid=medial_cents[i] if i < len(medial_cents) else None,
        ))
    return out


# ---- shared rendering primitives for the three infographic flowables ----

_NB_BLUE = colors.HexColor("#1F4FA8")
_TEXT_DARK = colors.HexColor("#101010")
_TEXT_MID = colors.HexColor("#404040")


def _fit_image_box(image_path, img_x, img_y, img_w, img_h):
    """Compute the (dx, dy, draw_w, draw_h) for fit-inside placement of an
    image into the given box; returns also (iw, ih) source pixel size."""
    from PIL import Image as _PIL
    with _PIL.open(image_path) as im:
        iw, ih = im.size
    aspect = iw / ih if ih else 1.0
    if img_w / aspect <= img_h:
        draw_w, draw_h = img_w, img_w / aspect
    else:
        draw_w, draw_h = img_h * aspect, img_h
    dx = img_x + (img_w - draw_w) / 2
    dy = img_y + (img_h - draw_h) / 2
    return dx, dy, draw_w, draw_h, iw, ih


def _draw_marker_circle(c, ax, ay, marker_text, radius=3.6 * mm,
                         font_size=9, fill_color=colors.white,
                         stroke_color=_NB_BLUE, text_color=_NB_BLUE,
                         stroke_width=1.0):
    c.setStrokeColor(stroke_color)
    c.setLineWidth(stroke_width)
    c.setFillColor(fill_color)
    c.circle(ax, ay, radius, stroke=1, fill=1)
    c.setFont("Helvetica-Bold", font_size)
    c.setFillColor(text_color)
    c.drawCentredString(ax, ay - font_size * 0.32, str(marker_text))


def _draw_legend_two_line(c, marker, name, second_line,
                           lx, ly, col_w, row_h,
                           marker_radius=2.8 * mm,
                           marker_font=8, name_font=8.5, second_font=7.5,
                           marker_color=_NB_BLUE,
                           name_color=_TEXT_DARK,
                           second_color=_TEXT_MID,
                           color_swatch_hex=None):
    """One legend row: marker circle + bold name + small second line.
    If `color_swatch_hex` is given (e.g. "#4F6D9A"), draw a swatch square
    between the marker and the name (used by the color-anatomy legend)."""
    cx = lx + marker_radius + 1 * mm
    cy = ly + row_h / 2
    _draw_marker_circle(c, cx, cy, marker,
                        radius=marker_radius, font_size=marker_font,
                        stroke_color=marker_color, text_color=marker_color)
    text_x = cx + marker_radius + 2 * mm
    if color_swatch_hex:
        sw = 5 * mm
        c.setStrokeColor(colors.HexColor("#888888"))
        c.setLineWidth(0.4)
        try:
            c.setFillColor(colors.HexColor(color_swatch_hex))
        except Exception:
            c.setFillColor(colors.white)
        c.rect(text_x, cy - sw / 2, sw, sw, stroke=1, fill=1)
        text_x += sw + 1.5 * mm

    c.setFont("Helvetica-Bold", name_font)
    c.setFillColor(name_color)
    c.drawString(text_x, cy + 0.5 * mm, name)
    if second_line:
        c.setFont("Helvetica", second_font)
        c.setFillColor(second_color)
        max_w = (lx + col_w) - text_x - 2 * mm
        s = second_line
        while s and c.stringWidth(s, "Helvetica", second_font) > max_w:
            s = s[:-1]
        if s != second_line and s:
            s = s.rstrip() + "…"
        c.drawString(text_x, cy - 3 * mm, s)


class _AnatomyInfographic:
    """Parts-anatomy BOM page: side-by-side lateral + medial photos with
    numbered circle callouts on each, and a multi-column legend below
    mapping each number to display name + description.

    Operates on a unified `callouts: List[Callout]`. A callout appears
    on a given side if its corresponding `*_centroid` is not None. Same
    callout on both views shares the same number.
    """

    def __init__(self, lateral_image_path, medial_image_path,
                 callouts: List[Callout],
                 title_main: str, title_sub: str,
                 box_w: float, box_h: float):
        from reportlab.platypus import Flowable
        self._Flowable = Flowable
        self.lateral_image_path = lateral_image_path
        self.medial_image_path = medial_image_path
        self.callouts = list(callouts)
        self.title_main = title_main
        self.title_sub = title_sub
        self.box_w = box_w
        self.box_h = box_h

    def as_flowable(self):
        outer = self
        Flowable = self._Flowable

        class _F(Flowable):
            def wrap(self, aw, ah):
                return outer.box_w, outer.box_h

            def draw(self):
                outer._draw(self.canv)

        return _F()

    def _draw_side(self, c, image_path, cent_attr, side_label,
                    img_x, img_y, img_w, img_h, marker_for):
        dx, dy, draw_w, draw_h, _iw, _ih = _fit_image_box(
            image_path, img_x, img_y, img_w, img_h,
        )
        c.drawImage(image_path, dx, dy, draw_w, draw_h,
                    preserveAspectRatio=False, anchor="sw", mask="auto")
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(_NB_BLUE)
        c.drawCentredString(dx + draw_w / 2, dy - 4 * mm, side_label)

        for co in self.callouts:
            cent = getattr(co, cent_attr)
            if cent is None:
                continue
            marker = marker_for(co)
            if marker is None:
                continue
            fx, fy = cent
            ax = dx + fx * draw_w
            ay = dy + (1.0 - fy) * draw_h
            _draw_marker_circle(c, ax, ay, marker)

    def _draw(self, c):
        visible = [co for co in self.callouts
                   if co.lateral_centroid is not None or co.medial_centroid is not None]
        order = list(enumerate(visible, start=1))
        label_to_num = {co.label: i for i, co in order}

        def marker_for(co):
            return label_to_num.get(co.label)

        # Title.
        title_h = 12 * mm
        c.setFont("Helvetica-Bold", 16)
        c.setFillColor(_TEXT_DARK)
        c.drawCentredString(self.box_w / 2, self.box_h - 7 * mm, self.title_main)
        c.setFont("Helvetica", 10)
        c.setFillColor(_NB_BLUE)
        c.drawCentredString(self.box_w / 2, self.box_h - 11.5 * mm, self.title_sub)

        below_title = self.box_h - title_h
        img_band_h = below_title * 0.58
        legend_band_h = below_title * 0.42

        gap = 6 * mm
        img_w = (self.box_w - gap) / 2
        img_h = img_band_h - 8 * mm
        img_y = legend_band_h + 8 * mm
        self._draw_side(c, self.lateral_image_path, "lateral_centroid",
                         "LATERAL SIDE (OUTBOARD)",
                         0, img_y, img_w, img_h, marker_for)
        self._draw_side(c, self.medial_image_path, "medial_centroid",
                         "MEDIAL SIDE (INBOARD)",
                         img_w + gap, img_y, img_w, img_h, marker_for)

        n = len(visible)
        if n == 0:
            return
        n_cols = 4 if n > 9 else (3 if n > 6 else 2)
        rows_per_col = (n + n_cols - 1) // n_cols
        col_w = self.box_w / n_cols
        row_h = (legend_band_h - 2 * mm) / max(rows_per_col, 1)
        row_h = min(row_h, 9 * mm)

        for i, co in order:
            col = (i - 1) // rows_per_col
            row = (i - 1) % rows_per_col
            lx = col * col_w
            ly = (legend_band_h - 2 * mm) - (row + 1) * row_h
            _draw_legend_two_line(
                c, marker=i, name=co.display_name,
                second_line=co.description, lx=lx, ly=ly,
                col_w=col_w, row_h=row_h,
            )


class _FabricAnatomyInfographic(_AnatomyInfographic):
    """Same layout as _AnatomyInfographic but the legend row's second
    line shows the inferred material (e.g. "Suede", "EVA Foam") instead
    of the part description."""

    def _draw(self, c):
        visible = [co for co in self.callouts
                   if co.lateral_centroid is not None or co.medial_centroid is not None]
        order = list(enumerate(visible, start=1))
        label_to_num = {co.label: i for i, co in order}

        def marker_for(co):
            return label_to_num.get(co.label)

        title_h = 12 * mm
        c.setFont("Helvetica-Bold", 16)
        c.setFillColor(_TEXT_DARK)
        c.drawCentredString(self.box_w / 2, self.box_h - 7 * mm, self.title_main)
        c.setFont("Helvetica", 10)
        c.setFillColor(_NB_BLUE)
        c.drawCentredString(self.box_w / 2, self.box_h - 11.5 * mm, self.title_sub)

        below_title = self.box_h - title_h
        img_band_h = below_title * 0.58
        legend_band_h = below_title * 0.42

        gap = 6 * mm
        img_w = (self.box_w - gap) / 2
        img_h = img_band_h - 8 * mm
        img_y = legend_band_h + 8 * mm
        self._draw_side(c, self.lateral_image_path, "lateral_centroid",
                         "LATERAL SIDE (OUTBOARD)",
                         0, img_y, img_w, img_h, marker_for)
        self._draw_side(c, self.medial_image_path, "medial_centroid",
                         "MEDIAL SIDE (INBOARD)",
                         img_w + gap, img_y, img_w, img_h, marker_for)

        n = len(visible)
        if n == 0:
            return
        n_cols = 4 if n > 9 else (3 if n > 6 else 2)
        rows_per_col = (n + n_cols - 1) // n_cols
        col_w = self.box_w / n_cols
        row_h = (legend_band_h - 2 * mm) / max(rows_per_col, 1)
        row_h = min(row_h, 9 * mm)

        for i, co in order:
            col = (i - 1) // rows_per_col
            row = (i - 1) % rows_per_col
            lx = col * col_w
            ly = (legend_band_h - 2 * mm) - (row + 1) * row_h
            second = co.material or co.description
            name = (f"{co.display_name} ({co.material})"
                    if co.material else co.display_name)
            _draw_legend_two_line(
                c, marker=i, name=name, second_line=co.description,
                lx=lx, ly=ly, col_w=col_w, row_h=row_h,
            )


class _ColorAnatomyInfographic:
    """Color-anatomy page: single lateral photo on the left with letter
    callouts (A, B, C, ...) at component centroids; a legend on the right
    mapping each letter to part name + color swatch + hex code; and a
    "Dominant Colors Palette" strip below the legend.
    """

    def __init__(self, lateral_image_path: str, callouts: List[Callout],
                 dominant_palette: List, title_main: str, title_sub: str,
                 summary: str, box_w: float, box_h: float,
                 vibrant_palette: Optional[List] = None):
        from reportlab.platypus import Flowable
        self._Flowable = Flowable
        self.lateral_image_path = lateral_image_path
        # Keep only callouts that have a centroid on lateral AND a hex_color.
        self.callouts = [co for co in callouts
                          if co.lateral_centroid is not None and co.hex_color]
        # Vibrant palette (most-saturated per-component swatches) is the
        # headline strip when available; the area-weighted `dominant_palette`
        # remains the fallback for legacy callers.
        self.dominant_palette = list(vibrant_palette or dominant_palette or [])
        self.title_main = title_main
        self.title_sub = title_sub
        self.summary = summary
        self.box_w = box_w
        self.box_h = box_h

    def as_flowable(self):
        outer = self
        Flowable = self._Flowable

        class _F(Flowable):
            def wrap(self, aw, ah):
                return outer.box_w, outer.box_h

            def draw(self):
                outer._draw(self.canv)

        return _F()

    def _draw(self, c):
        # Title.
        c.setFont("Helvetica-Bold", 16)
        c.setFillColor(_TEXT_DARK)
        c.drawCentredString(self.box_w / 2, self.box_h - 7 * mm, self.title_main)
        c.setFont("Helvetica", 10)
        c.setFillColor(_NB_BLUE)
        c.drawCentredString(self.box_w / 2, self.box_h - 11.5 * mm, self.title_sub)

        below_title = self.box_h - 14 * mm

        # Left half: photo. Right half: letter legend + palette strip.
        photo_w = self.box_w * 0.45
        photo_h = below_title * 0.78
        photo_x = 0
        photo_y = self.box_h - 14 * mm - photo_h

        dx, dy, draw_w, draw_h, _iw, _ih = _fit_image_box(
            self.lateral_image_path, photo_x, photo_y, photo_w, photo_h,
        )
        c.drawImage(self.lateral_image_path, dx, dy, draw_w, draw_h,
                    preserveAspectRatio=False, anchor="sw", mask="auto")
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(_NB_BLUE)
        c.drawCentredString(dx + draw_w / 2, dy - 4 * mm, "LATERAL SIDE (OUTBOARD)")

        # Letter labels: A, B, C, ... drawn on the photo at each callout's
        # lateral centroid.
        from string import ascii_uppercase
        letters = list(ascii_uppercase)
        for i, co in enumerate(self.callouts):
            if i >= len(letters):
                break
            letter = letters[i]
            fx, fy = co.lateral_centroid
            ax = dx + fx * draw_w
            ay = dy + (1.0 - fy) * draw_h
            _draw_marker_circle(c, ax, ay, letter, radius=3.2 * mm,
                                 font_size=9)

        # Right half: legend.
        right_x = self.box_w * 0.48
        right_w = self.box_w - right_x
        legend_h = below_title * 0.55
        legend_y_top = self.box_h - 14 * mm
        n = len(self.callouts)
        if n > 0:
            row_h = min(8 * mm, legend_h / max(n, 1))
            for i, co in enumerate(self.callouts):
                if i >= len(letters):
                    break
                letter = letters[i]
                ly = legend_y_top - (i + 1) * row_h
                second = co.hex_color or ""
                _draw_legend_two_line(
                    c, marker=letter, name=co.display_name, second_line=second,
                    lx=right_x, ly=ly, col_w=right_w, row_h=row_h,
                    color_swatch_hex=co.hex_color,
                )

        # Dominant Colors Palette strip — bottom of right column.
        palette_band_y = photo_y - 4 * mm
        palette_band_h = max(0.0, palette_band_y - 12 * mm)
        # Header.
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(_NB_BLUE)
        c.drawString(right_x, photo_y + 6 * mm, "DOMINANT COLORS PALETTE")

        # Swatch row.
        sw_y = photo_y - 14 * mm
        sw_h = 16 * mm
        n_sw = max(1, len(self.dominant_palette))
        sw_w = (self.box_w - right_x) / max(n_sw, 1) - 1 * mm
        for i, pe in enumerate(self.dominant_palette):
            x = right_x + i * (sw_w + 1 * mm)
            try:
                fill = colors.HexColor(pe.hex)
            except Exception:
                fill = colors.white
            c.setFillColor(fill)
            c.setStrokeColor(colors.HexColor("#888888"))
            c.setLineWidth(0.4)
            c.rect(x, sw_y, sw_w, sw_h, stroke=1, fill=1)
            c.setFont("Helvetica", 7)
            c.setFillColor(_TEXT_DARK)
            c.drawCentredString(x + sw_w / 2, sw_y - 3.5 * mm, pe.hex)

        # Summary text at bottom-left under the photo.
        if self.summary:
            c.setFont("Helvetica-Oblique", 8.5)
            c.setFillColor(_TEXT_MID)
            # Wrap manually.
            from reportlab.lib.utils import simpleSplit
            lines = simpleSplit(self.summary, "Helvetica-Oblique", 8.5,
                                 photo_w - 2 * mm)
            y = photo_y - 8 * mm
            for line in lines:
                c.drawString(photo_x, y, line)
                y -= 4 * mm


# Display-name + function lookup for the Parts List panel.
_MPP_DISPLAY_MAP = {
    "rubber-outsole":  "Outsole",
    "eva-midsole":     "Midsole",
    "shoe-laces":      "Laces",
    "vamp":            "Vamp / Toe Box",
    "mudguard":        "Mudguard",
    "quarter-lateral": "Quarter (Lateral)",
    "quarter-medial":  "Quarter (Medial)",
    "heel-counter":    "Heel Counter",
    "heel-tab":        "Heel Tab",
    "collar":          "Collar Lining",
    "tongue":          "Tongue",
    "eyestay":         "Eyestay",
}

_MPP_FUNCTION_MAP = {
    "rubber-outsole":  "Traction / Durability",
    "eva-midsole":     "Cushioning",
    "shoe-laces":      "Lace Closure",
    "vamp":            "Toe Coverage",
    "mudguard":        "Toe Protection",
    "quarter-lateral": "Structure / Style",
    "quarter-medial":  "Structure / Style",
    "heel-counter":    "Heel Support",
    "heel-tab":        "On / Off Pull",
    "collar":          "Ankle Padding",
    "tongue":          "Lacing Comfort",
    "eyestay":         "Lace Passage",
}


class _MaterialsPartsPalette:
    """Landscape A4 colorway page 3: three engineering reference panels
    — Fabric Swatches, Parts List, Color Palette — laid out as three
    columns.

    Was originally drafted for the tech-drawings master sheet; moved
    here so the techdrawings PDF stays line-art only.

    Parameters:
      ga: GeometryAnalysis (for ga.components).
      materials: List[MaterialPrediction] parallel to ga.components.
      vibrant_palette: List[PaletteEntry] for the right column.
      face_uvs: (Nf, 3, 2) per-face UV coords (used to crop swatches).
      diffuse_image: (H, W, 3) uint8 texture map.
      swatch_cache_dir: where to write the cropped swatch PNGs.
    """

    def __init__(self, ga, materials, vibrant_palette,
                 face_uvs, diffuse_image, swatch_cache_dir,
                 title_main: str, box_w: float, box_h: float):
        from reportlab.platypus import Flowable
        self._Flowable = Flowable
        self.ga = ga
        self.materials = materials
        self.vibrant_palette = list(vibrant_palette or [])
        self.face_uvs = face_uvs
        self.diffuse_image = diffuse_image
        self.swatch_cache_dir = swatch_cache_dir
        self.title_main = title_main
        self.box_w = box_w
        self.box_h = box_h

    def as_flowable(self):
        outer = self
        Flowable = self._Flowable

        class _F(Flowable):
            def wrap(self, aw, ah):
                return outer.box_w, outer.box_h

            def draw(self):
                outer._draw(self.canv)

        return _F()

    def _swatch_png(self, idx, comp):
        """Save (and return path of) a UV-cropped fabric swatch for
        `comp`. Falls back to a solid-colour PNG if UVs/diffuse missing."""
        if not self.swatch_cache_dir:
            return None
        from .material import extract_component_patch
        out_path = os.path.join(
            self.swatch_cache_dir, f"mpp_swatch_{idx}_{comp.name}.png",
        )
        patch = None
        if self.face_uvs is not None and self.diffuse_image is not None:
            try:
                patch = extract_component_patch(
                    face_indices=comp.face_indices,
                    face_uvs=self.face_uvs,
                    diffuse=self.diffuse_image,
                    fill_rgb=tuple(int(x) for x in (getattr(comp, "dominant_color_rgb",
                                                              (200, 200, 200)) or
                                                      (200, 200, 200))),
                    target_size=160,
                )
            except Exception:
                patch = None
        if patch is None:
            from PIL import Image as _PIL
            rgb_t = getattr(comp, "dominant_color_rgb", None) or (200, 200, 200)
            import numpy as _np
            arr = _np.zeros((128, 128, 3), dtype=_np.uint8)
            arr[..., 0] = int(rgb_t[0]); arr[..., 1] = int(rgb_t[1]); arr[..., 2] = int(rgb_t[2])
            _PIL.fromarray(arr).save(out_path)
            return out_path
        from PIL import Image as _PIL
        _PIL.fromarray(patch).save(out_path)
        return out_path

    def _draw(self, c):
        # Title.
        c.setFont("Helvetica-Bold", 16)
        c.setFillColor(_TEXT_DARK)
        c.drawCentredString(self.box_w / 2, self.box_h - 7 * mm, self.title_main)
        c.setFont("Helvetica", 10)
        c.setFillColor(_NB_BLUE)
        c.drawCentredString(self.box_w / 2, self.box_h - 11.5 * mm,
                             "Fabric swatches · Parts list · Color palette")

        content_top = self.box_h - 16 * mm
        content_bot = 0
        content_h = content_top - content_bot

        # Three columns: swatches | parts list | palette.
        # The parts list is the widest because it's a 4-column table.
        gap = 6 * mm
        col_w = (self.box_w - 2 * gap)
        sw_w = col_w * 0.22
        pl_w = col_w * 0.48
        pa_w = col_w * 0.30

        sw_x = 0
        pl_x = sw_x + sw_w + gap
        pa_x = pl_x + pl_w + gap

        components = list(self.ga.components)[:12]

        # ----- Column 1: Fabric Swatches -----
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(_NB_BLUE)
        c.drawString(sw_x, content_top - 4, "FABRIC SWATCHES")
        c.setFont("Helvetica", 7)
        c.setFillColor(_TEXT_MID)
        c.drawString(sw_x, content_top - 13, "Sampled from each component's UV texture.")

        n = min(len(components), 8)
        if n > 0:
            row_h = (content_h - 22) / n
            swatch_size = min(28 * mm, row_h - 4)
            for i, comp in enumerate(components[:n]):
                ry = content_top - 22 - (i + 1) * row_h
                swatch_path = self._swatch_png(i, comp)
                sy = ry + (row_h - swatch_size) / 2
                if swatch_path and os.path.exists(swatch_path):
                    c.drawImage(swatch_path, sw_x, sy,
                                 swatch_size, swatch_size,
                                 preserveAspectRatio=False, anchor="sw",
                                 mask="auto")
                # Label + material.
                tx = sw_x + swatch_size + 3 * mm
                name = _MPP_DISPLAY_MAP.get(comp.name, comp.name)
                mat = ""
                if self.materials and i < len(self.materials) and self.materials[i] is not None:
                    mat = getattr(self.materials[i], "label", "") or ""
                c.setFont("Helvetica-Bold", 8.5)
                c.setFillColor(_TEXT_DARK)
                c.drawString(tx, sy + swatch_size / 2 + 2, name)
                if mat:
                    c.setFont("Helvetica", 7.5)
                    c.setFillColor(_TEXT_MID)
                    c.drawString(tx, sy + swatch_size / 2 - 7, mat)

        # ----- Column 2: Parts List -----
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(_NB_BLUE)
        c.drawString(pl_x, content_top - 4, "PARTS LIST")
        c.setFont("Helvetica", 7)
        c.setFillColor(_TEXT_MID)
        c.drawString(pl_x, content_top - 13,
                     "Detected components with inferred material and primary function.")

        cols_w = [pl_w * 0.06, pl_w * 0.32, pl_w * 0.30, pl_w * 0.32]
        header_h = 13
        table_top = content_top - 22
        table_bot = content_bot + 10
        nrows = len(components)
        if nrows > 0:
            body_h = (table_top - header_h) - table_bot
            row_h_t = min(13 * mm, body_h / max(nrows, 1))
            # Header row.
            c.setFillColor(_NB_BLUE)
            c.rect(pl_x, table_top - header_h, pl_w, header_h,
                   stroke=0, fill=1)
            c.setFont("Helvetica-Bold", 8)
            c.setFillColor(colors.white)
            cx = pl_x + 3
            for ct, cw_ in zip(["NO", "PART NAME", "MATERIAL", "FUNCTION"], cols_w):
                c.drawString(cx, table_top - header_h + 4, ct)
                cx += cw_
            # Body rows.
            for i, comp in enumerate(components):
                row_y = table_top - header_h - (i + 1) * row_h_t
                if i % 2 == 0:
                    c.setFillColor(colors.HexColor("#F6F6F8"))
                    c.rect(pl_x, row_y, pl_w, row_h_t, stroke=0, fill=1)
                mat = ""
                if self.materials and i < len(self.materials) and self.materials[i] is not None:
                    mat = getattr(self.materials[i], "label", "") or ""
                fn = _MPP_FUNCTION_MAP.get(comp.name, "")
                name = _MPP_DISPLAY_MAP.get(comp.name, comp.name)
                cx = pl_x + 3
                c.setFont("Helvetica", 8)
                c.setFillColor(_TEXT_DARK)
                for s, cw_ in zip([str(i + 1), name, mat, fn], cols_w):
                    while s and c.stringWidth(s, "Helvetica", 8) > cw_ - 4:
                        s = s[:-1]
                    c.drawString(cx, row_y + row_h_t / 2 - 2.5, s)
                    cx += cw_

        # ----- Column 3: Color Palette -----
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(_NB_BLUE)
        c.drawString(pa_x, content_top - 4, "COLOR PALETTE")
        c.setFont("Helvetica", 7)
        c.setFillColor(_TEXT_MID)
        c.drawString(pa_x, content_top - 13,
                     "Vibrant per-component swatches with hex codes.")

        pal = self.vibrant_palette[:8]
        m_n = len(pal)
        if m_n > 0:
            row_h_p = (content_h - 22) / m_n
            sw_h = min(14 * mm, row_h_p - 4)
            for i, entry in enumerate(pal):
                ry = content_top - 22 - (i + 1) * row_h_p
                try:
                    c.setFillColor(colors.HexColor(entry.hex))
                except Exception:
                    c.setFillColor(colors.gray)
                c.setStrokeColor(colors.HexColor("#888888"))
                c.setLineWidth(0.4)
                c.rect(pa_x, ry + (row_h_p - sw_h) / 2,
                       18 * mm, sw_h, stroke=1, fill=1)
                name = getattr(entry, "pantone", "") or entry.hex
                short = (name.split(" ", 2)[-1]
                         if name and len(name.split(" ")) > 2 else name)
                c.setFont("Helvetica-Bold", 8.5)
                c.setFillColor(_TEXT_DARK)
                c.drawString(pa_x + 20 * mm, ry + row_h_p / 2 + 2, short[:30])
                c.setFont("Helvetica", 7.5)
                c.setFillColor(_TEXT_MID)
                c.drawString(pa_x + 20 * mm, ry + row_h_p / 2 - 7,
                              f"{entry.hex}")


def _lateral_annotation_flowables(lineart_path, id_mask_path, id_palette,
                                   labels, styles,
                                   title="Bill of Materials — Component map",
                                   subtitle="") -> list:
    """Build the flowables for the BOM page-1 annotation diagram (landscape A4)."""
    flowables = [Paragraph(title, styles["title"])]
    if subtitle:
        flowables.append(Paragraph(subtitle, styles["small"]))
    flowables.append(Spacer(1, 3 * mm))
    cb = _LateralAnnotationCallout(
        lineart_path=lineart_path,
        id_mask_path=id_mask_path,
        id_palette=id_palette,
        labels=labels,
        box_w=267 * mm,
        box_h=155 * mm,
    )
    flowables.append(cb.as_flowable())
    return flowables


def _colorway_callout_flowables(image_path, id_mask_path, id_palette,
                                 components, styles) -> list:
    flowables = []
    flowables.append(Paragraph("Colorway — Component Callouts", styles["title"]))
    flowables.append(Paragraph(
        "Each label shows the component's dominant texture color and its "
        "approximate Pantone TCX match. Arrows point to the centroid of the "
        "component on the lateral-side render.",
        styles["small"],
    ))
    flowables.append(Spacer(1, 3 * mm))
    # Use the landscape-A4 work area: 267 mm × ~150 mm.
    callout = _ColorwayCallout(
        image_path=image_path,
        id_mask_path=id_mask_path,
        id_palette=id_palette,
        components=components,
        box_w=267 * mm,
        box_h=150 * mm,
    )
    flowables.append(callout.as_flowable())
    return flowables


# Per-component colorway cards

def _component_cards_flowables(
    component_names: List[str],
    component_palettes: List[List[PaletteEntry]],
    materials: List,    # List[MaterialPrediction] but typed loosely to avoid import
    styles: dict,
) -> list:
    """A grid of per-component cards. Each card shows:
      - Component name + inferred material (with confidence)
      - 1-3 swatches (the mini-palette in OKLab) with area %
      - Hex / Pantone TCX (ΔE2000) / RAL Classic / CMYK for the dominant swatch
    Cards laid out 2 per row on a landscape page.
    """
    flowables = []
    flowables.append(Paragraph("Per-component colorway", styles["title"]))
    flowables.append(Paragraph(
        "Each card is one segmented component. Colors clustered in OKLab "
        "(perceptually uniform); Pantone TCX and RAL Classic matches via "
        "ΔE2000 (CIEDE2000) under the D50 illuminant. CMYK via ICC profile "
        "(FOGRA39-class). Material class from OpenCLIP ViT-B/32 zero-shot, "
        "fused with PBR-feature heuristics — confidence below 0.35 or "
        "CLIP/PBR disagreement is flagged for human review.",
        styles["small"],
    ))
    flowables.append(Spacer(1, 3 * mm))

    CARD_W = 130 * mm
    rows = []
    pair: list = []
    for name, mini_palette, mat in zip(component_names, component_palettes, materials):
        card = _build_component_card(name, mini_palette, mat, styles, CARD_W)
        pair.append(card)
        if len(pair) == 2:
            rows.append(pair)
            pair = []
    if pair:
        pair.append(Spacer(1, 1))
        rows.append(pair)
    grid = Table(rows, colWidths=[CARD_W, CARD_W])
    grid.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    flowables.append(grid)
    return flowables


def _build_component_card(
    name: str,
    mini_palette: List[PaletteEntry],
    mat,                # MaterialPrediction
    s: dict,
    card_w: float,
) -> Table:
    # Header line: component name + material label
    review_tag = " · REVIEW" if getattr(mat, "needs_review", False) else ""
    header_html = (
        f"<b>{name}</b> &nbsp;·&nbsp; "
        f"<font color='#333333'>{mat.label}</font> "
        f"<font color='#888888' size='8'>({mat.confidence:.0%}{review_tag})</font>"
    )
    rows = [[Paragraph(header_html, s["body"])]]

    # CLIP top-3 line (compact)
    if mat.clip_top3:
        top3 = " / ".join(f"{lbl} {p:.0%}" for lbl, p in mat.clip_top3[:3])
        rows.append([Paragraph(
            f"<font color='#666666' size='7.5'>CLIP top-3: {top3}</font>",
            s["body"])])

    # Mini-palette as a horizontal row of swatches with metadata under each
    if mini_palette:
        n = len(mini_palette)
        sw_table_rows = [[], [], []]
        for entry in mini_palette:
            # Swatch
            swatch = Table([[""]], colWidths=[18 * mm], rowHeights=[18 * mm])
            r, g, b = entry.rgb
            swatch.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1),
                 colors.Color(r / 255, g / 255, b / 255)),
                ("BOX", (0, 0), (-1, -1), 0.3, colors.HexColor("#888888")),
            ]))
            sw_table_rows[0].append(swatch)
            sw_table_rows[1].append(Paragraph(
                f"<font size='7.5'><b>{entry.hex}</b><br/>"
                f"{entry.fraction * 100:.0f}%</font>",
                s["body"]))
            sw_table_rows[2].append(Paragraph(
                f"<font size='6.5' color='#555555'>"
                f"{entry.pantone}<br/>"
                f"ΔE2000 {entry.pantone_deltaE:.1f}<br/>"
                f"{entry.ral}<br/>"
                f"C{entry.cmyk[0]} M{entry.cmyk[1]} Y{entry.cmyk[2]} K{entry.cmyk[3]}"
                f"</font>",
                s["body"]))
        # Equal columns sized to fit card width
        col_w = (card_w - 8 * mm) / max(n, 1)
        sw_table = Table(sw_table_rows, colWidths=[col_w] * n)
        sw_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 1),
            ("RIGHTPADDING", (0, 0), (-1, -1), 1),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ]))
        rows.append([sw_table])

    card = Table(rows, colWidths=[card_w])
    card.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#666666")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F4F4F4")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, 0), (-1, 0), 0.3, colors.HexColor("#888888")),
    ]))
    return card


# Colorway PDF

def write_colorway(
    output_path: str,
    header: TechPackHeader,
    palette: List[PaletteEntry],
    component_swatches: List[Tuple[str, PaletteEntry]],
    callout_image_path: Optional[str] = None,
    callout_id_mask_path: Optional[str] = None,
    callout_id_palette: Optional[List[Tuple[float, float, float]]] = None,
    component_names: Optional[List[str]] = None,
    component_palettes: Optional[List[List[PaletteEntry]]] = None,
    materials: Optional[List] = None,   # List[MaterialPrediction]
    colorway_anatomy: Optional[dict] = None,
    # Inputs for the new page-3 ("Materials, Parts & Palette" reference).
    ga: Optional[GeometryAnalysis] = None,
    face_uvs=None,                                       # (Nf, 3, 2) np.ndarray
    diffuse_image=None,                                  # (H, W, 3) np.ndarray
    vibrant_palette: Optional[List[PaletteEntry]] = None,
    swatch_cache_dir: Optional[str] = None,
) -> str:
    s = _styles()
    doc = _make_doc(output_path, header, "Colorway", pagesize=landscape(A4))
    story = []

    # New: fabric anatomy + color anatomy pages — drawn first so they're
    # the headline of the colorway PDF. Both use the unified Callout
    # payload built by pipeline._build_anatomy_payload.
    # NOTE: SimpleDocTemplate's Frame adds 6 pt of padding on each side,
    # so the usable inner box is (page - margins - 2*6pt) in each axis.
    # Without that subtraction the flowable overflows the frame and
    # reportlab raises LayoutError ("too large on page N").
    L_W, L_H = landscape(A4)
    margin = 15 * mm
    top_margin = 22 * mm
    bottom_margin = 22 * mm
    frame_pad = 6  # reportlab Frame default (pt)
    box_w = L_W - 2 * margin - 2 * frame_pad
    box_h = L_H - top_margin - bottom_margin - 2 * frame_pad

    have_anatomy = (
        colorway_anatomy is not None
        and colorway_anatomy.get("lateral_image_path")
        and os.path.exists(colorway_anatomy["lateral_image_path"])
    )
    if have_anatomy:
        callouts = colorway_anatomy.get("callouts", [])
        medial_path = colorway_anatomy.get("medial_image_path")
        if medial_path and os.path.exists(medial_path):
            fa = _FabricAnatomyInfographic(
                lateral_image_path=colorway_anatomy["lateral_image_path"],
                medial_image_path=medial_path,
                callouts=callouts,
                title_main=colorway_anatomy.get(
                    "fabric_title_main",
                    f"FABRIC (MATERIAL) ANATOMY OF THE {header.model_name.upper()}",
                ),
                title_sub=colorway_anatomy.get(
                    "fabric_title_sub",
                    "Inferred materials per visible component.",
                ),
                box_w=box_w, box_h=box_h,
            )
            story.append(fa.as_flowable())
            story.append(PageBreak())

        ca = _ColorAnatomyInfographic(
            lateral_image_path=colorway_anatomy["lateral_image_path"],
            callouts=callouts,
            dominant_palette=palette,
            vibrant_palette=colorway_anatomy.get("vibrant_palette"),
            title_main=colorway_anatomy.get(
                "color_title_main",
                f"COLOR ANATOMY OF THE {header.model_name.upper()}",
            ),
            title_sub=colorway_anatomy.get(
                "color_title_sub",
                "Dominant color per component + global palette.",
            ),
            summary=colorway_anatomy.get(
                "color_summary",
                "Per-component dominant colors are sampled from the diffuse "
                "texture map and matched to nearest Pantone TCX and RAL "
                "Classic references via CIE ΔE2000.",
            ),
            box_w=box_w, box_h=box_h,
        )
        story.append(ca.as_flowable())

        # Optional page 3 — Materials, Parts & Palette reference.
        # Combines the three engineering panels the user wanted moved
        # off the techdrawings sheet:
        #   • Fabric Swatches (UV-cropped texture squares per component)
        #   • Parts List (# / Part / Material / Function)
        #   • Color Palette (vibrant per-component swatches)
        # Only rendered if `ga` is supplied; the first two pages remain
        # the headline regardless.
        if ga is not None:
            story.append(PageBreak())
            mpp = _MaterialsPartsPalette(
                ga=ga,
                materials=materials or [],
                vibrant_palette=vibrant_palette or palette or [],
                face_uvs=face_uvs,
                diffuse_image=diffuse_image,
                swatch_cache_dir=swatch_cache_dir,
                title_main=colorway_anatomy.get(
                    "mpp_title_main",
                    f"MATERIALS · PARTS · PALETTE — {header.model_name.upper()}",
                ),
                box_w=box_w, box_h=box_h,
            )
            story.append(mpp.as_flowable())

        doc.build(
            story,
            onFirstPage=lambda c, d: _header_footer(c, d, header, "COLORWAY"),
            onLaterPages=lambda c, d: _header_footer(c, d, header, "COLORWAY"),
        )
        return output_path

    # ---- Legacy fallback (only when the new infographic data is
    # missing — keeps `write_colorway` usable for callers that don't
    # supply a colorway_anatomy payload).
    if (callout_image_path and os.path.exists(callout_image_path)
            and callout_id_mask_path and os.path.exists(callout_id_mask_path)
            and callout_id_palette):
        story.extend(_colorway_callout_flowables(
            callout_image_path, callout_id_mask_path,
            callout_id_palette, component_swatches, s,
        ))
        story.append(PageBreak())

    if component_names and component_palettes and materials:
        story.extend(_component_cards_flowables(
            component_names, component_palettes, materials, s,
        ))
        story.append(PageBreak())

    story.append(Paragraph("Colorway", s["title"]))
    story.append(Paragraph(
        "Palette is the K dominant colors of the diffuse texture map, sorted "
        "by area share. Pantone matches use TCX (Textile Cotton eXtended) "
        "reference data and CIE Lab ΔE2000 (CIEDE2000) under the D50 "
        "illuminant; treat ΔE2000 &gt; 4 as 'approximate'.",
        s["small"],
    ))
    story.append(Spacer(1, 4 * mm))

    # Palette
    story.append(Paragraph("Diffuse-texture palette", s["h2"]))
    rows = [["Swatch", "HEX", "RGB", "CMYK", "Pantone TCX (approx)", "ΔE00", "Area %"]]
    for p in palette:
        rows.append([
            "",  # painted via style
            p.hex,
            f"{p.rgb[0]}, {p.rgb[1]}, {p.rgb[2]}",
            "{}/{}/{}/{}".format(*p.cmyk),
            p.pantone,
            f"{p.pantone_deltaE:.1f}",
            f"{100 * p.fraction:.1f}%",
        ])
    t = Table(rows, colWidths=[18 * mm, 22 * mm, 28 * mm, 28 * mm,
                                50 * mm, 12 * mm, 17 * mm])
    style_cmds = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#222222")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("ALIGN", (5, 1), (6, -1), "RIGHT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]
    for ri, p in enumerate(palette, start=1):
        rgb = p.rgb
        style_cmds.append(("BACKGROUND", (0, ri), (0, ri),
                           colors.Color(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)))
    t.setStyle(TableStyle(style_cmds))
    story.append(t)
    story.append(Spacer(1, 8 * mm))

    # Per-component swatches
    story.append(Paragraph("Per-component colors (dominant)", s["h2"]))
    rows = [["Swatch", "Component", "HEX", "RGB", "Pantone TCX (approx)", "ΔE00"]]
    for name, p in component_swatches:
        rows.append([
            "", name, p.hex,
            f"{p.rgb[0]}, {p.rgb[1]}, {p.rgb[2]}",
            p.pantone,
            f"{p.pantone_deltaE:.1f}",
        ])
    ct = Table(rows, colWidths=[18 * mm, 32 * mm, 22 * mm, 28 * mm, 64 * mm, 12 * mm])
    style_cmds = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#222222")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("ALIGN", (5, 1), (5, -1), "RIGHT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]
    for ri, (_, p) in enumerate(component_swatches, start=1):
        rgb = p.rgb
        style_cmds.append(("BACKGROUND", (0, ri), (0, ri),
                           colors.Color(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)))
    ct.setStyle(TableStyle(style_cmds))
    story.append(ct)

    doc.build(
        story,
        onFirstPage=lambda c, d: _header_footer(c, d, header, "COLORWAY"),
        onLaterPages=lambda c, d: _header_footer(c, d, header, "COLORWAY"),
    )
    return output_path


# Construction PDF

def make_header(source_path: str, model_id: str, model_name: str,
                date_iso: Optional[str] = None,
                designer: str = "—", factory: str = "—", season: str = "—",
                ) -> TechPackHeader:
    h = hashlib.sha256()
    with open(source_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return TechPackHeader(
        model_id=model_id,
        model_name=model_name,
        date_iso=date_iso or _dt.date.today().isoformat(),
        source_file=source_path,
        source_hash=h.hexdigest(),
        designer=designer,
        factory=factory,
        season=season,
    )
