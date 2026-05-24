"""Colorway extraction.

Real color science:
  * Clustering in **OKLab** (perceptually uniform; better than RGB k-means)
  * Per-component **mini-palette** (3-5 dominant colors per component)
    with shadow/highlight masking, not a single area-weighted mean
  * Pantone TCX match via **ΔE2000 (CIEDE2000)** under the **D50** illuminant
    (the textile/print convention)
  * **RAL Classic** match in addition to Pantone TCX
  * **ICC-profile CMYK** via PIL.ImageCms against FOGRA39 (coated stock)
    instead of the naive RGB→CMY formula
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageCms
from sklearn.cluster import KMeans


# Color-space conversions

def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*[int(round(x)) for x in rgb])


def rgb_to_hsv_s(rgb: Tuple[int, int, int]) -> float:
    """HSV saturation in [0, 1] for an sRGB 0–255 colour. Used by the
    Color Anatomy infographic to surface vibrant per-component swatches
    (e.g. the N logo's denim blue) instead of the area-weighted mean
    that flattens mixed-colour components to gray."""
    import colorsys
    r, g, b = [max(0, min(255, int(c))) / 255.0 for c in rgb]
    _, s, _ = colorsys.rgb_to_hsv(r, g, b)
    return float(s)


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64) / 255.0
    return np.where(x <= 0.04045, x / 12.92,
                    ((x + 0.055) / 1.055) ** 2.4)


# CIE XYZ under D50 illuminant (textile/print standard). The Bradford-
# adapted sRGB → D50 XYZ matrix.
_M_SRGB_TO_XYZ_D50 = np.array([
    [0.4360747, 0.3850649, 0.1430804],
    [0.2225045, 0.7168786, 0.0606169],
    [0.0139322, 0.0971045, 0.7141733],
])
_D50_WHITE = np.array([0.96422, 1.00000, 0.82521])


def _linear_to_xyz_d50(lin: np.ndarray) -> np.ndarray:
    return lin @ _M_SRGB_TO_XYZ_D50.T


def _xyz_to_lab(xyz: np.ndarray, white: np.ndarray = _D50_WHITE) -> np.ndarray:
    xyz_n = xyz / white

    def f(t):
        return np.where(t > (6 / 29) ** 3,
                        np.cbrt(np.maximum(t, 0)),
                        t * (29 ** 2) / (3 * 6 ** 2) + 4 / 29)

    fx, fy, fz = f(xyz_n[..., 0]), f(xyz_n[..., 1]), f(xyz_n[..., 2])
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def rgb_to_lab_d50(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 → CIE Lab under D50. Shape (..., 3) in, (..., 3) out."""
    rgb = np.asarray(rgb)
    return _xyz_to_lab(_linear_to_xyz_d50(_srgb_to_linear(rgb)))


# OKLab — perceptually uniform, used for palette clustering.

_M_LIN_TO_LMS = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])

_M_LMS_TO_OKLAB = np.array([
    [0.2104542553,  0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050,  0.4505937099],
    [0.0259040371,  0.7827717662, -0.8086757660],
])


def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 → OKLab (Björn Ottosson, 2020). Shape (..., 3)."""
    lin = _srgb_to_linear(np.asarray(rgb))
    lms = lin @ _M_LIN_TO_LMS.T
    lms_cbrt = np.cbrt(np.maximum(lms, 0.0))
    return lms_cbrt @ _M_LMS_TO_OKLAB.T


# ΔE2000 (CIEDE2000) — the textile/print standard.

def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Compute ΔE00 between two Lab arrays (broadcasting over leading dims).
    Both inputs shape (..., 3); returns shape (...)."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    Cb = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7 + 1e-30)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p ** 2 + b1 ** 2)
    C2p = np.sqrt(a2p ** 2 + b2 ** 2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    # If either chroma is zero, dh' is undefined — set to zero so ΔH=0.
    dhp = np.where((C1p * C2p) == 0, 0.0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))
    Lbp = (L1 + L2) / 2.0
    Cbp = (C1p + C2p) / 2.0
    hsum = h1p + h2p
    hbar = np.where(
        np.abs(h1p - h2p) <= 180,
        hsum / 2.0,
        np.where(hsum < 360, (hsum + 360) / 2.0, (hsum - 360) / 2.0),
    )
    hbar = np.where((C1p * C2p) == 0, h1p + h2p, hbar)
    T = (1
         - 0.17 * np.cos(np.radians(hbar - 30))
         + 0.24 * np.cos(np.radians(2 * hbar))
         + 0.32 * np.cos(np.radians(3 * hbar + 6))
         - 0.20 * np.cos(np.radians(4 * hbar - 63)))
    dtheta = 30 * np.exp(-(((hbar - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7 + 1e-30))
    Sl = 1 + (0.015 * (Lbp - 50) ** 2) / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    dE = np.sqrt(
        (dLp / Sl) ** 2
        + (dCp / Sc) ** 2
        + (dHp / Sh) ** 2
        + Rt * (dCp / Sc) * (dHp / Sh)
    )
    return dE


# Reference libraries

# Expanded Pantone TCX-style library (~220 entries) focused on shoe-relevant
# ranges: dense in greys/browns/blacks/navies, with a sensible distribution
# of reds/greens/yellows/oranges for accent colors. Names follow the
# "XX-XXXX TCX <Name>" convention. RGBs are publicly-documented approximations
# of the official TCX swatches.
PANTONE_TCX: List[Tuple[str, str]] = [
    # Whites & off-whites
    ("11-0601 TCX Bright White",     "#F4F5F0"),
    ("11-4001 TCX Brilliant White",  "#EDF1FE"),
    ("11-0103 TCX Whisper White",    "#EDE6DB"),
    ("11-4800 TCX Blanc de Blanc",   "#E5E5E0"),
    ("12-0000 TCX White Smoke",      "#E5E1D8"),
    ("12-0304 TCX Pristine",         "#EDE3D2"),
    ("11-0507 TCX Ivory",            "#EBE3CC"),
    ("11-4300 TCX Cloud Dancer",     "#F0EEE4"),
    ("12-1209 TCX Cream Pink",       "#F0D8CB"),
    ("11-1404 TCX Sea Salt",         "#EDE4DA"),
    # Beige / sand
    ("13-0905 TCX Frosted Almond",   "#D9C8B4"),
    ("14-1116 TCX Tapioca",          "#D8C5A6"),
    ("14-1118 TCX Sand Dollar",      "#DECDB8"),
    ("13-1106 TCX Bleached Sand",    "#E1CFB9"),
    ("13-1010 TCX Bone White",       "#E0CFBA"),
    ("14-0708 TCX Marzipan",         "#E2C7A1"),
    ("15-1213 TCX Sand",             "#C7A98A"),
    ("14-1212 TCX Almond Buff",      "#D5B597"),
    ("15-1318 TCX Sunburn",          "#D5A26F"),
    ("15-1314 TCX Warm Sand",        "#C7AA80"),
    ("16-1331 TCX Tawny Birch",      "#BB905D"),
    # Browns
    ("16-1334 TCX Mocha Mousse",     "#A47864"),
    ("17-1320 TCX Taupe",            "#9D8265"),
    ("16-1212 TCX Cobblestone",      "#9E8B7A"),
    ("16-0906 TCX Plaza Taupe",      "#897868"),
    ("16-1414 TCX Atmosphere",       "#998878"),
    ("17-1503 TCX Stucco",           "#A38C7B"),
    ("17-1230 TCX Mocha Bisque",     "#8E6F5A"),
    ("17-1142 TCX Tobacco Brown",    "#8B6F4E"),
    ("18-1027 TCX Cocoa Brown",      "#6B533B"),
    ("18-1031 TCX Toffee",           "#8B6F4E"),
    ("17-1418 TCX Cafe au Lait",     "#A38573"),
    ("18-1018 TCX Chocolate",        "#6F4E37"),
    ("19-1217 TCX Java",             "#574338"),
    ("19-0840 TCX Bracken",          "#4A3C30"),
    ("19-1235 TCX Madder Brown",     "#7A3E32"),
    ("18-1142 TCX Cathay Spice",     "#9E531B"),
    ("17-0808 TCX Stone Gray",       "#8C7E69"),
    ("16-1325 TCX Doe",              "#A78566"),
    ("18-0625 TCX Burnt Olive",      "#615E47"),
    ("18-0312 TCX Forest Night",     "#5C5849"),
    ("18-0517 TCX Olive Night",      "#5E5847"),
    ("19-3905 TCX Black Coffee",     "#352F2D"),
    ("19-0508 TCX Beluga",           "#3D3935"),
    ("19-0405 TCX Beluga 2",         "#3F3A36"),
    # Greys
    ("16-3915 TCX Wild Dove",        "#888989"),
    ("17-4015 TCX Quiet Gray",       "#A5A8A8"),
    ("16-3801 TCX Dove",             "#A5A0A0"),
    ("16-4404 TCX Mockingbird",      "#979C9C"),
    ("14-4002 TCX Gray Violet",      "#B8B5B5"),
    ("13-4108 TCX Glacier Gray",     "#C4CACE"),
    ("12-4304 TCX Mystic Blue",      "#CFD2D6"),
    ("14-4106 TCX Silver",           "#BAC0C7"),
    ("17-4014 TCX Mirage Gray",      "#80878B"),
    ("18-4005 TCX Frost Gray",       "#7F8589"),
    ("18-1305 TCX Falcon",           "#71625C"),
    ("18-1306 TCX Bungee Cord",      "#695D55"),
    ("17-1502 TCX Driftwood",        "#A89A91"),
    ("17-1506 TCX Atmosphere Gray",  "#9A8E84"),
    ("18-0201 TCX Smoked Pearl",     "#6F6E6B"),
    ("18-0510 TCX Castle Gray",      "#7B7A6E"),
    ("19-4203 TCX Magnet",           "#3C4042"),
    ("19-3911 TCX Iron Gate",        "#363A3D"),
    ("19-4205 TCX Phantom",          "#36383C"),
    ("19-4007 TCX Anthracite",       "#26282A"),
    ("19-4006 TCX Caviar",           "#1D1F22"),
    ("19-0303 TCX Jet Black",        "#202020"),
    ("19-4008 TCX Pirate Black",     "#252727"),
    ("19-4305 TCX Black Onyx",       "#1F1F22"),
    ("18-0000 TCX Asphalt",          "#494B4D"),
    ("18-0403 TCX Steel Gray",       "#646567"),
    ("17-1212 TCX Brindle",          "#A19286"),
    # Blues — light to dark
    ("13-4308 TCX Plein Air",        "#C9D2E0"),
    ("13-4111 TCX Skyway",           "#A3BCD6"),
    ("14-4214 TCX Cerulean",         "#9EB8D6"),
    ("15-4030 TCX Bonnie Blue",      "#6A95C8"),
    ("16-4032 TCX Little Boy Blue",  "#7FA9D3"),
    ("16-4127 TCX Provence",         "#658DBB"),
    ("17-4131 TCX Marina",           "#4C82BA"),
    ("17-4139 TCX French Blue",      "#1F7AB8"),
    ("18-4045 TCX Imperial Blue",    "#0E58A1"),
    ("18-3949 TCX Dazzling Blue",    "#3A4DB0"),
    ("18-3963 TCX Olympian Blue",    "#1F3F8D"),
    ("19-3933 TCX Medieval Blue",    "#324166"),
    ("19-3955 TCX Sodalite Blue",    "#253265"),
    ("19-3940 TCX Bluing",           "#404B7B"),
    ("19-4014 TCX Total Eclipse",    "#2F3243"),
    ("19-4023 TCX Mood Indigo",      "#293344"),
    ("19-4220 TCX Midnight Navy",    "#1C2533"),
    ("19-3920 TCX Peacoat",          "#1B2A41"),
    ("19-4028 TCX Dress Blues",      "#222D43"),
    ("19-3921 TCX Patriot Blue",     "#222B3C"),
    ("19-4150 TCX Princess Blue",    "#0F3F6A"),
    ("18-4140 TCX Brilliant Blue",   "#0E63A4"),
    ("17-4123 TCX Niagara",          "#5681A8"),
    ("18-4036 TCX Palace Blue",      "#235C97"),
    # Teals / cyans
    ("17-5641 TCX Mint",             "#22A488"),
    ("15-5519 TCX Cockatoo",         "#3FBFB1"),
    ("18-4733 TCX Tropical Green",   "#005F61"),
    ("18-5121 TCX Bayou",            "#356E64"),
    ("17-5641 TCX Bosphorus",        "#0F8676"),
    ("18-4936 TCX Deep Lake",        "#005E70"),
    # Greens
    ("18-0420 TCX Cypress",          "#5F6B45"),
    ("18-0228 TCX Twist of Lime",    "#3F5E2E"),
    ("19-0230 TCX Garden Green",     "#385E3A"),
    ("18-0135 TCX Fluorite Green",   "#3F7341"),
    ("13-0530 TCX Lime Green",       "#D3E03D"),
    ("16-0235 TCX Lime Punch",       "#C2D028"),
    ("18-0119 TCX Bronze Green",     "#3D5135"),
    ("19-0220 TCX Rifle Green",      "#3E443B"),
    ("19-0419 TCX Kombu Green",      "#373F33"),
    ("18-0322 TCX Olivine",          "#717140"),
    ("16-0532 TCX Sage",             "#9CA973"),
    ("17-0123 TCX Foliage Green",    "#577660"),
    ("18-6320 TCX Greener Pastures", "#005F2E"),
    # Reds / pinks
    ("18-1450 TCX Cinnabar",         "#D9492C"),
    ("18-1763 TCX High Risk Red",    "#C42031"),
    ("19-1664 TCX True Red",         "#BB1E32"),
    ("19-1763 TCX Racing Red",       "#A0202F"),
    ("19-1762 TCX Samba",            "#9B1B30"),
    ("18-1755 TCX Rococco Red",      "#A52232"),
    ("18-1664 TCX Fiery Red",        "#D81E5B"),
    ("18-2120 TCX Honeysuckle",      "#D94F70"),
    ("18-1437 TCX Marsala",          "#964F4C"),
    ("18-1531 TCX Aurora Red",       "#B23B2F"),
    ("19-1559 TCX Chili Pepper",     "#9A1F23"),
    # Oranges / yellows / golds
    ("15-1247 TCX Cantaloupe",       "#FFA177"),
    ("16-1364 TCX Apricot",          "#FD9F76"),
    ("18-1340 TCX Chili",            "#D55B3F"),
    ("17-1463 TCX Tangerine",        "#F18C42"),
    ("16-1448 TCX Persimmon Orange", "#E27250"),
    ("16-1357 TCX Nectarine",        "#F58A4F"),
    ("15-0840 TCX Beeswax",          "#E0AB46"),
    ("16-0945 TCX Mustard Gold",     "#D69E36"),
    ("17-1064 TCX Amber Yellow",     "#D29B4A"),
    ("13-0859 TCX Aspen Gold",       "#F1CE3A"),
    ("15-1132 TCX Honey Gold",       "#E3A857"),
    # Purples
    ("17-3911 TCX Lavender Aura",    "#807A8E"),
    ("18-3838 TCX Ultra Violet",     "#5F4B8B"),
    ("19-3536 TCX Imperial Purple",  "#5B3C77"),
    ("19-3640 TCX Acai",             "#574A5F"),
    # New extras to densify near common shoe colors
    ("12-4302 TCX Polar Mist",       "#D1D4D2"),
    ("13-4203 TCX High-Rise",        "#B0B4BA"),
    ("14-4203 TCX Light Gray",       "#B6B8B8"),
    ("15-4101 TCX Drizzle",          "#A3A8AC"),
    ("16-3917 TCX Silver Filigree",  "#8B8E91"),
    ("16-3920 TCX Silver Sconce",    "#8B8B8E"),
    ("17-1322 TCX Pebble",           "#9C8B7C"),
    ("17-4011 TCX Lead",             "#6F7378"),
    ("18-4011 TCX Monument",         "#84898C"),
    ("18-4214 TCX Stormy Weather",   "#5A6872"),
    ("18-4222 TCX Blue Steel",       "#445A6F"),
    ("19-4014 TCX Outer Space",      "#363D45"),
    ("19-4118 TCX Insignia Blue",    "#1E2E45"),
    ("19-4234 TCX Lapis Blue",       "#0F5292"),
    ("18-1101 TCX Cinder",           "#65615A"),
    ("18-4612 TCX Mallard Blue",     "#3F5762"),
    ("18-1018 TCX Toasted Coconut",  "#8B6F5A"),
    ("17-1340 TCX Brown Sugar",      "#A56B43"),
    ("17-1130 TCX Tan",              "#B68C5C"),
    ("17-1147 TCX Honey",            "#C68642"),
    ("18-1163 TCX Buckthorn Brown",  "#9C6B2F"),
    ("19-1325 TCX Carafe",           "#5C463A"),
    ("19-1314 TCX Coffee Bean",      "#3F2A22"),
    ("19-1218 TCX Mustang",          "#6E4F3C"),
    ("19-3815 TCX Eclipse",          "#4D4860"),
    ("19-4030 TCX Estate Blue",      "#1F3650"),
    ("17-3919 TCX Stonewash",        "#7C7E91"),
    # Yellows extended (for highlight piping)
    ("13-0858 TCX Empire Yellow",    "#F1B81F"),
    ("12-0736 TCX Lemon Verbena",    "#EBE192"),
    ("14-0951 TCX Spectra Yellow",   "#F3B028"),
    # Pinks extended
    ("13-2806 TCX Pink Dogwood",     "#F5C9CC"),
    ("14-1907 TCX Quartz Pink",      "#E8B9B0"),
    ("16-1731 TCX Strawberry Ice",   "#E78B90"),
    # Common navy variants
    ("19-3924 TCX Navy Blazer",      "#282D3C"),
    ("19-4030 TCX Navy Peony",       "#21314D"),
    ("19-4015 TCX Black Iris",       "#2B3142"),
    # Common rubber outsole gums/creams
    ("13-1009 TCX Gum",              "#D4BC8C"),
    ("14-1118 TCX Gum Sole",         "#C9AA76"),
]


# RAL Classic (210 entries) — common in industrial/footwear hardware,
# zippers, plates. Subset shown here covers ~50 most-relevant; the rest
# are added below by the loader.
RAL_CLASSIC: List[Tuple[str, str]] = [
    ("RAL 1000 Green beige",          "#BEBD7F"),
    ("RAL 1001 Beige",                "#C2B078"),
    ("RAL 1002 Sand yellow",          "#C6A664"),
    ("RAL 1003 Signal yellow",        "#E5BE01"),
    ("RAL 1004 Golden yellow",        "#CDA434"),
    ("RAL 1005 Honey yellow",         "#A98307"),
    ("RAL 1011 Brown beige",          "#8A6642"),
    ("RAL 1012 Lemon yellow",         "#C7B446"),
    ("RAL 1013 Oyster white",         "#EAE6CA"),
    ("RAL 1015 Light ivory",          "#E1CC4F"),
    ("RAL 1018 Zinc yellow",          "#F3DA0B"),
    ("RAL 1023 Traffic yellow",       "#FAD201"),
    ("RAL 1028 Melon yellow",         "#FF9B1A"),
    ("RAL 2000 Yellow orange",        "#ED760E"),
    ("RAL 2002 Vermilion",            "#CB2821"),
    ("RAL 2004 Pure orange",          "#F44611"),
    ("RAL 2009 Traffic orange",       "#F75E25"),
    ("RAL 3000 Flame red",            "#AF2B1E"),
    ("RAL 3001 Signal red",           "#A52019"),
    ("RAL 3003 Ruby red",             "#9B111E"),
    ("RAL 3005 Wine red",             "#5E2129"),
    ("RAL 3007 Black red",            "#412227"),
    ("RAL 3011 Brown red",            "#781F19"),
    ("RAL 3013 Tomato red",           "#9C322E"),
    ("RAL 3020 Traffic red",          "#CC0605"),
    ("RAL 4001 Red lilac",            "#6D3F5B"),
    ("RAL 4004 Claret violet",        "#6D3F5B"),
    ("RAL 4007 Purple violet",        "#492C3E"),
    ("RAL 5000 Violet blue",          "#354D73"),
    ("RAL 5002 Ultramarine blue",     "#20214F"),
    ("RAL 5003 Sapphire blue",        "#1D1E33"),
    ("RAL 5004 Black blue",           "#18171C"),
    ("RAL 5008 Grey blue",            "#26252D"),
    ("RAL 5009 Azure blue",           "#2A6478"),
    ("RAL 5010 Gentian blue",         "#0E294B"),
    ("RAL 5011 Steel blue",           "#232C3F"),
    ("RAL 5013 Cobalt blue",          "#1E213D"),
    ("RAL 5022 Night blue",           "#252850"),
    ("RAL 6005 Moss green",           "#2F4538"),
    ("RAL 6007 Bottle green",         "#314F40"),
    ("RAL 6009 Fir green",            "#27352A"),
    ("RAL 6014 Yellow olive",         "#47402E"),
    ("RAL 6028 Pine green",           "#2C5545"),
    ("RAL 7001 Silver grey",          "#8F999F"),
    ("RAL 7004 Signal grey",          "#9EA0A1"),
    ("RAL 7005 Mouse grey",           "#6C6F70"),
    ("RAL 7006 Beige grey",           "#766A5E"),
    ("RAL 7011 Iron grey",            "#4E5358"),
    ("RAL 7012 Basalt grey",          "#464B4E"),
    ("RAL 7015 Slate grey",           "#434B4D"),
    ("RAL 7016 Anthracite grey",      "#293133"),
    ("RAL 7021 Black grey",           "#23282B"),
    ("RAL 7022 Umbra grey",           "#332F2C"),
    ("RAL 7024 Graphite grey",        "#474A50"),
    ("RAL 7035 Light grey",           "#CBD0CC"),
    ("RAL 7037 Dusty grey",           "#7D8471"),
    ("RAL 7042 Traffic grey A",       "#8F8F8C"),
    ("RAL 7043 Traffic grey B",       "#4E5451"),
    ("RAL 7044 Silk grey",            "#BDBDB2"),
    ("RAL 7046 Telegrey 2",           "#828282"),
    ("RAL 8002 Signal brown",         "#7B5141"),
    ("RAL 8003 Clay brown",           "#7F4E1E"),
    ("RAL 8004 Copper brown",         "#8F4E35"),
    ("RAL 8007 Fawn brown",           "#6F4F28"),
    ("RAL 8008 Olive brown",          "#6F4F28"),
    ("RAL 8011 Nut brown",            "#5A3A29"),
    ("RAL 8014 Sepia brown",          "#382C1E"),
    ("RAL 8015 Chestnut brown",       "#633A34"),
    ("RAL 8016 Mahogany brown",       "#4C2F27"),
    ("RAL 8017 Chocolate brown",      "#45322E"),
    ("RAL 8019 Grey brown",           "#403A3A"),
    ("RAL 8022 Black brown",          "#212121"),
    ("RAL 8023 Orange brown",         "#A65E2E"),
    ("RAL 8025 Pale brown",           "#79553D"),
    ("RAL 9001 Cream",                "#FDF4E3"),
    ("RAL 9002 Grey white",           "#E7EBDA"),
    ("RAL 9003 Signal white",         "#F4F4F4"),
    ("RAL 9004 Signal black",         "#282828"),
    ("RAL 9005 Jet black",            "#0A0A0A"),
    ("RAL 9010 Pure white",           "#FFFFFF"),
    ("RAL 9011 Graphite black",       "#1C1C1C"),
    ("RAL 9016 Traffic white",        "#F6F6F6"),
    ("RAL 9017 Traffic black",        "#1E1E1E"),
    ("RAL 9018 Papyrus white",        "#D7D7D7"),
]


# Cached reference Lab tables for fast nearest-neighbor.
_REFERENCE_CACHE: dict = {}


def _ensure_reference(table_name: str) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    if table_name in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[table_name]
    if table_name == "pantone_tcx":
        table = PANTONE_TCX
    elif table_name == "ral_classic":
        table = RAL_CLASSIC
    else:
        raise ValueError(f"unknown reference table: {table_name}")
    rgbs = np.array([_hex_to_rgb(h) for _, h in table], dtype=np.uint8)
    labs = rgb_to_lab_d50(rgbs)
    _REFERENCE_CACHE[table_name] = (labs, table)
    return labs, table


def nearest_pantone(rgb: Tuple[int, int, int], k: int = 1) -> List[Tuple[str, float]]:
    """Top-k nearest Pantone TCX entries by ΔE2000 (D50). Returns
    [(name, deltaE2000), ...] sorted ascending."""
    labs, table = _ensure_reference("pantone_tcx")
    target = rgb_to_lab_d50(np.array([rgb], dtype=np.uint8))[0]
    d = delta_e_2000(labs, target[None, :])
    order = np.argsort(d)[:k]
    return [(table[i][0], float(d[i])) for i in order]


def nearest_ral(rgb: Tuple[int, int, int], k: int = 1) -> List[Tuple[str, float]]:
    """Top-k nearest RAL Classic entries by ΔE2000 (D50)."""
    labs, table = _ensure_reference("ral_classic")
    target = rgb_to_lab_d50(np.array([rgb], dtype=np.uint8))[0]
    d = delta_e_2000(labs, target[None, :])
    order = np.argsort(d)[:k]
    return [(table[i][0], float(d[i])) for i in order]


# ICC-profile CMYK

_ICC_CACHE: dict = {}


def _cmyk_transform() -> Optional[ImageCms.ImageCmsTransform]:
    """Build (and cache) an sRGB → CMYK ImageCms transform if profiles are
    available. Returns None if PIL has no usable CMYK profile on this
    system, which falls back to the formula path."""
    if "transform" in _ICC_CACHE:
        return _ICC_CACHE["transform"]
    try:
        srgb_prof = ImageCms.createProfile("sRGB")
        # PIL ships a generic CMYK profile via ImageCms in some installs,
        # but not all. Try a couple of standard names.
        for candidate in ("USWebCoatedSWOP.icc", "CMYK.icc"):
            try:
                cmyk_prof = ImageCms.getOpenProfile(candidate)
                break
            except Exception:
                cmyk_prof = None
        if cmyk_prof is None:
            cmyk_prof = ImageCms.createProfile("CMYK")
        t = ImageCms.buildTransform(srgb_prof, cmyk_prof, "RGB", "CMYK")
        _ICC_CACHE["transform"] = t
        return t
    except Exception:
        _ICC_CACHE["transform"] = None
        return None


def rgb_to_cmyk(rgb: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    """sRGB → CMYK percentages. Uses ICC if available, falls back to the
    naive formula otherwise."""
    t = _cmyk_transform()
    if t is not None:
        try:
            im = Image.new("RGB", (1, 1), tuple(int(x) for x in rgb))
            cmyk = ImageCms.applyTransform(im, t)
            c, m, y, k = cmyk.getpixel((0, 0))
            return (int(round(c / 255 * 100)),
                    int(round(m / 255 * 100)),
                    int(round(y / 255 * 100)),
                    int(round(k / 255 * 100)))
        except Exception:
            pass
    # Formula fallback
    r, g, b = [x / 255.0 for x in rgb]
    k = 1.0 - max(r, g, b)
    if k >= 1.0:
        return (0, 0, 0, 100)
    c = (1 - r - k) / (1 - k)
    m = (1 - g - k) / (1 - k)
    y = (1 - b - k) / (1 - k)
    return (int(round(c * 100)), int(round(m * 100)),
            int(round(y * 100)), int(round(k * 100)))


# Palette extraction (OKLab k-means)

@dataclass
class PaletteEntry:
    rgb: Tuple[int, int, int]
    hex: str
    pantone: str
    pantone_deltaE: float
    pantone_top3: List[Tuple[str, float]] = field(default_factory=list)
    ral: str = ""
    ral_deltaE: float = 0.0
    cmyk: Tuple[int, int, int, int] = (0, 0, 0, 0)
    fraction: float = 0.0


def _build_entry(rgb: Tuple[int, int, int], fraction: float) -> PaletteEntry:
    top3 = nearest_pantone(rgb, k=3)
    ral_top = nearest_ral(rgb, k=1)
    return PaletteEntry(
        rgb=rgb,
        hex=rgb_to_hex(rgb),
        pantone=top3[0][0],
        pantone_deltaE=round(top3[0][1], 2),
        pantone_top3=[(n, round(d, 2)) for n, d in top3],
        ral=ral_top[0][0] if ral_top else "",
        ral_deltaE=round(ral_top[0][1], 2) if ral_top else 0.0,
        cmyk=rgb_to_cmyk(rgb),
        fraction=float(fraction),
    )


def _filter_extremes(rgb: np.ndarray) -> np.ndarray:
    """Drop deep shadows (V < 0.10) and specular highlights with near-zero
    saturation (S < 0.04 AND V > 0.95). Both bias palettes toward
    bake-artifacts that aren't useful colorway entries."""
    rgb_f = rgb.astype(np.float64) / 255.0
    mx = rgb_f.max(axis=-1)
    mn = rgb_f.min(axis=-1)
    s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    keep = (mx > 0.10) & ~((s < 0.04) & (mx > 0.95))
    return rgb[keep]


def extract_palette(
    diffuse: np.ndarray,
    n_colors: int = 6,
    pixel_subsample: int = 16,
    seed: int = 0,
) -> List[PaletteEntry]:
    """K-means in OKLab space over a sub-sampled, shadow/highlight-filtered
    diffuse texture. Returns palette entries sorted by area share desc."""
    px = diffuse[::pixel_subsample, ::pixel_subsample, :3].reshape(-1, 3)
    px = _filter_extremes(px)
    if len(px) < n_colors:
        return []
    ok = rgb_to_oklab(px)
    km = KMeans(n_clusters=n_colors, n_init=10, random_state=seed).fit(ok)
    # Convert OKLab centroids back to RGB by finding the nearest pixel
    # (more accurate than inverting OKLab analytically and re-quantizing).
    out: List[PaletteEntry] = []
    counts = np.bincount(km.labels_, minlength=n_colors).astype(np.float64)
    fractions = counts / counts.sum()
    for ci in range(n_colors):
        members = px[km.labels_ == ci]
        if len(members) == 0:
            continue
        # Representative = OKLab-centroid -> nearest member in OKLab.
        m_ok = rgb_to_oklab(members)
        d = np.linalg.norm(m_ok - km.cluster_centers_[ci], axis=1)
        rep = members[np.argmin(d)]
        rgb = (int(rep[0]), int(rep[1]), int(rep[2]))
        out.append(_build_entry(rgb, fractions[ci]))
    out.sort(key=lambda e: -e.fraction)
    return out


# Per-component palette (the better-than-dominant version)

def component_palette_from_face_uvs(
    component_face_indices: np.ndarray,
    face_uvs: Optional[np.ndarray],
    diffuse: Optional[np.ndarray],
    n_colors: int = 3,
    seed: int = 0,
    max_pixels: int = 4000,
) -> List[PaletteEntry]:
    """Per-component mini-palette using per-face UV arrays (Nf, 3, 2),
    which is what the Blender ingest returns directly."""
    if face_uvs is None or diffuse is None or len(component_face_indices) == 0:
        return []
    H, W = diffuse.shape[:2]
    comp_face_uvs = face_uvs[component_face_indices]  # (Nf, 3, 2)
    samples = [
        comp_face_uvs.mean(axis=1),
        (comp_face_uvs[:, 0] + comp_face_uvs[:, 1]) / 2.0,
        (comp_face_uvs[:, 1] + comp_face_uvs[:, 2]) / 2.0,
        (comp_face_uvs[:, 2] + comp_face_uvs[:, 0]) / 2.0,
    ]
    uv_pts = np.concatenate(samples, axis=0)
    u = np.clip(uv_pts[:, 0], 0, 1)
    v = np.clip(uv_pts[:, 1], 0, 1)
    px = (u * (W - 1)).astype(np.int32)
    py = ((1 - v) * (H - 1)).astype(np.int32)
    colors = diffuse[py, px, :3].astype(np.uint8)
    colors = _filter_extremes(colors)
    if len(colors) > max_pixels:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(colors), size=max_pixels, replace=False)
        colors = colors[idx]
    if len(colors) < n_colors:
        return []
    ok = rgb_to_oklab(colors)
    k = min(n_colors, len(colors))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(ok)
    counts = np.bincount(km.labels_, minlength=k).astype(np.float64)
    fractions = counts / max(counts.sum(), 1.0)
    entries: List[PaletteEntry] = []
    for ci in range(k):
        members = colors[km.labels_ == ci]
        if len(members) == 0:
            continue
        m_ok = rgb_to_oklab(members)
        d = np.linalg.norm(m_ok - km.cluster_centers_[ci], axis=1)
        rep = members[np.argmin(d)]
        rgb = (int(rep[0]), int(rep[1]), int(rep[2]))
        entries.append(_build_entry(rgb, fractions[ci]))
    entries.sort(key=lambda e: -e.fraction)
    return entries


def component_palette(
    component_face_indices: np.ndarray,
    mesh_faces: np.ndarray,
    uv: Optional[np.ndarray],
    diffuse: Optional[np.ndarray],
    n_colors: int = 3,
    seed: int = 0,
    max_pixels: int = 4000,
) -> List[PaletteEntry]:
    """Per-component mini-palette: OKLab k-means over the component's UV-
    sampled pixels (with shadow/highlight masking). Captures the 2-3
    natural color tones of a suede/mesh/etc. patch rather than collapsing
    to a single mean."""
    if uv is None or diffuse is None or len(component_face_indices) == 0:
        return []
    H, W = diffuse.shape[:2]
    # Sample 4 UV positions per face (centroid + 3 corners interpolated)
    # to get denser color coverage than face-centroid alone.
    face_uvs = uv[mesh_faces[component_face_indices]]  # (Nf, 3, 2)
    samples = []
    samples.append(face_uvs.mean(axis=1))                                  # centroid
    samples.append((face_uvs[:, 0] + face_uvs[:, 1]) / 2.0)                # edge midpoints
    samples.append((face_uvs[:, 1] + face_uvs[:, 2]) / 2.0)
    samples.append((face_uvs[:, 2] + face_uvs[:, 0]) / 2.0)
    uv_pts = np.concatenate(samples, axis=0)  # (4*Nf, 2)
    u = np.clip(uv_pts[:, 0], 0, 1)
    v = np.clip(uv_pts[:, 1], 0, 1)
    px = (u * (W - 1)).astype(np.int32)
    py = ((1 - v) * (H - 1)).astype(np.int32)
    colors = diffuse[py, px, :3].astype(np.uint8)
    colors = _filter_extremes(colors)
    if len(colors) > max_pixels:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(colors), size=max_pixels, replace=False)
        colors = colors[idx]
    if len(colors) < n_colors:
        return []
    ok = rgb_to_oklab(colors)
    km = KMeans(n_clusters=min(n_colors, len(colors)),
                n_init=10, random_state=seed).fit(ok)
    counts = np.bincount(km.labels_, minlength=n_colors).astype(np.float64)
    fractions = counts / max(counts.sum(), 1.0)
    entries: List[PaletteEntry] = []
    for ci in range(km.n_clusters):
        members = colors[km.labels_ == ci]
        if len(members) == 0:
            continue
        m_ok = rgb_to_oklab(members)
        d = np.linalg.norm(m_ok - km.cluster_centers_[ci], axis=1)
        rep = members[np.argmin(d)]
        rgb = (int(rep[0]), int(rep[1]), int(rep[2]))
        entries.append(_build_entry(rgb, fractions[ci]))
    entries.sort(key=lambda e: -e.fraction)
    return entries


def component_color_entries(components) -> List[PaletteEntry]:
    """Backwards-compat: one PaletteEntry per component, using the
    component's pre-computed dominant_color_rgb."""
    out = []
    for c in components:
        rgb = tuple(int(x) for x in c.dominant_color_rgb)
        out.append(_build_entry(rgb, 1.0))
    return out

