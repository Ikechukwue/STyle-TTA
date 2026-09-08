"""
Generate a draw.io (.drawio / mxGraph XML) version of the STyle-TTA
framework diagram, mirroring the layout of styl_tta_horizontal.svg.

Output:
    styl_tta_horizontal.drawio
"""

from pathlib import Path
from html import escape

OUTPUT_FILE = "styl_tta_horizontal.drawio"

# ============================================================
# Colors (same palette as the SVG script)
# ============================================================

TEXT = "#263238"
MUTED = "#607D8B"
BORDER = "#455A64"

BLUE_FILL, BLUE_BORDER, BLUE_TEXT = "#E3F2FD", "#1976D2", "#0D47A1"
GREEN_FILL, GREEN_BORDER, GREEN_TEXT = "#E8F5E9", "#388E3C", "#2E7D32"
PURPLE_FILL, PURPLE_BORDER, PURPLE_TEXT = "#F3E5F5", "#8E24AA", "#6A1B9A"
ORANGE_FILL, ORANGE_BORDER, ORANGE_TEXT = "#FFF3E0", "#FB8C00", "#E65100"
YELLOW_FILL, YELLOW_BORDER, YELLOW_TEXT = "#FFF8E1", "#F9A825", "#E65100"
GREY_FILL, GREY_BORDER, GREY_TEXT = "#F5F7FA", "#90A4AE", BORDER
RED_FILL, RED_BORDER, RED_TEXT = "#FDECEC", "#E57373", "#B71C1C"
ICON_FILL, ICON_BORDER = "#ECEFF1", BORDER
BAR_FILL, BAR_BORDER = "#ECEFF1", "#B0BEC5"

_id_counter = [0]


def next_id(prefix="id"):
    _id_counter[0] += 1
    return f"{prefix}{_id_counter[0]}"


def esc(s):
    return escape(s, quote=True)


# ============================================================
# Cell builders
# ============================================================

cells = []  # list of xml strings


def add_box(x, y, w, h, title, body_lines=None, fill="#FFFFFF",
            stroke=BORDER, font_color=TEXT, title_size=16,
            body_size=12, rounded=True, bold_title=True):
    """A rounded rectangle with a bold title line + smaller body lines."""
    body_lines = body_lines or []
    label_parts = [f"<b style='font-size:{title_size}px;'>{esc(title)}</b>"] if bold_title else [esc(title)]
    for line in body_lines:
        label_parts.append(f"<span style='font-size:{body_size}px;'>{esc(line)}</span>")
    label = "<br>".join(label_parts)
    label = esc(label)

    cid = next_id("box")
    style = (
        f"rounded={'1' if rounded else '0'};whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};fontColor={font_color};"
        f"align=center;verticalAlign=top;spacingTop=10;arcSize=8;"
        f"shadow=1;fontFamily=Helvetica;"
    )
    cells.append(f'''
        <mxCell id="{cid}" value="{label}" style="{style}" vertex="1" parent="1">
          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />
        </mxCell>''')
    return cid


def add_icon(x, y, w, h, label, fill=ICON_FILL, stroke=ICON_BORDER, font_size=13):
    """Small 'image' placeholder box used for x, s1, s2, T(x,s1), etc."""
    cid = next_id("icon")
    style = (
        f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
        f"fontColor={TEXT};fontStyle=1;fontSize={font_size};align=center;"
        f"verticalAlign=middle;arcSize=15;shape=mxgraph.basic.picture_frame;"
        f"sketch=0;"
    )
    # Fallback to a plain rounded rect (safer cross-version) instead of a
    # special shape, since custom shape libraries may not always be present:
    style = (
        f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
        f"fontColor={TEXT};fontStyle=1;fontSize={font_size};align=center;"
        f"verticalAlign=middle;arcSize=15;"
    )
    cells.append(f'''
        <mxCell id="{cid}" value="{esc(label)}" style="{style}" vertex="1" parent="1">
          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />
        </mxCell>''')
    return cid


def add_text(x, y, w, h, content, size=20, bold=False, color=TEXT, align="center"):
    cid = next_id("txt")
    style = (
        f"text;html=1;align={align};verticalAlign=middle;fontSize={size};"
        f"fontColor={color};fontStyle={'1' if bold else '0'};"
    )
    cells.append(f'''
        <mxCell id="{cid}" value="{esc(content)}" style="{style}" vertex="1" parent="1">
          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />
        </mxCell>''')
    return cid


def add_edge(points, stroke=BORDER, width=2):
    """points: list of (x, y) tuples. First = source, last = target,
    any in-between = waypoints (for elbow/multi-segment connectors)."""
    cid = next_id("edge")
    sx, sy = points[0]
    tx, ty = points[-1]
    mid = points[1:-1]

    waypoints_xml = ""
    if mid:
        pts = "".join(f'<mxPoint x="{px}" y="{py}" />' for px, py in mid)
        waypoints_xml = f'<Array as="points">{pts}</Array>'

    style = (
        f"endArrow=block;html=1;strokeColor={stroke};strokeWidth={width};"
        f"rounded=0;edgeStyle=orthogonalEdgeStyle;fontColor={TEXT};"
    )
    cells.append(f'''
        <mxCell id="{cid}" style="{style}" edge="1" parent="1">
          <mxGeometry relative="1" as="geometry">
            <mxPoint x="{sx}" y="{sy}" as="sourcePoint" />
            <mxPoint x="{tx}" y="{ty}" as="targetPoint" />
            {waypoints_xml}
          </mxGeometry>
        </mxCell>''')
    return cid


# ============================================================
# Title
# ============================================================

add_text(900, 20, 600, 40, "Style-Based Test-Time Augmentation (STyle-TTA)",
          size=26, bold=True)
add_text(750, 60, 900, 30,
          "Style retrieval \u2192 style transfer \u2192 frozen classification \u2192 probability aggregation",
          size=15, color=MUTED)

# ============================================================
# TARGET IMAGE
# ============================================================

add_box(60, 160, 270, 220, "TARGET IMAGE",
        body_lines=["Target-domain sample", "x \u2208 D\u1d40"],
        fill=BLUE_FILL, stroke=BLUE_BORDER, font_color=BLUE_TEXT)
add_icon(155, 225, 80, 60, "x")

# ============================================================
# DINOv3
# ============================================================

add_box(390, 195, 300, 150, "DINOv3",
        body_lines=["Pretrained feature extractor", "\u03c6(x) \u2192 \u211d\u1d48",
                     "classification head removed"],
        fill=BLUE_FILL, stroke=BLUE_BORDER, font_color=BLUE_TEXT)

# ============================================================
# SOURCE POOL
# ============================================================

add_box(390, 390, 300, 170, "SOURCE-DOMAIN STYLE POOL",
        body_lines=["S = {(x\u1d62\u02e2,y\u1d62\u02e2) \u2208 D\u02e2 | y\u1d62\u02e2 \u2208 C}"],
        fill=RED_FILL, stroke=RED_BORDER, font_color=RED_TEXT)
add_icon(415, 450, 60, 48, "s\u2081")
add_icon(485, 450, 60, 48, "s\u2082")
add_icon(555, 450, 60, 48, "\u2026")
add_icon(625, 450, 60, 48, "s\u2099")

# ============================================================
# L2 NORMALIZATION
# ============================================================

add_box(760, 190, 250, 160, "L2 NORMALIZATION",
        body_lines=["z = \u03c6(x) / \u2016\u03c6(x)\u2016\u2082",
                     "Normalized embeddings", "\u2192 cosine similarity"],
        fill=YELLOW_FILL, stroke=YELLOW_BORDER, font_color=YELLOW_TEXT)

# ============================================================
# FAISS
# ============================================================

add_box(1080, 175, 330, 190, "FAISS RETRIEVAL",
        body_lines=["Inner-product index", "Top-K nearest neighbors",
                     "S\u2096(x) = TopK(z\u2093 \u00b7 z\u209b)"],
        fill=GREEN_FILL, stroke=GREEN_BORDER, font_color=GREEN_TEXT)

# ============================================================
# TOP-K STYLE REFERENCES
# ============================================================

add_box(1460, 180, 360, 180, "TOP-K STYLE REFERENCES",
        fill=GREEN_FILL, stroke=GREEN_BORDER, font_color=GREEN_TEXT)
add_icon(1500, 255, 65, 50, "s\u2081")
add_icon(1600, 255, 65, 50, "s\u2082")
add_icon(1700, 255, 65, 50, "\u2026")
add_icon(1720, 255, 65, 50, "s\u2096")

# ============================================================
# STYLE TRANSFER
# ============================================================

add_box(1860, 165, 430, 210, "STYLE TRANSFER",
        body_lines=["For each retrieved reference s\u1d62:",
                     "x\u0303\u1d62 = T(x, s\u1d62)", "StyleID / AdaIN",
                     "Target = content \u2022 s\u1d62 = appearance"],
        fill=PURPLE_FILL, stroke=PURPLE_BORDER, font_color=PURPLE_TEXT)

# ============================================================
# STYLIZED VIEWS
# ============================================================

add_box(1460, 430, 830, 190, "K STYLE-CONDITIONED VIEWS",
        body_lines=["K views", "same semantic target"],
        fill="#FAF5FB", stroke=PURPLE_BORDER, font_color=PURPLE_TEXT)
add_icon(1515, 515, 85, 60, "T(x,s\u2081)")
add_icon(1670, 515, 85, 60, "T(x,s\u2082)")
add_icon(1825, 515, 85, 60, "\u2026")
add_icon(1980, 515, 85, 60, "T(x,s\u2096)")

# ============================================================
# ENSEMBLE OF K+1 VIEWS
# ============================================================

add_box(60, 710, 550, 220, "ENSEMBLE OF K+1 INPUT VIEWS",
        body_lines=["A(x) = {x, T(x,s\u2081), \u2026, T(x,s\u2096)}",
                     "Original target is explicitly retained"],
        fill=GREY_FILL, stroke=GREY_BORDER, font_color=GREY_TEXT)
add_icon(120, 795, 80, 60, "x")
add_icon(285, 795, 80, 60, "x\u0303\u2081")
add_icon(390, 795, 80, 60, "x\u0303\u2082")
add_text(475, 800, 40, 50, "\u2026", size=22, bold=True, color=MUTED)
add_text(530, 800, 50, 50, "x\u0303\u2096", size=17, bold=True, color=PURPLE_TEXT)

# ============================================================
# FROZEN CLASSIFIER
# ============================================================

add_box(680, 705, 350, 230, "FROZEN CLASSIFIER",
        body_lines=["f\u03b8", "Each image evaluated independently",
                     "p\u1d62 = softmax(f\u03b8(view\u1d62))"],
        fill=BLUE_FILL, stroke=BLUE_BORDER, font_color=BLUE_TEXT)

# ============================================================
# K+1 PREDICTION VECTORS
# ============================================================

add_box(1100, 705, 430, 230, "K+1 PREDICTION VECTORS",
        body_lines=["p\u2080 = p\u03b8(y | x)", "p\u2081, \u2026, p\u2096",
                     "Each vector \u2208 \u211d|C|",
                     "probabilities over shared label space C"],
        fill=ORANGE_FILL, stroke=ORANGE_BORDER, font_color=ORANGE_TEXT)

# ============================================================
# SOFT AGGREGATION
# ============================================================

add_box(1600, 700, 350, 240, "SOFT AGGREGATION",
        body_lines=["p\u0304(y|x) = 1/(K+1)", "\u00d7 [p\u2080 + \u03a3 p\u1d62]",
                     "Probability averaging",
                     "preserves confidence information"],
        fill=YELLOW_FILL, stroke=YELLOW_BORDER, font_color=YELLOW_TEXT)

# ============================================================
# FINAL DECISION
# ============================================================

add_box(2020, 705, 310, 230, "FINAL DECISION",
        body_lines=["\u0177 = argmax_c p\u0304(y=c|x)", "Predicted class"],
        fill=GREEN_FILL, stroke=GREEN_BORDER, font_color=GREEN_TEXT)

# ============================================================
# BOTTOM PRINCIPLES BAR
# ============================================================

add_box(560, 970, 1280, 48,
        "Inference only  \u2022  Frozen classifier  \u2022  No target labels  \u2022  "
        "No parameter updates  \u2022  Input-space adaptation",
        fill=BAR_FILL, stroke=BAR_BORDER, font_color=BORDER,
        title_size=14, bold_title=True, rounded=True)

# ============================================================
# EDGES
# ============================================================

add_edge([(330, 270), (390, 270)])                     # Target -> DINOv3
add_edge([(540, 345), (540, 390)], stroke="#E57373")   # DINOv3 -> Source pool
add_edge([(690, 270), (760, 270)])                     # DINOv3 -> L2 Norm
add_edge([(1010, 270), (1080, 270)])                   # L2 Norm -> FAISS
add_edge([(1410, 270), (1460, 270)])                   # FAISS -> Top-K refs
add_edge([(1820, 270), (1860, 270)])                   # Top-K refs -> Style transfer
add_edge([(2075, 375), (2075, 430), (1875, 430)],
          stroke=PURPLE_BORDER)                        # Style transfer -> Stylized views
add_edge([(1600, 620), (1600, 670), (335, 670), (335, 710)],
          stroke=PURPLE_BORDER)                        # Stylized views -> Ensemble
add_edge([(335, 660), (335, 710)])                     # Original -> Ensemble
add_edge([(610, 820), (680, 820)])                     # Ensemble -> Classifier
add_edge([(1030, 820), (1100, 820)])                   # Classifier -> Prediction vectors
add_edge([(1530, 820), (1600, 820)])                   # Prediction vectors -> Soft agg
add_edge([(1950, 820), (2020, 820)])                   # Soft agg -> Final decision

# ============================================================
# Assemble full mxfile document
# ============================================================

body = "\n".join(cells)

xml = f'''<mxfile host="app.diagrams.net" agent="python-generator" version="24.0.0">
  <diagram name="STyle-TTA" id="STyle-TTA-diagram">
    <mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" tooltips="1"
        connect="1" arrows="1" fold="1" page="1" pageScale="1"
        pageWidth="2400" pageHeight="1050" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
{body}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
'''

output_path = Path(OUTPUT_FILE)
output_path.write_text(xml, encoding="utf-8")
print(f"draw.io XML generated successfully:")
print(output_path.resolve())
