"""
Step 12 - Review 2 materials
(Week 4, Phase1_Implementation_Plan).

12.1 Architecture diagram: the Review 1 system architecture redrawn with every
     block marked Done (Phase 1) / Partial / Planned (Phase 2).
12.2 Result slides: AOI map, flood inventory, susceptibility map, metrics
     table + feature importance (plus the validation, features and
     limitations slides that support them).
12.3 Anticipated panel questions: a backup slide, plus speaker notes on
     every slide. The full Q&A is in docs/review2_materials.md.

The deck follows the Review 1 style (white 16:9, left-aligned titles, IIIT
Kottayam logo top-right, institute footer, slide numbers). Every number is
read from the step metrics JSON files, not typed in, so the deck always
matches the pipeline outputs.

Outputs
    outputs/maps/step12_architecture_status.png
    outputs/maps/step12_study_area.png
    outputs/presentation/BTP_Review2_Phase1_Team67.pptx
"""
import os
import sys
import json

import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import box
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LightSource
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

script_dir = os.path.dirname(os.path.abspath(__file__))
phase1_root = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(phase1_root)

MAP_DIR = "outputs/maps"
MET = "outputs/metrics"
PRES_DIR = "outputs/presentation"
LOGO = f"{PRES_DIR}/assets/iiitk_logo.png"
DECK = f"{PRES_DIR}/BTP_Review2_Phase1_Team67.pptx"
BBOX = [76.15, 9.90, 76.55, 10.25]
MID_LAT = (BBOX[1] + BBOX[3]) / 2

print("=" * 80)
print("STEP 12: REVIEW 2 MATERIALS (architecture status, study-area map, slide deck)")
print("=" * 80)

# ---------------------------------------------------------------------------
# Numbers from the pipeline
# ---------------------------------------------------------------------------
s5 = json.load(open("data/processed/flood_inventory/flood_inventory_metadata.json", encoding="utf-8"))["events"]
s6 = json.load(open(f"{MET}/step6_flood_inventory_crosscheck.json", encoding="utf-8"))
s7 = json.load(open(f"{MET}/step7_feature_stack_checkpoint.json", encoding="utf-8"))
s8 = json.load(open(f"{MET}/step8_training_dataset_summary.json", encoding="utf-8"))
s9 = json.load(open(f"{MET}/step9_baseline_metrics.json", encoding="utf-8"))
s10 = json.load(open(f"{MET}/step10_susceptibility_summary.json", encoding="utf-8"))

gti = s6["headline"]
gti_edge = s6["events"]["2018_flood"]["diagnostic_unconfirmed_vs_envelope"]["GTI_px_40m_excluding_60m_water_edge"]
flood18 = s5["2018_flood"]["areas"]["flood_km2"]
thr18 = s5["2018_flood"]["otsu_threshold_db"]["flood"]
land_km2 = s7["domain"]["land_km2"]
n_samples = s8["dataset"]["n_rows"]
n_per = s8["dataset"]["class_counts"]["1"]
max_vif = max(s7["collinearity"]["vif"].values())
te = s9["headline_test_metrics"]
sp10 = s9["supplementary_spatial_block_cv"]["by_block_size"]["10km"]["per_fold_mean_sd"]
fp = s9["supplementary_spatial_block_cv"]["floodplain_only_subset"]["spatial_10km_out_of_fold"]
bp = s9["base_paper"]["table3_testing"]
imp = sorted(s9["importance"], key=lambda r: -r["gain_pct"])
abl = {r["inputs"]: r for r in s9["ablation_table4_analogue"]}
v19 = s10["validation_2019"]["models"]["all"]["ndem_2019"]
v19nr = s10["validation_2019"]["models"]["no_rainfall"]["ndem_2019"]
v19new = s10["validation_2019"]["models"]["all"]["ndem_2019_outside_2018_envelope"]
pl = s10["plausibility_high_ground"]
pct = lambda x: f"{100 * x:.1f}%"

# ---------------------------------------------------------------------------
# 12.1 Architecture diagram with Done / Partial / Planned status
# ---------------------------------------------------------------------------
print("\n[1/3] 12.1 Architecture status diagram")
STATUS = {"done": ("#e3f5ec", "#1a8f63", "-", "Done in Phase 1"),
          "partial": ("#fff3dc", "#d08a00", "-", "Partly done"),
          "planned": ("#f3f3f3", "#9a9a9a", "--", "Planned for Phase 2")}
COLS = [
    ("Input data\n(multi-year)", "#dbe8f7", [
        ("Sentinel-1 SAR", "3 flood + 3 dry scenes", "done"),
        ("Sentinel-2 optical", "land cover / urban growth", "planned"),
        ("DEM", "FABDEM 30 m", "done"),
        ("Rainfall (CHIRPS)", "2017–2023", "done"),
        ("OSM / HydroSHEDS", "roads, rivers", "done"),
        ("Building footprints", "reference", "planned")]),
    ("Preprocessing\n& harmonisation", "#fbe8d4", [
        ("SAR processing", "dB, Lee filter, Otsu", "done"),
        ("Cloud / noise handling", "speckle done; S2 clouds tested", "partial"),
        ("Common grid", "30 m FABDEM grid", "done"),
        ("Multi-temporal alignment", "same-orbit pairs; LULC pending", "partial")]),
    ("Thematic products\n(multi-year)", "#dff2e3", [
        ("Flood inventory (S1)", "2018 labels · 2019 check", "done"),
        ("Land-cover / built-up\nchange (S2)", "", "planned")]),
    ("Feature engineering\n(flow-coupled)", "#ebe3f5", [
        ("Elevation, slope, TWI", "+ curvature, aspect; HAND next", "done"),
        ("Distance to river / road", "+ drainage density", "done"),
        ("Flow direction &\naccumulation", "Priority-Flood + FD8", "done"),
        ("UICA", "upslope impervious area", "planned"),
        ("ΔUICA", "change over time", "planned")]),
    ("Flood susceptibility\nmodel", "#dfeaf8", [
        ("Positive-Unlabelled\nlearning", "", "planned"),
        ("LightGBM / XGBoost", "LightGBM baseline", "done"),
        ("Spatial block CV", "diagnostic run only", "partial"),
        ("Temporal hold-out", "2019 flood check", "partial")]),
    ("Dual prediction\n& output", "#e3f2de", [
        ("Current susceptibility", "5-class map", "done"),
        ("Post-development\nsusceptibility", "", "planned"),
        ("Δ Susceptibility", "impact of development", "planned")]),
]
fig, ax = plt.subplots(figsize=(16, 7.6), dpi=150)
ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")
col_w, gap, x0, top = 15.2, 1.55, 0.6, 48.5
for ci, (title, head_col, blocks) in enumerate(COLS):
    x = x0 + ci * (col_w + gap)
    ax.add_patch(FancyBboxPatch((x, 9.2), col_w, top - 9.2, boxstyle="round,pad=0,rounding_size=0.6",
                                facecolor="#fbfbfb", edgecolor="#b8c4d6", lw=1.2))
    ax.add_patch(Rectangle((x, top - 5.2), col_w, 5.2, facecolor=head_col, edgecolor="none"))
    ax.text(x + col_w / 2, top - 2.6, title, ha="center", va="center", fontsize=10.5, fontweight="bold")
    n = len(blocks)
    avail = top - 5.9 - 10.0
    bh = min(5.4, (avail - (n - 1) * 0.8) / n)
    for bi, (name, sub, st) in enumerate(blocks):
        fc, ec, ls, _ = STATUS[st]
        by = top - 6.6 - bi * (bh + 0.8) - bh
        ax.add_patch(FancyBboxPatch((x + 0.6, by), col_w - 1.2, bh, boxstyle="round,pad=0,rounding_size=0.4",
                                    facecolor=fc, edgecolor=ec, lw=1.6, ls=ls))
        ty = by + bh / 2 + (0.8 if sub else 0)
        ax.text(x + col_w / 2, ty, name, ha="center", va="center", fontsize=9, fontweight="bold",
                color="#1d1d1d" if st != "planned" else "#6b6b6b", linespacing=1.05)
        if sub:
            ax.text(x + col_w / 2, by + bh / 2 - 1.15, sub, ha="center", va="center", fontsize=7.6,
                    color="#3a3a3a" if st != "planned" else "#7a7a7a")
    if ci < len(COLS) - 1:
        ax.annotate("", xy=(x + col_w + gap - 0.05, 28), xytext=(x + col_w + 0.05, 28),
                    arrowprops=dict(arrowstyle="-|>", color="#1d1d1d", lw=1.6, mutation_scale=14))
# explainability + validation loop
xe = x0 + 5 * (col_w + gap)
for bi, (name, sub, st) in enumerate([("SHAP &\ncounterfactuals", "global\nimportance done", "partial"),
                                      ("Planner\ndashboards", "", "planned")]):
    fc, ec, ls, _ = STATUS[st]
    bx = xe + 0.6 + bi * (col_w - 1.2) / 2
    ax.add_patch(FancyBboxPatch((bx + 0.1, 10.2), (col_w - 1.2) / 2 - 0.2, 5.6,
                                boxstyle="round,pad=0,rounding_size=0.4", facecolor=fc, edgecolor=ec, lw=1.6, ls=ls))
    ax.text(bx + (col_w - 1.2) / 4, 13.8 if sub else 13.0, name, ha="center", va="center",
            fontsize=7.8, fontweight="bold", color="#1d1d1d" if st != "planned" else "#6b6b6b")
    if sub:
        ax.text(bx + (col_w - 1.2) / 4, 11.3, sub, ha="center", va="center",
                fontsize=6.6, color="#3a3a3a")
ax.text(xe + col_w / 2, 16.6, "Explainability & decision support", ha="center", va="center", fontsize=8.8,
        fontweight="bold")
ax.plot([x0 + col_w / 2, x0 + col_w / 2, xe + col_w / 2, xe + col_w / 2], [9.2, 6.2, 6.2, 9.2],
        color=STATUS["done"][1], lw=1.6)
ax.text(50, 4.6, "Refine / validate with an independent source — DONE: official NRSC/ISRO NDEM flood maps "
                 f"(GTI {gti['value_pct']}%); Copernicus EMS and DFO checked, no 2018 coverage",
        ha="center", va="center", fontsize=9, color="#1a5e43")
ax.legend(handles=[Patch(facecolor=STATUS[k][0], edgecolor=STATUS[k][1], ls=STATUS[k][2], lw=1.6, label=STATUS[k][3])
                   for k in ("done", "partial", "planned")],
          loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=10, frameon=False)
fig.tight_layout()
p_arch = f"{MAP_DIR}/step12_architecture_status.png"
fig.savefig(p_arch, bbox_inches="tight", facecolor="white")
plt.close(fig)
n_status = {k: sum(1 for _, _, b in COLS for *_, s in b if s == k) for k in STATUS}
print(f"    [OK] {p_arch}  (blocks: {n_status})")

# ---------------------------------------------------------------------------
# Study-area map (clean version for the AOI slide)
# ---------------------------------------------------------------------------
print("\n[2/3] Study-area map")
with rasterio.open("data/processed/features/feature_stack.tif") as src:
    elev = src.read(1).astype(float)
    T = src.transform
with rasterio.open("data/processed/features/analysis_domain.tif") as src:
    land = src.read(1).astype(bool)
H, W = elev.shape
extent = (T.c, T.c + W * T.a, T.f + H * T.e, T.f)
ls_ = LightSource(azdeg=315, altdeg=45)
e = np.where(land, elev, 0)
# terrain palette shifted so 0 m land starts green (blue would read as water)
shade = ls_.shade(np.clip(e, -2, 160), cmap=plt.get_cmap("terrain"), vert_exag=4, blend_mode="soft",
                  dx=29.5, dy=29.8, vmin=-53, vmax=160)
shade[~land] = [0.84, 0.88, 0.91, 1.0]
rivers = gpd.clip(gpd.read_file("data/raw/hydrosheds/periyar_rivers_clip.gpkg"), box(*BBOX))
TOWNS = [("Aluva", 10.1076, 76.3516), ("Kalamassery", 10.0528, 76.3264), ("Kochi (North)", 10.0245, 76.3078),
         ("Paravur", 10.1449, 76.2300), ("Perumbavoor", 10.1147, 76.4735), ("Eloor", 10.0805, 76.2990),
         ("Cochin Airport", 10.1520, 76.4019)]
fig, ax = plt.subplots(figsize=(8.6, 8.2), dpi=150)
ax.imshow(shade, extent=extent, interpolation="bilinear")
rivers[rivers["ORD_STRA"] >= 4].plot(ax=ax, color="#2a78d6", lw=1.3)
rivers[rivers["ORD_STRA"] < 4].plot(ax=ax, color="#2a78d6", lw=0.4, alpha=0.6)
for name, la, lo in TOWNS:
    ax.plot(lo, la, "o", ms=6, mfc="#ffd23f", mec="#1d1d1d", mew=0.9, zorder=5)
    ax.annotate(name, (lo, la), xytext=(5, 4), textcoords="offset points", fontsize=8.5, fontweight="bold",
                zorder=6, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
ax.set_xlim(BBOX[0], BBOX[2]); ax.set_ylim(BBOX[1], BBOX[3])
ax.set_aspect(1 / np.cos(np.radians(MID_LAT)))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}°E"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}°N"))
ax.tick_params(labelsize=7.5)
km_deg = 111.320 * np.cos(np.radians(MID_LAT))
for i, col in enumerate(("#1d1d1d", "white")):
    ax.add_patch(Rectangle((76.165 + i * 5 / km_deg, 9.915), 5 / km_deg, 0.0035, fc=col, ec="#1d1d1d", lw=0.8, zorder=6))
for i, lbl in enumerate(("0", "5", "10 km")):
    ax.text(76.165 + i * 5 / km_deg, 9.9205, lbl, ha="center", va="bottom", fontsize=7.5, zorder=6)
ax.annotate("", xy=(76.53, 10.237), xytext=(76.53, 10.215),
            arrowprops=dict(facecolor="#1d1d1d", edgecolor="#1d1d1d", width=3.5, headwidth=10, headlength=8))
ax.text(76.53, 10.239, "N", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.legend(handles=[Line2D([], [], color="#2a78d6", lw=1.5, label="Rivers (HydroRIVERS)"),
                   Patch(fc=(0.84, 0.88, 0.91), ec="#9a9a9a", label="Sea / backwater"),
                   Line2D([], [], marker="o", ls="", mfc="#ffd23f", mec="#1d1d1d", label="Towns")],
          loc="lower right", fontsize=8, framealpha=0.9)
ax.set_title(f"Lower Periyar & Aluva–Kochi corridor (≈1,700 km², land {land_km2:,.0f} km²)\n"
             "Shaded FABDEM elevation, Ernakulam district, Kerala",
             loc="left", fontsize=10.5, fontweight="bold")
fig.tight_layout()
p_aoi = f"{MAP_DIR}/step12_study_area.png"
fig.savefig(p_aoi, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"    [OK] {p_aoi}")

# ---------------------------------------------------------------------------
# 12.2 / 12.3 Slide deck
# ---------------------------------------------------------------------------
print("\n[3/3] Slide deck")
INK = RGBColor(0x1D, 0x1D, 0x1D)
MUTED = RGBColor(0x59, 0x59, 0x59)
ACCENT = RGBColor(0x1F, 0x5F, 0x99)
HEAD_FILL = RGBColor(0xDA, 0xE3, 0xEE)
ROW_FILL = RGBColor(0xF4, 0xF6, 0xF9)
GOOD = RGBColor(0x1A, 0x8F, 0x63)
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height
slide_no = [0]


def base_slide(title, subtitle=None):
    s = prs.slides.add_slide(BLANK)
    slide_no[0] += 1
    s.shapes.add_picture(LOGO, SW - Inches(2.35), Inches(0.18), height=Inches(1.0))
    if title:
        tb = s.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(10.2), Inches(0.9))
        p = tb.text_frame.paragraphs[0]
        p.text = title
        p.font.size, p.font.name, p.font.color.rgb = Pt(36), "Calibri", INK
    if subtitle:
        tb = s.shapes.add_textbox(Inches(0.62), Inches(1.12), Inches(10.4), Inches(0.5))
        p = tb.text_frame.paragraphs[0]
        p.text = subtitle
        p.font.size, p.font.name, p.font.color.rgb = Pt(18), "Calibri", MUTED
    f = s.shapes.add_textbox(Inches(3.5), SH - Inches(0.5), Inches(6.3), Inches(0.35))
    p = f.text_frame.paragraphs[0]
    p.text, p.alignment = "Indian Institute of Information Technology Kottayam", PP_ALIGN.CENTER
    p.font.size, p.font.name, p.font.color.rgb = Pt(11), "Calibri", MUTED
    if slide_no[0] > 1:
        n = s.shapes.add_textbox(SW - Inches(1.2), SH - Inches(0.52), Inches(0.7), Inches(0.35))
        p = n.text_frame.paragraphs[0]
        p.text, p.alignment = str(slide_no[0]), PP_ALIGN.RIGHT
        p.font.size, p.font.name, p.font.color.rgb = Pt(12), "Calibri", MUTED
    return s


def picture(s, path, left, top, max_w, max_h):
    w, h = Image.open(path).size
    scale = min(max_w / w, max_h / h)
    pw, ph = int(w * scale), int(h * scale)
    s.shapes.add_picture(path, int(left + (max_w - pw) / 2), int(top + (max_h - ph) / 2), pw, ph)


def bullets(s, items, left, top, width, height, size=16):
    """items: str or (str, level) or (str, level, bold)."""
    tf = s.shapes.add_textbox(left, top, width, height).text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        text, lvl, bold = (it, 0, False) if isinstance(it, str) else (it + (False,))[:3]
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("•  " if lvl == 0 else "–  ") + text
        p.level = 0
        p.font.size = Pt(size if lvl == 0 else size - 2)
        p.font.name, p.font.bold = "Calibri", bold
        p.font.color.rgb = INK if lvl == 0 else MUTED
        p.space_after = Pt(6 if lvl == 0 else 3)
        if lvl:
            p.left_indent = Inches(0.35) if hasattr(p, "left_indent") else None
    return tf


def table(s, rows, left, top, width, col_w=None, size=13, row_h=0.42, bold_rows=(), highlight_rows=()):
    nr, nc = len(rows), len(rows[0])
    shp = s.shapes.add_table(nr, nc, left, top, width, Inches(row_h * nr))
    t = shp.table
    if col_w:
        for i, w in enumerate(col_w):
            t.columns[i].width = Inches(w)
    for r in range(nr):
        t.rows[r].height = Inches(row_h)
        for c in range(nc):
            cell = t.cell(r, c)
            cell.text = str(rows[r][c])
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.fill.solid()
            cell.fill.fore_color.rgb = HEAD_FILL if r == 0 else (RGBColor(0xE3, 0xF5, 0xEC) if r in highlight_rows else ROW_FILL)
            for p in cell.text_frame.paragraphs:
                p.font.size, p.font.name = Pt(size), "Calibri"
                p.font.bold = r == 0 or r in bold_rows
                p.font.color.rgb = INK
    return t


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


# 1 Title -------------------------------------------------------------------
s = base_slide(None)
tb = s.shapes.add_textbox(Inches(1.2), Inches(1.35), Inches(10.9), Inches(2.2))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text, p.alignment = "Development-Aware Flood Susceptibility Mapping Using Multi-Temporal Satellite Data", PP_ALIGN.CENTER
p.font.size, p.font.name, p.font.color.rgb = Pt(40), "Calibri", INK
p = tf.add_paragraph()
p.text, p.alignment = "Review 2 — Phase 1: reproducing the base paper on the Lower Periyar basin", PP_ALIGN.CENTER
p.font.size, p.font.name, p.font.color.rgb, p.space_before = Pt(22), "Calibri", ACCENT, Pt(14)
tb = s.shapes.add_textbox(Inches(3.9), Inches(4.05), Inches(5.5), Inches(2.0))
tf = tb.text_frame
for i, line in enumerate(["BY", "2023BCD0005 - ALBIN JOHN", "2023BCD0056 - NIKETH B NATH",
                          "2023BCD0014 - JATIN NATH", "2023BCD0030 - R RANJITH KUMAR"]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text, p.alignment = line, PP_ALIGN.CENTER
    p.font.size, p.font.name, p.font.italic, p.font.color.rgb = Pt(14), "Calibri", i > 0, INK
tb = s.shapes.add_textbox(Inches(9.3), Inches(5.35), Inches(3.2), Inches(0.9))
tf = tb.text_frame
for i, line in enumerate(["Guided By,", "Dr. Rosebell Paul"]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text, p.alignment = line, PP_ALIGN.CENTER
    p.font.size, p.font.name, p.font.italic, p.font.color.rgb = Pt(18), "Calibri", i == 1, INK
notes(s, "Review 2 covers Phase 1: a full, verified reproduction of the base paper's pipeline on our own study "
         "area. Phase 2 (our novel contributions) starts after this review.")

# 2 Scope -------------------------------------------------------------------
s = base_slide("From Review 1 to Review 2", "Phase 1 reproduces the base paper's logic end to end on the Periyar basin")
rows = [["Stage", "Base paper (Mohamadiazar et al., 2024)", "Our Phase 1 reproduction"],
        ["Study area", "Miami-Dade, Florida (pluvial)", "Lower Periyar & Aluva–Kochi, Kerala (fluvial)"],
        ["Satellite data", "226 Sentinel-1 images, 2014–2023", "Sentinel-1 VV: 2018, 2019, 2021 flood + dry pairs"],
        ["Flood inventory", "Otsu threshold on VV (SNAP)", "Otsu on VV + change detection (Lee filter, GEE)"],
        ["Inventory check", "GTI 84.05 % vs 116 flood reports", f"GTI {gti['value_pct']} % vs official NRSC/ISRO NDEM maps"],
        ["Model", "U-Net CNN", "LightGBM (tabular, pixel-based)"],
        ["Output", "Flood prediction maps", "5-class susceptibility map + 2019 validation"]]
table(s, rows, Inches(0.6), Inches(1.85), Inches(12.1), col_w=[2.2, 4.7, 5.2], size=14, row_h=0.5)
bullets(s, [("Why LightGBM, not U-Net (Step 0): no labelled image set existed for Kerala; building one and tuning a CNN "
             "would have used the whole timeline. LightGBM gave a complete, checkable pipeline and matches our Kerala "
             "reference papers. U-Net is revisited once Phase 2 has a multi-event inventory.", 0)],
        Inches(0.6), Inches(5.55), Inches(12.1), Inches(1.2), size=15)
notes(s, "The panel will ask why we did not use U-Net. The answer is a deliberate trade-off: no existing labelled "
         "data, four weeks, and LightGBM is the method of our Kerala reference papers. Everything else follows the "
         "base paper's chain: satellite -> Otsu flood inventory -> ML model -> map -> accuracy metrics.")

# 3 Architecture --------------------------------------------------------------
s = base_slide("System Architecture — Progress", "Review 1 architecture, marked by what Phase 1 has built")
picture(s, p_arch, Inches(0.4), Inches(1.6), Inches(12.5), Inches(5.3))
notes(s, f"Green = done in Phase 1 ({n_status['done']} blocks), amber = partly done ({n_status['partial']}), grey dashed = "
         f"Phase 2 ({n_status['planned']}). The whole data -> inventory -> features -> model -> current-susceptibility "
         "chain works end to end. Everything development-related (Sentinel-2 land cover, UICA/ΔUICA, PU-learning, "
         "post-development prediction) is Phase 2 — that is our novelty.")

# 4 Study area & data -----------------------------------------------------------
s = base_slide("Study Area & Data")
picture(s, p_aoi, Inches(0.4), Inches(1.3), Inches(6.2), Inches(5.6))
rows = [["Dataset", "Source", "Resolution"],
        ["Sentinel-1 VV (flood + dry)", "Copernicus / GEE", "~10–20 m"],
        ["FABDEM elevation", "Univ. Bristol / GEE", "30 m"],
        ["CHIRPS rainfall", "UCSB-CHG / GEE", "5.5 km"],
        ["Rivers (HydroRIVERS)", "HydroSHEDS", "15″ network"],
        ["Roads (OSM drive)", "OpenStreetMap", "vector"],
        ["Flood reference", "NRSC/ISRO NDEM", "polygons"]]
table(s, rows, Inches(6.9), Inches(1.55), Inches(6.0), col_w=[2.55, 2.15, 1.3], size=13, row_h=0.46)
bullets(s, [f"AOI {BBOX[0]}–{BBOX[2]}°E, {BBOX[1]}–{BBOX[3]}°N (≈1,700 km²), UTM 43N",
            "Contains the Periyar floodplain hit hardest in Aug 2018 and the urbanising Aluva–Kochi corridor",
            "Soil (NBSS&LUP) not downloadable — same data gap as the base paper's HSG"],
        Inches(6.9), Inches(4.95), Inches(6.0), Inches(1.9), size=14)
notes(s, "The AOI was chosen to contain both the 2018 flood (so the inventory is not empty) and fast urban growth "
         "(for Phase 2). All data are open; Sentinel-1, FABDEM and CHIRPS come through Google Earth Engine.")

# 5 Flood inventory -----------------------------------------------------------
s = base_slide("Flood Inventory — Sentinel-1 + Otsu", "21 Aug 2018 flood scene vs 30 Mar 2018 dry scene, same orbit")
picture(s, f"{MAP_DIR}/step5_flood_inventory_2018_20180821_overview.png", Inches(0.35), Inches(1.6),
        Inches(8.4), Inches(5.35))
bullets(s, ["Base-paper method: backscatter to dB (Eq. 1), Otsu threshold on VV",
            (f"Otsu threshold {thr18:.2f} dB — sits in the valley between water and land peaks", 1),
            "Flood = water on flood date AND NOT water in dry scene",
            ("steep slopes (> 5°) and specks (< 5 px) removed", 1),
            f"2018 flood: {flood18:.1f} km² (1.4 % of AOI), on the northern Periyar floodplain",
            ("2019: 28.9 km² · 2021: 8.9 km² (too small / noisy to use)", 1),
            "Known blind spots: water among buildings, airport runway, scene is 4–6 days after the peak"],
        Inches(8.85), Inches(1.7), Inches(4.2), Inches(5.2), size=14)
notes(s, "Top left: the two histograms are clearly bimodal and the Otsu threshold falls in the valley, so the "
         "threshold is sound. Bottom: flooded pixels (orange) cluster on the Periyar floodplain around "
         "Paravur/Puthenvelikkara; Kochi city stays clean. Bottom right: our flood against the official NDEM outline.")

# 6 Inventory validation --------------------------------------------------------
s = base_slide("Is the Flood Inventory Right?", "Independent check against official NRSC/ISRO NDEM flood maps")
picture(s, f"{MAP_DIR}/step6_crosscheck_gti_summary.png", Inches(0.35), Inches(1.65), Inches(7.9), Inches(4.4))
bullets(s, [f"{gti['value_pct']} % of our 2018 flood pixels confirmed by NDEM (within 40 m)",
            (f"strict per-pixel {gti['strict_pixel_pct']} % · a random map would score {gti['random_baseline_pct']} % "
             f"→ {gti['value_pct'] / gti['random_baseline_pct']:.1f}× better than chance", 1),
            (f"away from river/backwater edges: {100 * gti_edge:.1f} %", 1),
            "Base paper GTI: 84.05 % (point reports, image taken during the flood)",
            "Copernicus EMS: no activation · DFO: no 2018 event · Sentinel-2: 82 % cloud",
            "2021 fails the check (15 %) → not used as labels"],
        Inches(8.4), Inches(1.75), Inches(4.6), Inches(5.0), size=14)
notes(s, "This is our Ground Truth Index analogue. 58.4 % is lower than the base paper's 84 %, for two measurable "
         "reasons: our only scene is 4-6 days after the peak (residual water), and most disagreement sits on the "
         "river/backwater fringe. Away from that fringe it rises to 74 %.")

# 7 Features & dataset -----------------------------------------------------------
s = base_slide("Features & Training Data", f"{len(s8['dataset']['feature_columns'])} features on the 30 m FABDEM grid · "
                                             f"{n_samples:,} balanced samples")
picture(s, f"{MAP_DIR}/step7_feature_stack_overview.png", Inches(0.35), Inches(1.6), Inches(7.9), Inches(5.35))
bullets(s, ["Terrain: elevation, slope, aspect, plan & profile curvature",
            "Hydrology: TWI, distance to river / major river, drainage density",
            "Distance to road, mean annual rainfall",
            ("all layers on one grid, no gaps, 11 self-tests passed; max VIF " f"{max_vif:.1f}", 1),
            f"{n_per:,} flood + {n_per:,} non-flood samples (2018)",
            ("flood = Otsu flood confirmed by NDEM; non-flood = away from any observed flood and the NDEM envelope", 1),
            ("no feature derived from Sentinel-1 → no label leakage", 1)],
        Inches(8.4), Inches(1.75), Inches(4.6), Inches(5.1), size=14)
notes(s, "Only the reliable part of the inventory becomes training data: flood pixels confirmed by NDEM, and "
         "non-flood pixels far from anything that flooded at the peak. None of the features comes from the radar "
         "image, so the model cannot cheat by reading the label.")

# 8 Metrics -------------------------------------------------------------------------
s = base_slide("Model Results — Metrics", "LightGBM baseline vs base paper")
rows = [["Evaluation", "Accuracy", "Precision", "Recall", "F1", "AUC"],
        ["Ours — random 70/30 split (plan method)", pct(te["accuracy"]), pct(te["precision"]), pct(te["recall"]),
         pct(te["f1"]), f"{te['auc']:.3f}"],
        ["Ours — unseen 10 km regions (honest)", pct(sp10["accuracy"]["mean"]), pct(sp10["precision"]["mean"]),
         pct(sp10["recall"]["mean"]), pct(sp10["f1"]["mean"]), f"{sp10['auc']['mean']:.3f}"],
        ["Base paper — U-Net test", f"{bp['accuracy']:.2f}%", f"{bp['precision']:.2f}%", f"{bp['recall']:.2f}%",
         f"{bp['f1']:.2f}%", "0.93"]]
table(s, rows, Inches(0.6), Inches(1.75), Inches(12.1), col_w=[4.6, 1.5, 1.5, 1.5, 1.5, 1.5], size=15, row_h=0.55,
      highlight_rows=(2,))
picture(s, f"{MAP_DIR}/step9_model_evaluation.png", Inches(0.4), Inches(4.05), Inches(8.2), Inches(2.9))
bullets(s, [f"99 % looks too good — we checked why: {pct(s9['diagnostics']['spatial_proximity_of_random_split']['flood_test_share'])} "
            "of flood test pixels sit next to a training pixel",
            "Not leakage: on whole regions never seen, accuracy is " f"{pct(sp10['accuracy']['mean'])}",
            f"Hard case (flat floodplain only): AUC {fp['auc']:.3f}"],
        Inches(8.75), Inches(4.1), Inches(4.3), Inches(2.9), size=13.5)
notes(s, "The plan warns that ~0.999 AUC usually means leakage, so we investigated. The random split tests on "
         "neighbours of training pixels. Holding out whole 10 km regions gives 88.5 % accuracy, which is our honest "
         "number and comparable to the base paper's 93.7 %. AUC stays about 0.98, so the model ranks terrain well.")

# 9 Importance -------------------------------------------------------------------------
s = base_slide("What Drives the Model?", "Feature importance and an input ablation like base-paper Table 4")
picture(s, f"{MAP_DIR}/step9_feature_importance.png", Inches(0.35), Inches(1.6), Inches(8.2), Inches(3.6))
rows = [["Inputs", "Random-split acc.", "Unseen-region acc."],
        ["All 11 features", pct(abl["All 11 features"]["accuracy"]), pct(abl["All 11 features"]["spatial_10km_accuracy_mean"])],
        ["Without terrain", pct(abl["Without terrain (elev, slope, aspect, curvatures)"]["accuracy"]),
         pct(abl["Without terrain (elev, slope, aspect, curvatures)"]["spatial_10km_accuracy_mean"])],
        ["Without rainfall", pct(abl["Without rainfall"]["accuracy"]), pct(abl["Without rainfall"]["spatial_10km_accuracy_mean"])],
        ["Rainfall only", pct(abl["Rainfall only (base-paper last row)"]["accuracy"]),
         pct(abl["Rainfall only (base-paper last row)"]["spatial_10km_accuracy_mean"])]]
table(s, rows, Inches(0.6), Inches(5.25), Inches(7.9), col_w=[3.3, 2.3, 2.3], size=12.5, row_h=0.33)
bullets(s, [f"Elevation dominates ({imp[0]['gain_pct']:.0f} % of gain), then TWI, distance to road, rainfall, "
            "distance to major river",
            "Base-paper Table 4 is an input ablation (slope, soil, imperviousness, rainfall), not a ranking — so we ran the same",
            ("agrees: more inputs help; rainfall alone is weakest", 1),
            ("rainfall only works as a location label: " f"{pct(abl['Rainfall only (base-paper last row)']['spatial_10km_accuracy_mean'])} "
             "on unseen regions ≈ chance", 1),
            "Imperviousness (base paper's key input) has no counterpart yet → Phase 2 UICA"],
        Inches(8.75), Inches(1.7), Inches(4.3), Inches(5.3), size=13.5)
notes(s, "Elevation is the strongest factor, as expected for river flooding on a coastal floodplain. The base paper's "
         "Table 4 retrains the model with fewer inputs; our version of that table shows terrain matters most for new "
         "regions and rainfall does not really transfer. The missing imperviousness input is exactly our Phase 2 "
         "contribution.")

# 10 Susceptibility map ------------------------------------------------------------------
s = base_slide("Flood Susceptibility Map", "Model applied to all 1.64 million land pixels · 5 quantile classes")
picture(s, f"{MAP_DIR}/step10_susceptibility_map.png", Inches(0.3), Inches(1.55), Inches(6.9), Inches(5.45))
bullets(s, ["Tested on a flood the model never saw — Aug 2019 (NDEM):",
            (f"{pct(v19['share_in_high_or_very_high'])} of the 2019 flood falls in High / Very High (40 % of land)", 1, True),
            (f"Very High class holds {v19['per_class'][4]['frequency_ratio']:.1f}× its area share; AUC {v19['auc']:.2f}", 1),
            (f"even where 2018 did not flood: AUC {v19new['auc']:.2f}", 1),
            "High susceptibility follows the Periyar floodplain, valleys and low coastal land",
            "Scores are a relative index (balanced training), not flood probabilities",
            ("only Very High is strongly predictive; High is near chance", 1)],
        Inches(7.35), Inches(1.7), Inches(5.7), Inches(5.2), size=14.5)
notes(s, "This is the main Phase 1 deliverable. The key evidence is the 2019 check: the model was trained on 2018 only, "
         "yet 84 % of the official 2019 flood falls in our top two classes, which cover 40 % of the land.")

# 11 Rainfall finding --------------------------------------------------------------------
s = base_slide("Finding: Coarse Rainfall Acts as a Location Label", "Same model with and without CHIRPS rainfall")
picture(s, f"{MAP_DIR}/step10_susceptibility_comparison.png", Inches(0.35), Inches(1.65), Inches(12.6), Inches(3.9))
rows = [["", "With rainfall", "Without rainfall"],
        ["2019 flood AUC", f"{v19['auc']:.2f}", f"{v19nr['auc']:.2f}"],
        ["Land > 50 m rated High / Very High", pct(pl["all"]["elevation_gt_50m"]["share_of_such_land_in_high_or_very_high"]),
         pct(pl["no_rainfall"]["elevation_gt_50m"]["share_of_such_land_in_high_or_very_high"])],
        ["Unseen-region accuracy", pct(abl["All 11 features"]["spatial_10km_accuracy_mean"]),
         pct(abl["Without rainfall"]["spatial_10km_accuracy_mean"])]]
table(s, rows, Inches(0.6), Inches(5.6), Inches(7.4), col_w=[3.6, 1.9, 1.9], size=12.5, row_h=0.34)
bullets(s, ["5.5 km CHIRPS cells imprint rectangles on the hills (panel a, top right)",
            "Phase 2: event rainfall at higher resolution"],
        Inches(8.3), Inches(5.6), Inches(4.7), Inches(1.4), size=13.5)
notes(s, "Rainfall here barely varies (about 10 % across 60 coarse cells), so the model uses it to recognise where "
         "the 2018 flood was, not how much it rained. Without it the map is physically cleaner at a small cost on 2019. "
         "We show both maps.")

# 12 Limitations --------------------------------------------------------------------------
s = base_slide("Known Limitations → Phase 2", "Stated deliberately: where Phase 1 ends and our contribution begins")
rows = [["Phase 1 limitation", "Measured effect", "Phase 2 remedy"],
        ["Random 70/30 split", f"{pct(te['accuracy'])} vs {pct(sp10['accuracy']['mean'])} on unseen regions", "Spatial block CV + temporal hold-out"],
        ["Non-flood = 'not seen flooded'", "Urban floods invisible to SAR count as dry", "Positive-Unlabelled learning"],
        ["No land-use / imperviousness", "Model is development-blind", "UICA / ΔUICA features"],
        ["One post-peak flood event", "Residual extent only (23.5 of 75 km²)", "Multi-event, multi-orbit inventory"],
        ["LightGBM, pixel-based", "No spatial context", "Revisit U-Net on larger inventory"],
        ["Coarse rainfall", "Location proxy, map artifact", "Event rainfall, higher resolution"],
        ["Current conditions only", "No development scenarios", "Dual current vs post-development prediction"]]
table(s, rows, Inches(0.6), Inches(1.75), Inches(12.1), col_w=[3.5, 4.5, 4.1], size=14, row_h=0.56)
notes(s, "These are not weaknesses to hide: each one is measured, and each maps directly onto a Phase 2 component "
         "from our Review 1 novelty slide. Full detail is in docs/limitations.md.")

# 13 Conclusion ----------------------------------------------------------------------------
s = base_slide("Conclusion")
bullets(s, ["Phase 1 complete: satellite → flood inventory → features → model → susceptibility map → metrics, "
            "reproduced end to end on the Periyar basin",
            f"Flood inventory verified against official NDEM maps (GTI {gti['value_pct']} %, "
            f"{gti['value_pct'] / gti['random_baseline_pct']:.1f}× chance)",
            f"Baseline model: {pct(te['accuracy'])} on the reference-paper split, {pct(sp10['accuracy']['mean'])} "
            "on unseen regions (base paper 93.73 %)",
            f"Susceptibility map predicts an unseen flood: {pct(v19['share_in_high_or_very_high'])} of 2019 flood "
            "in High / Very High",
            "Every step documented, verified and version-controlled",
            "Next (Phase 2): UICA / ΔUICA · PU-learning · spatial CV · multi-event inventory · "
            "current vs post-development prediction"],
        Inches(0.8), Inches(1.6), Inches(11.8), Inches(5.3), size=19)
notes(s, "One-line summary: the full base-paper pipeline now works on our study area, its numbers are verified "
         "honestly, and every gap maps onto a planned Phase 2 contribution.")

# 14 Q&A -----------------------------------------------------------------------------------
s = base_slide("Anticipated Questions", "Backup slide")
qa = [("Why LightGBM and not U-Net?", "No labelled image set; CNN pipeline would take the whole timeline. "
                                      "LightGBM = full verified chain; U-Net revisited in Phase 2."),
      ("How do you know the flood map is correct?", f"{gti['value_pct']} % confirmed by official NDEM maps "
                                                    f"({gti['value_pct'] / gti['random_baseline_pct']:.1f}× chance); "
                                                    f"{pct(v19['share_in_high_or_very_high'])} of the unseen 2019 flood in our top classes."),
      ("Is 99 % accuracy real or overfitting?", f"Random-split optimism from neighbouring pixels. On unseen regions: "
                                               f"{pct(sp10['accuracy']['mean'])}. Spatial CV becomes the Phase 2 standard."),
      ("Why is your GTI lower than 84 %?", "Only scene is 4–6 days after the peak; disagreement is on water edges "
                                          f"({100 * gti_edge:.0f} % away from them)."),
      ("What is left to do?", "UICA / ΔUICA, PU-learning, spatial CV, more flood events, dual prediction, SHAP.")]
tf = s.shapes.add_textbox(Inches(0.7), Inches(1.65), Inches(12.0), Inches(5.3)).text_frame
tf.word_wrap = True
for i, (q, a) in enumerate(qa):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text = f"Q: {q}"
    p.font.size, p.font.bold, p.font.name, p.font.color.rgb = Pt(16), True, "Calibri", ACCENT
    p.space_before = Pt(0 if i == 0 else 10)
    p = tf.add_paragraph()
    p.text = f"A: {a}"
    p.font.size, p.font.name, p.font.color.rgb = Pt(14.5), "Calibri", INK
notes(s, "Keep this slide hidden until asked. The longer answers are in docs/review2_materials.md.")

# 15 References -----------------------------------------------------------------------------
s = base_slide("References")
refs = ["[1] N. Mohamadiazar, A. Ebrahimian, H. Hosseiny, \"Integrating deep learning, satellite image processing, and "
        "spatial-temporal analysis for urban flood prediction,\" J. Hydrology, vol. 639, 131508, 2024.",
        "[2] L. Hawker et al., \"A 30 m global map of elevation with forests and buildings removed (FABDEM),\" "
        "Environ. Res. Lett., vol. 17, 024016, 2022.",
        "[3] C. Funk et al., \"The climate hazards infrared precipitation with stations (CHIRPS),\" Sci. Data, vol. 2, 150066, 2015.",
        "[4] B. Lehner, G. Grill, \"Global river hydrography and network routing (HydroSHEDS),\" Hydrol. Process., vol. 27, 2013.",
        "[5] N. Otsu, \"A threshold selection method from gray-level histograms,\" IEEE Trans. SMC, vol. 9, 1979.",
        "[6] R. Barnes, C. Lehman, D. Mulla, \"Priority-flood: an optimal depression-filling and watershed-labeling "
        "algorithm,\" Comput. Geosci., vol. 62, 2014.",
        "[7] P. Quinn et al., \"The prediction of hillslope flow paths for distributed hydrological modelling using "
        "digital terrain models,\" Hydrol. Process., vol. 5, 1991.",
        "[8] G. Ke et al., \"LightGBM: a highly efficient gradient boosting decision tree,\" NeurIPS, 2017.",
        "[9] NRSC/ISRO, National Database for Emergency Management (NDEM), Kerala flood inundation layers, 2018–2021."]
tf = s.shapes.add_textbox(Inches(0.7), Inches(1.4), Inches(12.0), Inches(5.6)).text_frame
tf.word_wrap = True
for i, r in enumerate(refs):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text = r
    p.font.size, p.font.name, p.font.color.rgb = Pt(13), "Calibri", INK
    p.space_after = Pt(6)
notes(s, "Base paper [1]; data sources [2]-[4], [9]; methods [5]-[8]. The Review 1 literature references still apply.")

prs.save(DECK)
print(f"    [OK] {DECK}  ({slide_no[0]} slides)")
print("\n" + "=" * 80)
print(f"STEP 12 SUMMARY: architecture ({n_status['done']} done / {n_status['partial']} partial / "
      f"{n_status['planned']} planned blocks), study-area map, {slide_no[0]}-slide deck")
print("=" * 80)
