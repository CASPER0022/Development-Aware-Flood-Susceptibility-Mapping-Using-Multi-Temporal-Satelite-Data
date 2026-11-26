"""
Phase 2 Step 12.4 - Phase 2 deck on the official BTP template.

Same template and helpers as the Review 2 deck (Phase 1 script 16). Every number
is read from the Phase 2 metrics JSON files. Speaker notes carry the split:
Albin, Niketh, Jatin, Ranjith.

Output: outputs/presentation/BTP Phase 2 Team 67.pptx
"""
import copy
import json
import os
import sys

from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

import common as C

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TEMPLATE = os.path.join(C.PROJECT_ROOT, "PPT", "Review 2", "BTP Presentation Template-3.pptx")
OUT_DIR = C.p2("outputs", "presentation")
os.makedirs(OUT_DIR, exist_ok=True)
DECK = os.path.join(OUT_DIR, "BTP Phase 2 Team 67.pptx")
MAP = C.MAP_DIR
J = lambda f: json.load(open(f"{C.METRICS_DIR}/{f}", encoding="utf-8"))

s2 = J("step2_builtup_timeseries.json")
s3 = J("step3_flood_inventory_events.json")
s4 = J("step4_uica.json")
s5 = J("step5_feature_stack_v2.json")
s6 = J("step6_eval_framework.json")["phase1_baseline"]
s7 = J("step7_pu_learning.json")
s8 = J("step8_model_v2_ablation.json")
s9a = J("step9a_ndem_labels.json")
s9b = J("step9b_constrained_model.json")
s9c = J("step9c_extra_runoff.json")
s9 = J("step9_dual_prediction.json")
s10 = J("step10_natural_experiment.json")
s10b = J("step10b_waterlogging_check.json")
s11 = J("step11_shap.json")
s12 = J("step12_headline_tables.json")
abl = s8["ablation_table"]
pb = abl["pu_bagging|C"]
lb = s9["layer_B_waterlogging_pressure"]
nat = s10["placebo_corrected"]["HAND<=5 | T1 upstream paving (UICA ratio)"]
wl = s10b["summary"]["UICA ratio 2025 (upstream paving)"]

INK, MUTED, ACCENT = RGBColor(0x1D, 0x1D, 0x1D), RGBColor(0x59, 0x59, 0x59), RGBColor(0x1F, 0x5F, 0x99)
HEAD_FILL, ROW_FILL, GOOD_FILL = RGBColor(0xDA, 0xE3, 0xEE), RGBColor(0xF4, 0xF6, 0xF9), RGBColor(0xE3, 0xF5, 0xEC)

prs = Presentation(TEMPLATE)
tpl_slides = list(prs.slides)
src = tpl_slides[1]
logo = next(sh for sh in src.shapes if sh.shape_type == 13)
logo_path = os.path.join(OUT_DIR, "_template_logo." + logo.image.ext)
open(logo_path, "wb").write(logo.image.blob)
logo_geom = (logo.left, logo.top, logo.width, logo.height)
footer_xml = [sh._element for sh in src.shapes if sh.is_placeholder and sh.placeholder_format.idx in (11, 12)]
TITLE_ONLY = prs.slide_layouts[5]
X0, W0 = Inches(0.92), Inches(11.5)


def new_slide(title, subtitle=None):
    s = prs.slides.add_slide(TITLE_ONLY)
    s.shapes.add_picture(logo_path, *logo_geom)
    for el in footer_xml:
        s.shapes._spTree.append(copy.deepcopy(el))
    t = s.shapes.title
    t.left, t.top, t.width, t.height = X0, Inches(0.35), Inches(9.4), Inches(0.95)
    t.text = title
    t.text_frame.paragraphs[0].runs[0].font.size = Pt(32)
    if subtitle:
        tb = s.shapes.add_textbox(X0, Inches(1.22), Inches(9.8), Inches(0.45))
        tb.text_frame.margin_left = 0
        p = tb.text_frame.paragraphs[0]
        p.text = subtitle
        p.font.size, p.font.color.rgb = Pt(16), MUTED
    return s


def picture(s, path, left, top, max_w, max_h):
    w, h = Image.open(path).size
    sc = min(max_w / w, max_h / h)
    pw, ph = int(w * sc), int(h * sc)
    s.shapes.add_picture(path, int(left + (max_w - pw) / 2), int(top + (max_h - ph) / 2), pw, ph)


def bullets(s, items, left, top, width, height, size=16):
    tf = s.shapes.add_textbox(left, top, width, height).text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        text, lvl = (it, 0) if isinstance(it, str) else it
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.font.size = Pt(size if lvl == 0 else size - 2)
        p.font.color.rgb = INK if lvl == 0 else MUTED
        p.space_after = Pt(7 if lvl == 0 else 3)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Inches(0.27 if lvl == 0 else 0.55)))
        pPr.set("indent", str(-Inches(0.24)))
        pPr.append(pPr.makeelement(qn("a:buFont"), {"typeface": "Arial"}))
        pPr.append(pPr.makeelement(qn("a:buChar"), {"char": "•" if lvl == 0 else "–"}))


def table(s, rows, left, top, width, col_w=None, size=12, row_h=0.4, highlight_rows=()):
    nr, nc = len(rows), len(rows[0])
    t = s.shapes.add_table(nr, nc, left, top, width, Inches(row_h * nr)).table
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
            cell.fill.fore_color.rgb = HEAD_FILL if r == 0 else (GOOD_FILL if r in highlight_rows else ROW_FILL)
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.size, run.font.bold, run.font.color.rgb = Pt(size), r == 0, INK


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


def fig_slide(title, subtitle, img, points, note, img_w=7.6):
    s = new_slide(title, subtitle)
    picture(s, f"{MAP}/{img}", X0, Inches(1.8), Inches(img_w), Inches(5.0))
    if points:
        bullets(s, points, X0 + Inches(img_w + 0.2), Inches(1.9), Inches(11.5 - img_w - 0.2), Inches(5.0), size=14)
    notes(s, note)
    return s


# 1 Title -------------------------------------------------------------------
ts = tpl_slides[0]
for sh in ts.shapes:
    if sh.is_placeholder and sh.placeholder_format.idx == 0:
        tf = sh.text_frame
        tf.text = "Development-Aware Flood Susceptibility Mapping Using Multi-Temporal Satellite Data"
        tf.paragraphs[0].runs[0].font.size = Pt(36)
        p = tf.add_paragraph()
        p.text = "Phase 2: development-aware modelling and results"
        p.font.size, p.font.color.rgb = Pt(22), ACCENT
    elif sh.is_placeholder and sh.placeholder_format.idx == 1:
        tf = sh.text_frame
        lines = ["BY", "2023BCD0005 - ALBIN JOHN", "2023BCD0056 - NIKETH B NATH", "2023BCD0014 - JATIN NATH",
                 "2023BCD0030 - R RANJITH KUMAR"]
        tf.text = lines[0]
        for ln in lines[1:]:
            p = tf.add_paragraph(); p.text = ln; p.font.italic = True
        for p in tf.paragraphs:
            p.font.size = Pt(15); p.alignment = PP_ALIGN.CENTER
    elif sh.has_text_frame and sh.text_frame.text.startswith("Guided"):
        tf = sh.text_frame
        tf.paragraphs[1].runs[0].text = "Dr. Rosebell Paul"
        for r in tf.paragraphs[1].runs[1:]:
            r.text = ""
notes(ts, "[Albin] In Phase 1 we built a working flood map for the Lower Periyar. Phase 2 was about the "
          "development part of our title: does urban growth change flood risk, and can we measure it.")

# 2 What Phase 2 set out to do ----------------------------------------------
s = new_slide("Phase 2 Goals and Status", "What Review 1 promised and where each item stands")
table(s, [["Goal", "Status"],
          ["Yearly paving record 2017-2025", "Done: 137.8 → 158.9 km² paved"],
          ["More flood events", "Done: 4 validated floods (2018, 2019, 2020, 2024)"],
          ["UICA: paved land draining into each place", "Done: full 9,130 km² basin, all self-tests pass"],
          ["Fair testing (new places, new years)", f"Done: Phase 1 {s6['strict']['2020']['auc_pos_vs_all_land']:.3f} → Phase 2 {pb['strict_2020_auc_land']:.3f}"],
          ["PU-learning", "Done: PU bagging best"],
          ["Current vs post-development maps", "Done: 2025 and 2030 scenarios"],
          ["Show the effect of development", "Partly: by physics yes; in observed floods not yet (Phase 3)"]],
      X0, Inches(1.85), W0, col_w=[5.0, 6.5], size=13, row_h=0.55, highlight_rows=(1, 2, 3, 4, 5, 6))
notes(s, "[Albin] Six of seven goals are done. The last one, showing the development effect in real flood "
         "records, is only partly done, and we explain why later.")

# 3 Architecture -------------------------------------------------------------
s = new_slide("Architecture: End of Phase 2", "Blue: Phase 1, orange: Phase 2, dashed: Phase 3")
picture(s, f"{MAP}/step12_architecture_status.png", X0, Inches(1.8), W0, Inches(5.0))
notes(s, "[Albin] Every orange block was built in Phase 2. The dashed ones are left for Phase 3.")

# 4 Data: paving + floods ----------------------------------------------------
fig_slide("Data 1: How the Area Is Growing", "Yearly paved fraction from Dynamic World + GHSL",
          "step2_builtup_first_last.png",
          ["Paved area 137.8 → 158.9 km² (2017 → 2025)", "About 1.8 % growth a year",
           "Growth mostly around Kakkanad, the airport and Aluva"],
          "[Albin] We built a paving map for every year from satellite data.")
fig_slide("Data 2: Four Validated Floods", "Sentinel-1 flood maps checked against ISRO's NDEM maps",
          "step3_kept_events.png",
          ["2018, 2019, 2020, 2024", "Agreement within 40 m: 30–56 %", "Train on 2018 + 2019, test on 2020 + 2024"],
          "[Albin] More floods means we can train on some years and test on others.")

# 5 UICA ---------------------------------------------------------------------
fig_slide("UICA: Paved Land Upstream", "How much paved land drains into each 30 m cell",
          "step4_uica_maps.png",
          ["Routed over the full 9,130 km² basin", f"River inflow {s4['routing']['periyar_inflow_at_east_edge_km2']:,.0f} km² vs "
           f"{s4['routing']['hydrorivers_reference_km2']:,} km² reference", "Kochi core: 45 % of upstream land paved; forest < 1 %",
           "Also new: HAND (height above nearest stream)"],
          "[Niketh] A place can flood even if it is not paved, because water comes from paved areas above it.")

# 6 Fair test ----------------------------------------------------------------
fig_slide("Testing Fairly", "Same Phase 1 model, five ways of testing it", "step6_baseline_rescored.png",
          ["Random split: 0.999 (too easy)", "New year only: 0.99 (same fields flood every year)",
           f"New year + new place: {s6['strict']['2020']['auc_pos_vs_all_land']:.3f} / {s6['strict']['2024']['auc_pos_vs_all_land']:.3f}",
           "This strict test is our main score now"],
          "[Niketh] We only trust the strict test: a year and an area the model never saw.")

# 7 Ablation -----------------------------------------------------------------
fig_slide("Which Inputs Help?", "Adding one group of features at a time, strict test", "step8_ablation.png",
          ["Hydrology and land cover: clear gains", f"Phase 2 model: {pb['strict_2020_auc_land']:.3f} / {pb['strict_2024_auc_land']:.3f}",
           "PU bagging beats the standard model", "UICA: no gain with satellite labels"],
          "[Niketh] Better terrain and land-cover inputs help. UICA did not, and the next slide shows why.", img_w=8.2)

# 8 Why UICA showed nothing ---------------------------------------------------
fig_slide("Why UICA Showed Nothing at First", "Satellite flood maps do not see towns", "step8b_uica_diagnostics.png",
          ["Radar cannot see water between buildings", "Only 0.1 % of flood examples are built-up",
           "Flood examples sit where UICA ≈ 0", "With ISRO NDEM maps: paved land +0.037 AUC (one test year)"],
          "[Niketh] The model never saw a town flooding, so it could not learn anything from paving.")

# 9 Constrained --------------------------------------------------------------
fig_slide("A Physical Rule for the Model", "More paving must never lower flood risk", "step9b_constraint_checks.png",
          ["The model had learned 'built = dry'", "Building on land made 94 % of places 'safer'",
           "With the rule: 0 %", "Accuracy cost: 0.864 → 0.854"],
          "[Jatin] Without this fix, our future maps would have said building reduces flooding.", img_w=8.2)

# 10 Physics -----------------------------------------------------------------
fig_slide("Physics: How Much More Water?", "SCS curve number runoff, routed through the drainage network", "step9c_extra_runoff.png",
          [f"Big monsoon storm: +{s9c['catchment_totals']['monsoon']['extra_pct']:.2f} % water",
           f"Frequent 50 mm storm: +{s9c['catchment_totals']['frequent']['extra_pct']:.1f} %",
           "Newly paved spots: about +40 %", "Soaked ground already sheds most rain"],
          "[Jatin] Development hardly matters in a big flood, but it matters a lot in everyday heavy rain.", img_w=8.2)

# 11 Development effect (headline) --------------------------------------------
s = new_slide("Main Result: Effect of Development", "Extra water from paving, and where it meets flood-prone land")
picture(s, f"{MAP}/step12_map_development_effect.png", X0, Inches(1.75), W0, Inches(4.25))
bullets(s, [f"Land getting ≥ 5 % more water: {lb['2025']['aoi_km2_more_water']:.0f} km² (2025) → {lb['2030 trend']['aoi_km2_more_water']:.0f} km² (2030)",
            f"Flood-prone hotspots: {lb['2025']['km2_hotspot']:.0f} → {lb['2030 trend']['km2_hotspot']:.0f} km², mostly eastern suburbs",
            f"{100 * lb['2025']['upstream_share_of_extra_water_median_in_more_water_cells']:.0f} % of the extra water comes from paving upstream (UICA)"],
        X0, Inches(6.05), W0, Inches(1.0), size=14)
notes(s, "[Jatin] This is the main development result: paving is already sending more water into places that "
         "flood, and that area grows by 2030.")

fig_slide("Current vs Post-Development Flood Risk", "Riverine susceptibility, 2025 vs 2030", "step12_map_current_vs_post_development.png",
          None, "[Jatin] For the big river floods, the map barely changes. The change is in the water map.", img_w=11.5)

# 12 Natural experiment -------------------------------------------------------
fig_slide("Checking Against Real Floods", "Floods before (2018-19) vs after (2020-24) development", "step10_natural_experiment.png",
          ["Compared growing places with matched non-growing ones",
           f"After a placebo check: {nat['did_minus_placebo_pp']:+.2f} points [{nat['ci95_pp'][0]:+.2f}, {nat['ci95_pp'][1]:+.2f}]",
           "No change in the big floods", "Matches the physics"],
          "[Ranjith] We found no change in the big floods, which is what the physics predicts.", img_w=8.2)

# 13 Waterlogging ------------------------------------------------------------
fig_slide("Real Waterlogging Reports", "15 Kochi spots from news reports (2019, 2024 events)", "step10b_waterlogging_check.png",
          ["Spots are low-lying (significant)", f"Below more paved land: {wl['mean_percentile']:.0f}th percentile, "
           f"{wl['spots_above_50']}/15 above average", "First real-world hint for UICA", "Not significant yet with 15 spots"],
          "[Ranjith] This is our first check against actual waterlogging. It points the right way but needs more data.", img_w=8.2)

# 14 SHAP --------------------------------------------------------------------
fig_slide("What the Model Relies On (SHAP)", "Explaining the river-flood model", "step11_shap_global.png",
          ["Terrain: about 60-66 %", f"Development: {100 * s11['global']['M1']['group_share']['development (paving, UICA)']:.1f} %",
           "Unconstrained models learned 'paving lowers risk'", "Fixed model: correct direction"],
          "[Ranjith] For big floods the model mostly looks at terrain, as expected.", img_w=8.2)

# 15 Phase 1 vs Phase 2 ---------------------------------------------------------
s = new_slide("Phase 1 vs Phase 2", "Same tests, same data")
picture(s, f"{MAP}/step12_map_phase1_vs_phase2.png", X0, Inches(1.7), W0, Inches(3.3))
table(s, s12["phase1_vs_phase2"], X0, Inches(5.1), W0, col_w=[5.5, 3.0, 3.0], size=12, row_h=0.34, highlight_rows=(4,))
notes(s, "[Ranjith] The biggest gain is on the strict test: 0.95 to 0.985.")

# 16 Effect of development table ------------------------------------------------
s = new_slide("Effect of Development: All Evidence", "Big river floods vs everyday heavy rain")
table(s, s12["effect_of_development"], X0, Inches(1.8), W0, col_w=[3.3, 4.0, 4.2], size=12, row_h=0.6)
notes(s, "[Albin] Development matters for everyday waterlogging, not for the big river floods. The waterlogging "
         "part is shown by physics; proving it with observations is Phase 3.")

# 17 Limitations -----------------------------------------------------------------
s = new_slide("Limitations")
bullets(s, ["Satellite and NDEM maps do not record town waterlogging",
            "Urban result rests on one test year (2020)",
            "Paving record is noisy (agreement r = 0.74)",
            "One soil type assumed; drains and paddy-field storage not modelled",
            "Only 15 waterlogging spots so far"], X0, Inches(1.8), W0, Inches(4.8), size=18)
notes(s, "[Albin] We list what we could not do and how much it matters.")

# 18 Phase 3 ---------------------------------------------------------------------
s = new_slide("Phase 3 Plan", "Main job: prove the development effect with real data")
bullets(s, ["Collect Kochi waterlogging records (news, corporation complaints)",
            "Detect flooding between buildings from radar (coherence, double bounce)",
            "Add paddy-field and wetland loss",
            "More flood events (2025-26)",
            "Planner dashboard and final report"], X0, Inches(1.8), W0, Inches(4.8), size=18)
notes(s, "[Albin] Phase 3 Work Package 1 is to test the development effect against real waterlogging.")

# 19 Conclusion -------------------------------------------------------------------
s = new_slide("Conclusion")
bullets(s, [f"Phase 2 model is clearly better on the strict test ({s6['strict']['2020']['auc_pos_vs_all_land']:.2f} → {pb['strict_2020_auc_land']:.3f})",
            "Development barely changes the big river floods",
            f"It adds water in everyday heavy rain: {lb['2025']['km2_hotspot']:.0f} → {lb['2030 trend']['km2_hotspot']:.0f} km² of flood-prone land",
            "Most of that water comes from paving upstream, which is the UICA idea",
            "Real waterlogging spots point the same way; Phase 3 will test it properly"], X0, Inches(1.8), W0, Inches(4.8), size=18)
notes(s, "[Albin] Thank you.")

# 20 References --------------------------------------------------------------------
s = new_slide("References (Phase 2 additions)")
refs = ["C. Elkan and K. Noto, \"Learning classifiers from only positive and unlabeled data,\" KDD, 2008.",
        "F. Mordelet and J.-P. Vert, \"A bagging SVM to learn from positive and unlabeled examples,\" Pattern Recognit. Lett., 2014.",
        "G. Ke et al., \"LightGBM: A highly efficient gradient boosting decision tree,\" NeurIPS, 2017.",
        "S. M. Lundberg and S.-I. Lee, \"A unified approach to interpreting model predictions,\" NeurIPS, 2017.",
        "P. Quinn et al., \"The prediction of hillslope flow paths...,\" Hydrol. Process., 1991.",
        "A. D. Nobre et al., \"Height Above the Nearest Drainage...,\" J. Hydrol., 2011.",
        "USDA NRCS, \"Urban Hydrology for Small Watersheds (TR-55),\" 1986.",
        "S. M. Iacus, G. King and G. Porro, \"Coarsened exact matching,\" Political Analysis, 2012.",
        "C. F. Brown et al., \"Dynamic World,\" Scientific Data, 2022.  M. Pesaresi and P. Politis, \"GHS-BUILT-S R2023A,\" JRC, 2023.",
        "Full list: Phase 2/docs/references_phase2.md"]
bullets(s, refs, X0, Inches(1.6), W0, Inches(5.4), size=12)

# drop template example slides, keep title ---------------------------------------
sldIdLst = prs.slides._sldIdLst
for sld in list(sldIdLst)[1:len(tpl_slides)]:
    prs.part.drop_rel(sld.rId)
    sldIdLst.remove(sld)
A_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
for sl in prs.slides:
    for para in sl._element.iter(A_NS + "p"):
        pPr = para.find(A_NS + "pPr")
        d = pPr.find(A_NS + "defRPr") if pPr is not None else None
        if d is None:
            continue
        for r in para.findall(A_NS + "r"):
            rpr = r.get_or_add_rPr()
            for k, v in d.attrib.items():
                if rpr.get(k) is None:
                    rpr.set(k, v)
prs.save(DECK)
os.remove(logo_path)
print(f"[OK] {DECK}  ({len(prs.slides)} slides)")
