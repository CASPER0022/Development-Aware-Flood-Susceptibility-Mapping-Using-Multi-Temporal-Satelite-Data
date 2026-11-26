"""Builds the Phase 2 study notebook (publish body + standalone copy)."""
import base64, html, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PUB = os.path.join(HERE, "phase2_study_notebook.html")
OUT_STANDALONE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "standalone.html")


def img(name, cap):
    b = base64.b64encode(open(os.path.join(HERE, "img", name + ".webp"), "rb").read()).decode()
    return (f'<figure><div class="fig"><img loading="lazy" alt="{html.escape(cap)}" src="data:image/webp;base64,{b}"></div>'
            f'<figcaption>{cap}</figcaption></figure>')


def table(rows, cls=""):
    h = "".join(f"<th>{c}</th>" for c in rows[0])
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows[1:])
    return f'<div class="tbl {cls}"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def read(*pts):
    return '<div class="read"><span class="lbl">What it means</span><ul>' + "".join(f"<li>{p}</li>" for p in pts) + "</ul></div>"


# ---------------------------------------------------------------- chapters
CH = []


def chapter(cid, part, title, body):
    CH.append((cid, part, title, body))


chapter("story", "A. The idea", "Phase 2 in three minutes", f"""
<p class="lead">Phase 1 built a working flood-susceptibility map for the Lower Periyar (Aluva–Kochi). Phase 2 had one job:
deliver the <b>development-aware</b> part of the title. Does urban growth change flood risk, and can we measure it?</p>
<ol class="story">
<li><b>We measured growth.</b> A paving map for every year 2017–2025: 137.8 → 158.9 km² paved in the study area.</li>
<li><b>We found more floods.</b> 16 candidate events checked against ISRO's NDEM maps; 4 validated (2018, 2019, 2020, 2024).</li>
<li><b>We built UICA.</b> For every 30 m cell: how much paved land drains into it, routed over the full 9,130 km² basin.</li>
<li><b>We tested fairly.</b> Train on 2018 + 2019, test on 2020 + 2024 in areas the model never saw. Phase 1: 0.949 / 0.954. Phase 2: 0.985 / 0.985.</li>
<li><b>UICA did not help at first.</b> Radar cannot see water between buildings, so only 0.1 % of flood examples are urban.</li>
<li><b>We fixed what we could.</b> ISRO NDEM labels gave UICA a small gain on paved land; a physical rule stopped the model saying "paving makes places safer".</li>
<li><b>We measured the effect by physics.</b> Big monsoon floods: +0.2 % water. Everyday heavy rain: flood-prone land getting ≥ 5 % more water grows from 113 km² (2025) to 190 km² (2030).</li>
<li><b>We checked real data.</b> No change in the big floods (as physics predicts). Fifteen reported waterlogging spots sit below more paved land: a first hint for UICA.</li>
</ol>
<div class="callout"><b>One line for the panel:</b> development barely changes where the big river floods go, but it adds a lot of water in
everyday heavy rain, mostly from paving upstream. We show it by physics and see a first hint in real waterlogging reports;
proving it with observations is Phase 3.</div>
{img("step12_architecture_status", "Architecture at the end of Phase 2. Blue: Phase 1, orange: Phase 2, dashed: Phase 3.")}
""")

chapter("terms", "A. The idea", "Words you need", table([
    ["Term", "Meaning"],
    ["Impervious / paved fraction", "Share of a 30 m cell covered by roofs, roads and concrete (0–1)."],
    ["UICA", "Upstream Impervious Contributing Area: km² of paved land that drains into a cell."],
    ["UICA ratio", "UICA ÷ upslope area: the share of a cell's catchment that is paved."],
    ["ΔUICA", "How much UICA grew between two years: the development signal."],
    ["HAND", "Height Above Nearest Drainage: how far a cell sits above the stream it drains to."],
    ["Label", "The answer the model learns: flooded or not."],
    ["PU-learning", "Positive-Unlabelled: treat 'not seen flooded' as unknown, not as dry."],
    ["Strict test", "Train on 2018 + 2019, test on 2020 + 2024 in 10 km blocks the model never saw."],
    ["AUC", "Chance the model ranks a random flooded cell above a random dry one. 0.5 = guessing, 1 = perfect."],
    ["Monotone rule", "A constraint: more paving can only keep risk the same or raise it."],
    ["SCS curve number", "Standard engineering formula for how much rain runs off, from soil and paving."],
    ["Difference-in-differences", "Change in the treated group minus change in a matched control group."],
    ["Placebo test", "Run the test where no effect can exist; a 'result' there means the method fools itself."],
    ["SHAP", "Splits a model's score at each place into the contribution of each input."],
]))

# ------------------------------------------------------------- theory
chapter("t-routing", "B. Theory from scratch", "Flow routing and UICA", f"""
<p>Rain that falls on a cell flows downhill. <b>Flow routing</b> computes, from the elevation model, where the water of every cell goes.
We use the same two steps as Phase 1:</p>
<ul><li><b>Priority-Flood</b> fills fake pits in the DEM so every cell can drain to the sea.</li>
<li><b>Quinn FD8</b> splits a cell's water among all lower neighbours, in proportion to slope. Accumulating this downhill gives each
cell's <b>upslope area</b>.</li></ul>
<p><b>UICA is the same accumulation with a different weight.</b> Instead of starting every cell with its area, start it with its
<i>paved</i> area (area × paved fraction). What arrives at a cell is the paved land upstream of it.</p>
<svg class="diagram" viewBox="0 0 640 170" role="img" aria-label="UICA concept">
 <rect x="10" y="20" width="120" height="50" rx="6" class="d-paved"/><text x="70" y="50" class="d-t">paved hill</text>
 <rect x="10" y="100" width="120" height="50" rx="6" class="d-green"/><text x="70" y="130" class="d-t">forest</text>
 <rect x="250" y="60" width="130" height="50" rx="6" class="d-box"/><text x="315" y="90" class="d-t">valley cell</text>
 <rect x="490" y="60" width="140" height="50" rx="6" class="d-box"/><text x="560" y="84" class="d-t">UICA = paved km²</text><text x="560" y="100" class="d-s">arriving from above</text>
 <line x1="130" y1="45" x2="248" y2="78" class="d-a"/><line x1="130" y1="125" x2="248" y2="95" class="d-a2"/><line x1="380" y1="85" x2="488" y2="85" class="d-a"/>
</svg>
<p>Phase 1 routed only inside the study area, which cut off the Periyar's 6,000 km² upstream. Phase 2 routes the <b>full 9,130 km²
basin</b>: the Periyar inflow comes out at 4,103 km² against 4,081 km² in HydroRIVERS (0.5 % apart).</p>
{img("step4_uica_maps", "UICA maps. Kochi core: about 45 % of the upstream land is paved; forested hills: below 1 %.")}
""")

chapter("t-hand", "B. Theory from scratch", "HAND: height above the nearest stream", f"""
<p>Raw elevation says how high a place is above the sea. For flooding, what matters is how high it is above <b>the stream it drains
to</b>. HAND follows each cell's steepest path down to a drainage cell (≥ 1 km² upslope) and records the height difference.</p>
<p>On the flat floodplain, elevation barely changes, but HAND still separates the riverbank from slightly higher ground.
Single-feature AUC: HAND 0.951 vs elevation 0.915; on the floodplain alone, 0.907 vs 0.764.</p>
{img("step5_twi_hand", "HAND and the full-basin wetness index (TWI).")}
""")

chapter("t-testing", "B. Theory from scratch", "Testing fairly: spatial, temporal and strict", f"""
<p>Neighbouring 30 m cells are nearly identical. A random train/test split puts test cells right next to training cells, so the model
can memorise the neighbourhood (Phase 1: 99.9 % on random split vs 88.5 % on unseen 10 km regions).</p>
{table([["Test", "Train on", "Test on", "What it checks"],
        ["Spatial (10 km blocks)", "4 events, 4 of 5 block groups", "the 5th group", "new places"],
        ["Temporal", "2018 + 2019", "2020, 2024", "new years (too easy: the same fields flood every year)"],
        ["<b>Strict</b>", "2018 + 2019, 4 groups", "2020, 2024 in the 5th group", "<b>new year and new place</b>"]])}
<p>The block-to-group assignment is saved once and reused by every model, so comparisons are fair. We also score
<b>flood vs all other land</b>, not only flood vs clearly dry land, which is harder and closer to the real map.</p>
{img("step6_baseline_rescored", "The same Phase 1 model under five tests. Only the strict test (green) measures real generalisation.")}
""")

chapter("t-pu", "B. Theory from scratch", "PU-learning", """
<p>Our "dry" cells are really "not seen flooded". The satellite passes once, often after the peak, and cannot see water between
buildings. PU-learning treats them as <b>unlabelled</b>.</p>
<ul><li><b>Elkan–Noto:</b> train flooded vs unlabelled, estimate how often a true flood gets labelled (<i>c</i>), divide the score by
<i>c</i>. Our <i>c</i> = 0.17, which implies about 7 % of land is flood-prone.</li>
<li><b>Bagging PU:</b> 25 small models, each trained on all floods plus a different random handful of unlabelled land, averaged.
This won: strict AUC 0.984 vs 0.975 for the standard model.</li></ul>
<p class="note">PU cannot invent a kind of flood it never saw. With almost no urban flood examples, it could not fix the Kochi blind spot.</p>
""")

chapter("t-calib", "B. Theory from scratch", "Calibration: from score to probability", """
<p>We train on equal numbers of flooded and dry cells (50/50), but only about 0.4–0.8 % of land floods in an event. So raw scores are
far too high (5.4 % predicted vs 0.37 % observed). <b>Prior correction</b> rescales them to the real share:
p' = p·π / (p·π + (1−p)(1−π)). That gave 0.38 % predicted vs 0.37 % observed. Platt and isotonic scaling learned the size of the
2018/2019 floods (twice as big as 2020/2024) and over-predicted 2×. The lesson: a flood probability depends on how big the event is.</p>
""")

chapter("t-mono", "B. Theory from scratch", "Monotone constraints", """
<p>A gradient-boosting model can learn any shape. With biased labels it learned that more paving means <i>less</i> flooding. A
<b>monotone constraint</b> forces the prediction to be non-decreasing in chosen inputs (paved fraction, UICA, UICA ratio, ΔUICA):
no split may make risk go down when those values go up. It encodes physics the data cannot show us. We also removed the land-cover
class, because its "built" category carried the same bias.</p>
""")

chapter("t-scs", "B. Theory from scratch", "SCS curve number runoff", """
<p>USDA TR-55 method used worldwide for drain design. Each cell gets a curve number <b>CN</b> (higher = more runoff):
CN = paved × 98 + (1 − paved) × 76 (woods–grass, soil group C). Then:</p>
<p class="eq">S = 25400 / CN − 254 &nbsp;&nbsp;·&nbsp;&nbsp; Q = (P − 0.2S)² / (P + 0.8S) when P &gt; 0.2S</p>
<p>P is the storm rainfall and Q the runoff, both in mm. Wet monsoon soil uses the AMC III conversion CN<sub>III</sub> = 23CN / (10 + 0.13CN).
Why it matters: on soaked soil, unpaved land already sheds 82 % of a 200 mm storm; on normal soil it sheds only 20 % of a 50 mm storm,
while paved land sheds 89 %. That is why development matters in everyday rain, not in the extreme flood.</p>
""")

chapter("t-did", "B. Theory from scratch", "Natural experiments: DiD, matching, placebo", """
<ul><li><b>Difference-in-differences:</b> (after − before) for places that developed, minus (after − before) for places that did not.
The subtraction removes everything that changed for both, like the size of the flood.</li>
<li><b>Coarsened exact matching:</b> compare only within groups of similar places (same HAND band, TWI, distance to river and rain band).</li>
<li><b>Placebo:</b> test 2018 vs 2019, before most of the growth. If "future" development already shows an effect there, the method is
picking up <i>which places</i> grow, not the growth.</li>
<li><b>Block bootstrap:</b> resample 2 km blocks 500 times to get honest uncertainty; the <b>minimum detectable effect</b> (2.8 × SE)
says how big an effect the design could have seen.</li></ul>
""")

chapter("t-shap", "B. Theory from scratch", "SHAP", """
<p>For each place, SHAP splits the model's score (in log-odds) into one contribution per input, adding up exactly to the prediction.
Averaging the absolute contributions over many places shows what the model relies on. It explains the <b>model</b>, not the world:
our models learned big river floods, so SHAP tells how they predict those, not waterlogging.</p>
""")

# ------------------------------------------------------------- steps
chapter("s1-3", "C. What we did, step by step", "Steps 1–3: baseline, paving record, more floods", f"""
<p><b>Step 1.</b> Moved all Phase 1 code into a shared module and reproduced Phase 1 exactly (10/10 checks, AUC 0.9993, every test
prediction identical). Tagged the frozen baseline <code>phase1-final</code>.</p>
<p><b>Step 2.</b> Built a paved-fraction map for each year 2017–2025: GHSL 2020 as the anchor, plus 0.214 × the change in Dynamic World
built-up. Paved area 137.8 → 158.9 km² (about 1.8 %/yr); agreement with WorldCover r = 0.74.</p>
{img("step2_builtup_first_last", "Paving 2017 vs 2025 and the change.")}
<p><b>Step 3.</b> Looked for floods in CHIRPS rainfall peaks and NDEM dates (16 candidates), mapped each with Sentinel-1 using the Phase 1
method, and kept only maps that agree with NDEM: 2018, 2019, 2020, 2024 (agreement within 40 m: 30–56 %).</p>
{img("step3_kept_events", "The four validated flood events.")}
""")

chapter("s4-5", "C. What we did, step by step", "Steps 4–5: UICA and the new feature stack", """
<p><b>Step 4.</b> Routed the full 9,130 km² upstream basin (61 HydroBASINS units), computed UICA for every year and ΔUICA. Five
self-tests pass, including an exact match to an analytic plane.</p>
<p><b>Step 5.</b> Added HAND, full-basin TWI, 5-day and 30-day event rainfall, paved fraction, land cover, UICA, UICA ratio and ΔUICA:
18 features for each of the 4 events. Largest VIF 9.5 (elevation and HAND are correlated, r = 0.93).</p>
""")

chapter("s6-8", "C. What we did, step by step", "Steps 6–8: fair testing, PU-learning, which inputs help", f"""
<p><b>Step 6</b> set up the strict test; the Phase 1 model scores 0.949 / 0.954 on it. <b>Step 7</b> compared PU methods; PU bagging
won. <b>Step 8</b> tuned LightGBM (18 settings, all 0.993–0.994, so tuning hardly matters) and added inputs one group at a time:</p>
{table([["Model", "Adds", "Strict AUC 2020 / 2024 (PU bagging)"],
        ["A", "Phase 1 inputs", "0.969 / 0.968"], ["B", "HAND, full TWI, event rain", "0.976 / 0.973"],
        ["<b>C</b>", "<b>paved fraction, land cover</b>", "<b>0.985 / 0.985</b>"], ["D", "UICA, UICA ratio, ΔUICA", "0.985 / 0.984"]])}
{img("step8_ablation", "Ablation under the same folds. Blue: standard, orange: PU bagging, dashed: XGBoost.")}
{read("Hydrology and land cover give real gains on new years and places.", "UICA (D vs C) adds nothing with radar labels.",
      "XGBoost does no better: the data, not the algorithm, is the limit.")}
{img("step8b_uica_diagnostics", "Why UICA showed nothing: flood labels sit where UICA is low; Kochi is high. UICA gets 0.8 % of the model's gain.")}
""")

chapter("s9a", "C. What we did, step by step", "Step 9.0a: ISRO NDEM labels", f"""
<p>NDEM maps also mark flooding the radar missed, some of it in towns. Only 2018 and 2020 outlines are detailed enough
(≥ 50 vertices per km); 2019 and 2024 are rough and were dropped. The added flooding is 5–10 % paved, against 0 % for radar flooding.</p>
{img("step9a_urban_tests", "C vs D on all land and urban subsets, with radar labels and with radar + NDEM labels.")}
{read("With NDEM labels, UICA helps on built-up land in the strict 2020 test: paved land +0.037 (95 % CI +0.021 to +0.058), built-up +0.015.",
      "Only one test year has detailed NDEM, so this is a small, consistent result, not proof.")}
""")

chapter("s9b", "C. What we did, step by step", "Step 9.0b: the physical rule", f"""
{img("step9b_constraint_checks", "Response to paving before and after the rule; share of 100,000 'developed' plots that became safer; accuracy cost.")}
{table([["Model", "Plots 'safer' after building"], ["Normal model", "94 %"], ["Rule, land cover kept", "53 %"], ["<b>Rule + no land cover</b>", "<b>0 %</b>"]])}
{read("Without the rule, future-development maps would say building reduces flooding.",
      "Cost: strict AUC 0.864 → 0.854 (the pre-set limit was 0.01; accepted because the alternative is physically wrong).",
      "With the rule the response to paving is flat: the model cannot size the development effect; physics has to.")}
""")

chapter("s9c", "C. What we did, step by step", "Step 9.0c: extra runoff from development", f"""
{img("step9c_extra_runoff", "Extra runoff 2017 → 2025: big monsoon storm (left), frequent storm (middle), sensitivity (right).")}
{table([["", "Monsoon storm (200 mm, wet soil)", "Frequent storm (50 mm, normal soil)"],
        ["Unpaved / paved land sheds", "82 % / 99 %", "20 % / 89 %"],
        ["Whole catchment, extra water", "+0.17 %", "+1.4 %"],
        ["Newly paved spots", "+5 %", "+43 %"]])}
""")

chapter("s9", "C. What we did, step by step", "Step 9: development effect, now and in 2030", f"""
{img("step12_map_development_effect", "Extra frequent-storm water 2017 → 2030 and the development-impact classes.")}
{table([["", "2025", "2030 trend", "2030 GHSL"],
        ["Land with ≥ 5 % more water", "280 km²", "434 km²", "299 km²"],
        ["<b>Hotspots</b> (flood-prone + more water)", "<b>113 km²</b>", "<b>190 km²</b>", "122 km²"],
        ["Extra water from upstream paving", "87 %", "100 %", "88 %"],
        ["River-flood map: land up a class", "1.1 %", "1.2 %", "1.3 %"]])}
{read("Development is already sending more water into flood-prone places, mostly from paving upstream: the UICA idea.",
      "The worst places are the growing eastern suburbs: Kolenchery, Pallikkara (Kakkanad side), Angamaly, Kalady.",
      "The trend holds at 2 % and 10 % thresholds. Absolute size: about 1 mm extra per 50 mm storm on affected land.")}
""")

chapter("s10", "C. What we did, step by step", "Step 10: did real floods change?", f"""
{img("step10_natural_experiment", "Difference-in-differences estimates with 95 % ranges, before/after rates, and the compared places.")}
{read("The simple estimate (+0.42 points) was matched by the placebo (+0.47): the places that would grow already behaved differently.",
      "Placebo-corrected: −0.05 points (range −0.71 to +0.58). No change in the big floods.",
      "This is what physics predicts (+0.2 % water in monsoon storms), far below what four floods can detect (about 0.5 points).",
      "Local paving could not be tested: radar never sees floods on paved land (floor effect).")}
""")

chapter("s10b", "C. What we did, step by step", "Step 10b: real waterlogging reports", f"""
<p>Fifteen Kochi places reported waterlogged in news and official sources (mainly 21 Oct 2019, about 20 cm in 4 hours, and the
28 May 2024 cloudburst), located with OpenStreetMap and compared with other urban land.</p>
{img("step10b_waterlogging_check", "The 15 spots over upstream paving, and where each ranks among other urban land.")}
{table([["Check", "Spots' percentile among urban land", "Meaning"],
        ["Height above stream", "28th (significant)", "waterlogging happens in low spots"],
        ["<b>Upstream paving (UICA ratio)</b>", "<b>57th–61st</b>; 11 of 15 above; old city 70th–87th", "first real-world hint for UICA"],
        ["Paving growth since 2017", "49th", "the reported spots are in the old, already-paved city"],
        ["River-flood model score", "45th", "the river model does not catch waterlogging"]])}
""")

chapter("s11", "C. What we did, step by step", "Step 11: what the model relies on", f"""
{img("step11_shap_global", "SHAP: input-group shares in three models, development's share, and the direction of the paving effect.")}
{read("Terrain and hydrology carry 59–66 % of every model's decisions.",
      "Development: 5.3 % (chosen model), 4.0 % (unconstrained), 0.6 % (rule-bound): small for big river floods, as expected.",
      "Unconstrained models learned 'paving lowers risk' (correlation −0.91 and −0.80); the rule-bound model gives UICA the physical sign (+0.61).")}
""")

chapter("s12", "C. What we did, step by step", "Step 12: Phase 1 vs Phase 2", f"""
{img("step12_map_phase1_vs_phase2", "Phase 1 map, Phase 2 map and the class change: 53 % same class, 92 % within one class.")}
{table([["Test", "Phase 1", "Phase 2"], ["Random split", "0.999", "0.999"], ["Unseen 10 km regions", "0.962", "0.992"],
        ["New year", "0.988 / 0.989", "0.992 / 0.993"], ["<b>New year + new place</b>", "<b>0.949 / 0.954</b>", "<b>0.985 / 0.985</b>"]])}
""")

# ------------------------------------------------------------- code
chapter("code", "D. The code", "Script by script", table([
    ["Script", "Does", "Main output"],
    ["common.py", "Shared code: paths, routing (FD8, Priority-Flood, HAND), label pools, test protocols, PU models", "–"],
    ["01_freeze_baseline_check.py", "Reproduces Phase 1 exactly through common.py", "step1 JSON (10/10)"],
    ["02_built_up_timeseries.py", "Yearly paved fraction 2017–2025", "impervious_fraction_30m.tif"],
    ["03_expand_flood_inventory.py", "16 candidate floods → 4 validated", "event table"],
    ["04_uica_full_basin.py", "Full-basin routing, UICA, ΔUICA, self-tests", "uica rasters"],
    ["05_feature_stack_v2.py", "HAND, event rain, land cover, 18-feature stacks", "stack_v2_*.tif"],
    ["06_eval_framework.py", "Dataset v2, fixed folds, strict test, Phase 1 re-scored", "samples_v2.parquet"],
    ["07_pu_learning.py", "Standard vs PU methods", "step7 JSON"],
    ["08_model_v2_ablation.py, 08b", "Tuning, A–D ablation, XGBoost, calibration, UICA diagnosis", "models, step8 JSON"],
    ["09a / 09b / 09c", "NDEM labels; monotone scenario model; SCS-CN extra runoff", "model_v2_scenario.txt"],
    ["09_dual_prediction.py", "Scenarios 2017/2025/2030, two layers, hotspots", "scenario_*.tif, hotspot CSV"],
    ["10_natural_experiment.py, 10b", "DiD with placebo; waterlogging spots check", "step10 JSONs"],
    ["11_shap_explanations.py", "TreeSHAP for three models", "step11 JSON"],
    ["12_phase2_maps.py, 12b", "Final maps, tables, architecture; the Phase 2 deck", "maps, .pptx"],
]))

# ------------------------------------------------------------- teaching
chapter("teach", "E. Presenting", "How to present it", table([
    ["Member", "Part", "Numbers to remember"],
    ["Albin", "Goals, data, UICA, fair testing", "137.8 → 158.9 km²; 4 floods; 4,103 vs 4,081 km²; 0.949 → 0.985"],
    ["Niketh", "Why UICA showed nothing, NDEM labels, the physical rule", "0.1 % urban labels; +0.037 on paved land; 94 % → 0 %"],
    ["Jatin", "Physics and the development effect", "+0.2 % vs +43 %; 280 → 434 km²; 113 → 190 km²; 87 % upstream"],
    ["Ranjith", "Real floods, waterlogging, SHAP, Phase 3", "placebo +0.47, corrected −0.05; 15 spots, 57th–61st pct; development 4–5 %"],
]))

QA = [
    ("Your title says development-aware. Where is the evidence that development increases flooding?",
     "Three parts. Physics: development adds +43 % runoff on newly paved land in frequent storms and puts 113 → 190 km² of flood-prone land under more water. Model: with NDEM labels, UICA improves prediction on paved land (+0.037). Observation: real waterlogging spots sit below more paved land (57th–61st percentile). The big river floods show no change, which the physics predicts. Proving the waterlogging effect with observations is Phase 3."),
    ("Why did UICA add nothing in the first ablation?", "The radar flood labels are 0.1 % built-up. The model never saw a town flood, so a feature about paved catchments had nothing to explain. UICA got 0.8 % of the model's gain."),
    ("Isn't 0.985 suspiciously high?", "It is flood vs all land on a new year and new place. The floods are mostly the same kind (river-side, low land), which is easier than city waterlogging. Against radar + NDEM truth the score is about 0.86."),
    ("Why not a random train/test split?", "Neighbouring cells are nearly identical; a random split lets the model memorise neighbourhoods (99.9 %). Our strict test hides both the year and the area."),
    ("Why is the temporal test alone not enough?", "The same floodplain floods every year, so a model that remembers where past floods were scores 0.99 on a new year."),
    ("What is PU-learning and did it help?", "It treats 'not seen flooded' as unknown. PU bagging improved the strict AUC from 0.975 to 0.984, but could not fix Kochi because there were no urban flood examples."),
    ("Why did you add a monotone rule? Isn't that forcing the answer?", "It only forbids a physically impossible answer (paving making water soak in better). Without it, 94 % of developed plots came out 'safer'. With it, the model's response is flat, so it does not force a positive effect either."),
    ("Why accept a 0.011 accuracy cost when your limit was 0.01?", "The lost accuracy came from the label bias itself (2024 truth is radar-only and rewards 'built = dry'), and the alternative gives wrong scenario maps. We state this openly."),
    ("Why the SCS curve number method?", "It is the standard method used for drain design (USDA TR-55), needs only paving, soil and rainfall, and gives the effect size without flood labels."),
    ("Which soil did you assume?", "One soil group (C) for the basin, because no detailed soil map can be downloaded. Groups B and D change the numbers about 2×, not the pattern."),
    ("Why does development matter so little in the big floods?", "Soaked monsoon soil already sheds 82 % of the rain; paving it changes little (+0.2 %). In ordinary storms unpaved soil sheds only 20 %, so paving matters."),
    ("What is a hotspot?", "Land the model already rates High or Very High for flooding that also gets at least 5 % more water in a frequent storm because of development."),
    ("How sure is the 2030 number?", "It depends on how fast building continues: 299–434 km² of land with more water (GHSL projection vs our trend). Hotspots 122–190 km²."),
    ("Why does most extra water come from upstream?", "Water from paved land flows downhill, so a place receives the runoff of all paved land above it. That is exactly what UICA measures; 87 % of the extra water at affected places comes from upstream."),
    ("What did the natural experiment show?", "No change in big floods after correcting with a placebo (−0.05 points, range −0.71 to +0.58). The first estimate (+0.42) was fake: the placebo showed the same."),
    ("Why use a placebo test?", "It checks whether the method finds an 'effect' where none can exist. It caught a fake result that we would otherwise have reported."),
    ("Why couldn't you test local paving?", "Radar never sees floods on paved land, so newly paved cells are 'never flooded' before and after (floor effect). The result is meaningless."),
    ("Where did the waterlogging data come from?", "Fifteen places named in news and official reports (2019, 2024 events), located with OpenStreetMap. Each has a source link in the CSV."),
    ("Isn't 15 spots too few?", "Yes, it is a first hint, not proof (p about 0.08–0.19). Phase 3 needs corporation complaint records and more events."),
    ("Why doesn't the 2017–2025 growth layer match the waterlogging spots?", "The reported spots are in the old city, paved long before 2017. Their waterlogging comes from existing paving and drainage. New growth areas have few reports yet."),
    ("What does SHAP show?", "Terrain and hydrology drive about 60–66 % of decisions; development 4–5 %. Development is a small factor for big river floods, as expected."),
    ("Why does the Phase 2 map rate Kochi low?", "The labels hardly contain urban flooding, so the river-flood model rates built-up land lower. Every map states that built-up risk is a lower bound."),
    ("Why LightGBM and not U-Net?", "Four flood events are far too few for a CNN. XGBoost also did no better than LightGBM, so the algorithm is not the limit."),
    ("Did tuning help?", "Barely: all 18 settings scored 0.993–0.994. The training-test gap moved from 0.0071 to 0.0065."),
    ("Why drop the 2019 and 2024 NDEM maps?", "Their outlines are rough shapes (about 23–26 vertices per km) that look like random land; 2018 and 2020 are traced in detail (177–329)."),
    ("Is the mean annual rainfall still a problem?", "It is a mild location stand-in: removing it costs 0.007 on the strict test and SHAP gives it 5–8 %. We keep it and flag it."),
    ("What are your probabilities?", "After prior correction: 'chance this cell is in the confirmed flood of an event like these'. Mean 0.38 % predicted vs 0.37 % observed on 2020/2024. They depend on event size."),
    ("What does Phase 3 add?", "Work Package 1: prove the development effect with real data (Kochi waterlogging records, radar detection between buildings, paddy and wetland loss, more events). Then a planner dashboard and the report."),
    ("What is the patent angle?", "The UICA-based development-aware method and the scenario workflow. Keep the code private until a provisional filing; the repository is public."),
    ("What would you do differently?", "Collect urban waterlogging records from the start, because the satellite labels could never show the development effect that the title is about."),
]
chapter("qa", "E. Presenting", f"Panel questions ({len(QA)})",
        '<p>Click a question to see the answer.</p>' + "".join(
            f"<details><summary>{html.escape(q)}</summary><p>{html.escape(a)}</p></details>" for q, a in QA))

chapter("sources", "E. Presenting", "Sources", """
<p>The full IEEE list ([P1]–[P32]) is in <code>Phase 2/docs/references_phase2.md</code>. Key methods: Elkan &amp; Noto 2008 (PU);
Mordelet &amp; Vert 2014 (bagging PU); Ke et al. 2017 (LightGBM); Lundberg &amp; Lee 2017 (SHAP); Quinn et al. 1991 (FD8);
Barnes et al. 2014 (Priority-Flood); Nobre et al. 2011 (HAND); USDA TR-55 1986 (curve number); Iacus, King &amp; Porro 2012 (matching).
Data: Dynamic World (Brown et al. 2022), GHSL R2023A, CHIRPS (Funk et al. 2015), HydroSHEDS (Lehner &amp; Grill 2013), FABDEM,
NDEM (NRSC/ISRO), Sentinel-1, OpenStreetMap. Waterlogging reports: Onmanorama, The News Minute, Deccan Herald, Urban Acres (links in
the reference file).</p>
""")

# ------------------------------------------------------------- page
parts = []
for cid, part, title, _ in CH:
    if part not in parts:
        parts.append(part)
nav = ""
for p in parts:
    nav += f'<div class="navpart">{p}</div>' + "".join(
        f'<a href="#{cid}">{t}</a>' for cid, pp, t, _ in CH if pp == p)
body = ""
for p in parts:
    body += f'<div class="partlabel">{p}</div>'
    for cid, pp, t, b in CH:
        if pp == p:
            body += f'<section id="{cid}"><h2>{t}</h2>{b}</section>'

CSS = """
:root{--bg:#f6f8f7;--panel:#ffffff;--fg:#1b2423;--muted:#56625f;--line:#d6dedb;--accent:#1d6a73;--accent2:#b5651d;--soft:#e7f0ef;--warn:#fff4e2;
--fd:'Fraunces',Georgia,serif;--fb:'IBM Plex Sans',system-ui,sans-serif;--fm:'IBM Plex Mono',ui-monospace,monospace;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121816;--panel:#19211f;--fg:#e4ecea;--muted:#9fb0ac;--line:#2c3835;--accent:#62b8c1;--accent2:#e3a160;--soft:#1f2b29;--warn:#2b2418;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#121816;--panel:#19211f;--fg:#e4ecea;--muted:#9fb0ac;--line:#2c3835;--accent:#62b8c1;--accent2:#e3a160;--soft:#1f2b29;--warn:#2b2418;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:16px/1.6 var(--fb)}
.wrap{display:grid;grid-template-columns:260px minmax(0,1fr);gap:40px;max-width:1240px;margin:0 auto;padding-block:28px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
nav{position:sticky;top:calc(env(safe-area-inset-top,0px) + 16px);align-self:start;max-height:calc(100vh - 32px);overflow:auto;font-size:13.5px}
nav a{display:block;color:var(--muted);text-decoration:none;padding:3px 0 3px 10px;border-left:2px solid var(--line)}
nav a:hover,nav a:focus-visible{color:var(--accent);border-left-color:var(--accent);outline:none}
.navpart{font:600 11px/1 var(--fm);letter-spacing:.08em;text-transform:uppercase;color:var(--accent2);margin:16px 0 6px}
header h1{font:600 clamp(28px,4vw,42px)/1.15 var(--fd);margin:0 0 8px;text-wrap:balance}
header p{color:var(--muted);margin:0 0 6px;max-width:70ch}
.toggle{font:13px var(--fb);background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer;margin-top:10px}
.toggle:focus-visible{outline:2px solid var(--accent)}
.partlabel{font:600 12px/1 var(--fm);letter-spacing:.08em;text-transform:uppercase;color:var(--accent2);margin:48px 0 0;padding-top:14px;border-top:1px solid var(--line)}
section{margin-top:26px;scroll-margin-top:16px}
h2{font:600 26px/1.25 var(--fd);margin:0 0 12px;text-wrap:balance}
p,li{max-width:72ch}
.lead{font-size:18px}
.story li{margin-bottom:6px}
.callout{background:var(--soft);border-radius:8px;padding:14px 16px;margin:16px 0;max-width:80ch}
.note{color:var(--muted);font-style:italic}
.eq{font:15px var(--fm);background:var(--soft);padding:10px 12px;border-radius:6px;overflow-x:auto}
code{font:14px var(--fm);background:var(--soft);padding:1px 5px;border-radius:4px}
figure{margin:18px 0}
.fig{background:#fff;border:1px solid var(--line);border-radius:6px;padding:6px;overflow:hidden}
.fig img{display:block;width:100%;height:auto}
figcaption{font-size:13.5px;color:var(--muted);margin-top:6px;max-width:90ch}
.tbl{overflow-x:auto;margin:14px 0}
table{border-collapse:collapse;font-size:14.5px;min-width:520px;width:100%;font-variant-numeric:tabular-nums}
th{text-align:left;background:var(--soft);font-weight:600}
th,td{border-bottom:1px solid var(--line);padding:7px 10px;vertical-align:top}
.read{border-left:3px solid var(--accent);padding:4px 14px;margin:14px 0}
.read .lbl{font:600 11px var(--fm);letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}
.read ul{margin:6px 0 0;padding-left:18px}
details{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px 14px;margin:8px 0;max-width:85ch}
summary{cursor:pointer;font-weight:600}
summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
details p{margin:8px 0 0;color:var(--fg)}
.diagram{width:100%;max-width:640px;height:auto;display:block;margin:12px 0}
.d-box{fill:var(--panel);stroke:var(--accent);stroke-width:1.5}.d-paved{fill:var(--warn);stroke:var(--accent2);stroke-width:1.5}
.d-green{fill:var(--soft);stroke:var(--line);stroke-width:1.5}.d-t{fill:var(--fg);font:13px var(--fb);text-anchor:middle}
.d-s{fill:var(--muted);font:11px var(--fb);text-anchor:middle}.d-a{stroke:var(--accent2);stroke-width:2.5}.d-a2{stroke:var(--line);stroke-width:2}
@media (max-width:860px){.wrap{grid-template-columns:minmax(0,1fr)}nav{position:static;max-height:none;border-bottom:1px solid var(--line);padding-bottom:12px}}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
"""
JS = """
(function(){var b=document.getElementById('themeBtn');function cur(){var t=document.documentElement.getAttribute('data-theme');
if(t)return t;return window.matchMedia&&matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'}
try{var s=localStorage.getItem('p2nb-theme');if(s)document.documentElement.setAttribute('data-theme',s)}catch(e){}
function lab(){b.textContent=cur()==='dark'?'Light theme':'Dark theme'}lab();
b.addEventListener('click',function(){var n=cur()==='dark'?'light':'dark';document.documentElement.setAttribute('data-theme',n);
try{localStorage.setItem('p2nb-theme',n)}catch(e){}lab()})})();
"""
HEAD = ('<title>Periyar Phase 2 Notebook</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap">\n'
        f"<style>{CSS}</style>\n")
PAGE = (f'<div class="wrap"><nav aria-label="Contents">{nav}</nav><main>'
        '<header><h1>Periyar Flood Study: Phase 2 Notebook</h1>'
        '<p>Development-aware flood susceptibility mapping, Lower Periyar (Aluva–Kochi). Team 67 · Guide: Dr. Rosebell Paul.</p>'
        '<p>Theory from scratch, every Phase 2 step with its real result figure, the code, and panel questions with answers.</p>'
        '<button class="toggle" id="themeBtn" type="button">Dark theme</button></header>'
        f"{body}</main></div><script>{JS}</script>")
open(OUT_PUB, "w", encoding="utf-8").write(HEAD + PAGE)
open(OUT_STANDALONE, "w", encoding="utf-8").write(
    '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    '<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>' + HEAD + "</head><body>" + PAGE + "</body></html>")
print("pub", os.path.getsize(OUT_PUB) // 1024, "KB; chapters", len(CH), "; QA", len(QA))
