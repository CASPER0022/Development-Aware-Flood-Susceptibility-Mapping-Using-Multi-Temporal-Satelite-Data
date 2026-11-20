# Step 9.0b: Physically constrained scenario model

Script: `scripts/09b_constrained_model.py` · numbers: `outputs/metrics/step9b_constrained_model.json` · model: `outputs/models/model_v2_scenario.txt` · figure: `outputs/maps/step9b_constraint_checks.png` · shared code: `common.ndem_label_dataset()` (cached Step 9.0a labels)

## Idea

Paving cannot make water soak in better. So the scenario model is not allowed to lower risk when paving rises:
- **Monotone constraints** (LightGBM, method "advanced"): risk is non-decreasing in `impervious_frac`, `uica_km2`, `uica_ratio` and `duica_since_2017_km2`.
- **Land-cover class removed** (`-LC`). Its "built" class is where the model hid "built = dry".

**Base model:** the Step 9.0a choice (set D, radar + NDEM labels, standard design, tuned settings, fixed folds).

## 1. Stress test: does paving make places safer?

- **Sample:** 100,000 random 2024 land cells that are < 20 % paved.
- **"Development" applied to every cell:**
  - paved fraction → at least 0.6;
  - upstream paved area × 1.5;
  - UICA ratio + 0.2;
  - ΔUICA + 50 % of UICA;
  - land cover → built.

| Model | Cells that get **safer** | Cells that get riskier | Mean change |
|---|---|---|---|
| D (unconstrained) | **94.0 %** | 6.0 % | −0.070 |
| D − land cover | 95.1 % | 4.9 % | −0.056 |
| D + monotone (land cover kept) | 52.9 % | 20.4 % | −0.045 |
| **D − land cover + monotone** | **0.0 %** | 26.2 % | +0.001 |

**The bug is large.** The unconstrained model says paving makes **94 % of places safer**. That is the radar-label bias, "built / paved = not flooded":
- Constraining the paving features alone is not enough (53 % still safer), because the land-cover "built" class carries the bias.
- Only constraints **plus** removing land cover give zero.

## 2. Response curves (2,000 cells, paving swept 0 → 1)

Number of drops along each curve (each drop means risk fell as paving rose):

| Model | Drops vs local paved fraction | Drops vs UICA ratio |
|---|---|---|
| D | 8 | 7 |
| D − land cover | 10 | 6 |
| D + monotone | 0 | 0 |
| D − land cover + monotone | 0 | 0 |

- **Unconstrained curves fall steeply:** risk drops by about a third as paved fraction goes 0 → 0.05.
- **Constrained curves are flat.** For UICA ratio there is a small rise (0.133 → 0.135) between 0 and 0.1, then flat.

## 3. Accuracy cost (AUC vs radar + NDEM truth)

| Model | Spatial CV | Strict 2020: all land | Strict 2020: built | Strict 2020: paved ≥ 20 % | Strict 2024: all land (radar-only truth) |
|---|---|---|---|---|---|
| D | 0.921 | 0.864 | 0.757 | 0.732 | 0.980 |
| D − land cover | 0.917 | 0.856 | 0.749 | 0.708 | 0.962 |
| D + monotone | 0.921 | 0.864 | 0.745 | 0.706 | 0.979 |
| **D − land cover + monotone** | 0.916 | **0.854** | 0.744 | 0.710 | 0.959 |

- The constraints themselves cost almost nothing (D vs D + monotone: −0.001).
- Most of the cost comes from **removing land cover**: −0.011 on strict 2020 and −0.021 on 2024.
- The 2024 truth is radar-only, so it *rewards* "built = dry". Part of the lost accuracy is the bias itself.

## Reading

1. **The unconstrained model must never be used for scenarios.** It would report that development reduces flood risk almost everywhere.
2. **The constrained model is safe, but it says paving has almost no effect** (mean +0.001, flat curves). The constraint can only remove the wrong direction; it cannot create a positive effect the labels never showed.
   - So **the ML model cannot supply the size of the development effect.** That has to come from physics (Step 9.0c, extra runoff) and the natural experiment (Step 10), exactly as the revised plan frames it: *ML says where floods happen; physics says how much development adds.*
3. **The cost is just over the 0.01 rule set in the script** (0.011 on strict 2020), so the script flags it for review.

## Decision (subject to the guide's agreement)

- The scenario model is **D − land cover + monotone** (`model_v2_scenario.txt`).
- The extra 0.001 over the rule is accuracy that came from a label artefact, while the unconstrained alternative gives physically wrong scenarios (94 % of cells "safer").
- In Step 9, the scenario Δ-maps from this model show **where** development raises risk. The **magnitude** comes from Step 9.0c.
