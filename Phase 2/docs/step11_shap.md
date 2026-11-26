# Step 11: SHAP explanations

Script: `scripts/11_shap_explanations.py` · numbers: `outputs/metrics/step11_shap.json` · figure: `outputs/maps/step11_shap_global.png`

**Scope.** SHAP explains the *models*, which were trained on big riverine flood labels (radar, radar + NDEM). It shows what the models rely on to place those floods. It is **not** evidence about waterlogging; that is Step 10b.

**Models.** TreeSHAP in log-odds, explained on 39,270 land cells (2018 reference flood):
- **M1:** the chosen riverine model (PU bagging, set C, radar labels, 25 bags averaged).
- **M2:** unconstrained D, radar + NDEM labels.
- **M3:** the Step 9.0b scenario model (monotone, no land cover).

## 11.1 What the models rely on (share of mean |SHAP|)

| Group | M1 | M2 | M3 |
|---|---|---|---|
| Terrain & hydrology (HAND, elevation, distances, TWI…) | **66 %** | **59 %** | **63 %** |
| Event rainfall | 7 % | 23 % | 25 % |
| Land cover | 13 % | 4 % | – |
| Annual rainfall (location proxy) | 5 % | 7 % | 8 % |
| Roads | 4 % | 2 % | 3 % |
| **Development (paving, UICA, ΔUICA)** | **5.3 %** | **4.0 %** | **0.6 %** |
| Development share in 2030 hotspots / Kochi core | 4.5 % / 9.9 % | 4.3 % / 4.3 % | 0.4 % / 0.6 % |

**Direction of the development effect.** Spearman rank correlation between a feature's value and its SHAP: positive means more paving → higher risk.

| | M1 | M2 | M3 |
|---|---|---|---|
| Paved fraction | **−0.91** | **−0.80** | not used (SHAP 0) |
| UICA ratio | – | **−0.66** | **+0.61** |
| Land cover "built", mean SHAP on built cells | −1.62 | −0.31 | – |

## 11.2 Places and the 2017 → 2030 change

- **Aluva, North Paravur, Kakkanad, Kolenchery:** the scores are driven by HAND, elevation and distance to the river. Development contributes about 0.00–0.01 log-odds (M3).
- **Ernakulam core:** low riverine score, driven by land cover (built), distance to the major river and event rainfall.
- **2030 trend hotspots (M3, 5,000 cells):** mean risk 0.572 → 0.576. 57 % of cells rise. All of the change comes from the upstream-paving features (UICA); local paving is unused by M3.

## Reading

1. **The models are physically sensible:** low land close to rivers floods. Terrain and hydrology carry about 60–66 % of every model.
2. **Development has a small, measured share in predicting big floods:** 4–5 % in the unconstrained models and 0.6 % in the constrained one. This puts a number on the Week 3 and 4 finding that development hardly changes where riverine floods occur.
3. **The unconstrained models learned the opposite of physics.** More paving lowers their predicted risk (ρ = −0.91 and −0.80). This is the radar-label bias, now visible in SHAP. In M3, the only development signal left (UICA ratio) has the physical sign (ρ = +0.61). This confirms the 9.0b fix.
4. **The annual-rainfall location proxy is used at 5–8 %:** noticeable, but not dominant. `aspect_deg` (7.6 % in M2) is another likely location stand-in; both are flagged.
5. The model gives the 2030 hotspots only a tiny riverine-risk increase. Their development importance is in the water volume (Step 9), not in riverine susceptibility.
