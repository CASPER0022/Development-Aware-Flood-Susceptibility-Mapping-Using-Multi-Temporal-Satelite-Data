# Likely panel questions and answers (Step 13.5)

Same 30 questions as the Phase 2 study notebook (`outputs/presentation/Periyar_Phase2_Study_Notebook.html`).

**1. Your title says development-aware. Where is the evidence that development increases flooding?**

Three parts. Physics: development adds +43 % runoff on newly paved land in frequent storms and puts 113 → 190 km² of flood-prone land under more water. Model: with NDEM labels, UICA improves prediction on paved land (+0.037). Observation: real waterlogging spots sit below more paved land (57th–61st percentile). The big river floods show no change, which the physics predicts. Proving the waterlogging effect with observations is Phase 3.

**2. Why did UICA add nothing in the first ablation?**

The radar flood labels are 0.1 % built-up. The model never saw a town flood, so a feature about paved catchments had nothing to explain. UICA got 0.8 % of the model's gain.

**3. Isn't 0.985 suspiciously high?**

It is flood vs all land on a new year and new place. The floods are mostly the same kind (river-side, low land), which is easier than city waterlogging. Against radar + NDEM truth the score is about 0.86.

**4. Why not a random train/test split?**

Neighbouring cells are nearly identical; a random split lets the model memorise neighbourhoods (99.9 %). Our strict test hides both the year and the area.

**5. Why is the temporal test alone not enough?**

The same floodplain floods every year, so a model that remembers where past floods were scores 0.99 on a new year.

**6. What is PU-learning and did it help?**

It treats 'not seen flooded' as unknown. PU bagging improved the strict AUC from 0.975 to 0.984, but could not fix Kochi because there were no urban flood examples.

**7. Why did you add a monotone rule? Isn't that forcing the answer?**

It only forbids a physically impossible answer (paving making water soak in better). Without it, 94 % of developed plots came out 'safer'. With it, the model's response is flat, so it does not force a positive effect either.

**8. Why accept a 0.011 accuracy cost when your limit was 0.01?**

The lost accuracy came from the label bias itself (2024 truth is radar-only and rewards 'built = dry'), and the alternative gives wrong scenario maps. We state this openly.

**9. Why the SCS curve number method?**

It is the standard method used for drain design (USDA TR-55), needs only paving, soil and rainfall, and gives the effect size without flood labels.

**10. Which soil did you assume?**

One soil group (C) for the basin, because no detailed soil map can be downloaded. Groups B and D change the numbers about 2×, not the pattern.

**11. Why does development matter so little in the big floods?**

Soaked monsoon soil already sheds 82 % of the rain; paving it changes little (+0.2 %). In ordinary storms unpaved soil sheds only 20 %, so paving matters.

**12. What is a hotspot?**

Land the model already rates High or Very High for flooding that also gets at least 5 % more water in a frequent storm because of development.

**13. How sure is the 2030 number?**

It depends on how fast building continues: 299–434 km² of land with more water (GHSL projection vs our trend). Hotspots 122–190 km².

**14. Why does most extra water come from upstream?**

Water from paved land flows downhill, so a place receives the runoff of all paved land above it. That is exactly what UICA measures; 87 % of the extra water at affected places comes from upstream.

**15. What did the natural experiment show?**

No change in big floods after correcting with a placebo (−0.05 points, range −0.71 to +0.58). The first estimate (+0.42) was fake: the placebo showed the same.

**16. Why use a placebo test?**

It checks whether the method finds an 'effect' where none can exist. It caught a fake result that we would otherwise have reported.

**17. Why couldn't you test local paving?**

Radar never sees floods on paved land, so newly paved cells are 'never flooded' before and after (floor effect). The result is meaningless.

**18. Where did the waterlogging data come from?**

Fifteen places named in news and official reports (2019, 2024 events), located with OpenStreetMap. Each has a source link in the CSV.

**19. Isn't 15 spots too few?**

Yes, it is a first hint, not proof (p about 0.08–0.19). Phase 3 needs corporation complaint records and more events.

**20. Why doesn't the 2017–2025 growth layer match the waterlogging spots?**

The reported spots are in the old city, paved long before 2017. Their waterlogging comes from existing paving and drainage. New growth areas have few reports yet.

**21. What does SHAP show?**

Terrain and hydrology drive about 60–66 % of decisions; development 4–5 %. Development is a small factor for big river floods, as expected.

**22. Why does the Phase 2 map rate Kochi low?**

The labels hardly contain urban flooding, so the river-flood model rates built-up land lower. Every map states that built-up risk is a lower bound.

**23. Why LightGBM and not U-Net?**

Four flood events are far too few for a CNN. XGBoost also did no better than LightGBM, so the algorithm is not the limit.

**24. Did tuning help?**

Barely: all 18 settings scored 0.993–0.994. The training-test gap moved from 0.0071 to 0.0065.

**25. Why drop the 2019 and 2024 NDEM maps?**

Their outlines are rough shapes (about 23–26 vertices per km) that look like random land; 2018 and 2020 are traced in detail (177–329).

**26. Is the mean annual rainfall still a problem?**

It is a mild location stand-in: removing it costs 0.007 on the strict test and SHAP gives it 5–8 %. We keep it and flag it.

**27. What are your probabilities?**

After prior correction: 'chance this cell is in the confirmed flood of an event like these'. Mean 0.38 % predicted vs 0.37 % observed on 2020/2024. They depend on event size.

**28. What does Phase 3 add?**

Work Package 1: prove the development effect with real data (Kochi waterlogging records, radar detection between buildings, paddy and wetland loss, more events). Then a planner dashboard and the report.

**29. What is the patent angle?**

The UICA-based development-aware method and the scenario workflow. Keep the code private until a provisional filing; the repository is public.

**30. What would you do differently?**

Collect urban waterlogging records from the start, because the satellite labels could never show the development effect that the title is about.
