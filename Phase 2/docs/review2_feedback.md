# Review 2 panel feedback and how Phase 2 addresses it (Step 1.1)

This file records every comment from the Review 2 panel, with one line on how Phase 2 responds. Feedback that changes the scope is decided here, at the start of Phase 2, and not later.

## Panel comments

> **To be filled in by the team.** Enter one row per comment, in the panel's words where possible. The team holds the comments; they are not in the repository.

| # | Panel comment (who / slide) | Phase 2 response | Step | Scope change? |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

## Scope decisions already fixed by the Phase 2 plan

| Decision | Reason | Where |
|---|---|---|
| U-Net stays out of Phase 2 | Even the expanded inventory has only a few usable events, far from the 226 images a U-Net needs. LightGBM is kept, with one comparison model. | Plan, Step 0 |
| The headline evaluation changes from a random split to spatial and temporal hold-out | The random 70/30 split scored 99.1 %, but unseen 10 km regions scored 88.5 % (Phase 1 Step 9). | Steps 6, 10 |
| Counterfactual recommendations are a stretch goal only | They are done only if Steps 1–11 finish on schedule. | Step 11.3 |
| Planner dashboard moves to Phase 3 | It does not fit in the 40 days of Phase 2. | Step 13 |

## Questions likely to recur, and where the answer will come from

| Likely question | Phase 2 evidence |
|---|---|
| "Is 99 % accuracy real?" | Step 1 reproduces it exactly. Step 6 replaces it with spatial-block and temporal hold-out numbers. |
| "Where is the urbanisation / development part?" | Step 2 (yearly built-up record), Step 4 (UICA, ΔUICA), Step 8 (ablation D vs C). |
| "Non-flood labels are not true negatives." | Step 7 PU-learning. |
| "Only one flood event was used." | Step 3 expanded inventory, Step 10 temporal hold-out. |
