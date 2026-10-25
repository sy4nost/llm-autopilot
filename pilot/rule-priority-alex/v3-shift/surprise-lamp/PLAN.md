# Lamp = "stumble on an unexpected word" — PLAN v1 (2026-10-24), exploratory

Alex, 2026-10-24 00:21Z: move towards the unexpected word; measure unexpectedness against the neighbouring words and against the
whole task. Idea (Alex + Claude): the model reads the task voting for each next word before seeing it; a word that got very few
points is a "stumble". Humans notice first (cheap) and check second (expensive); a stumble could be the cheap pre-answer trigger.
No reasoning, no fitting: one forward pass per task, per model.

## Data (no new items)
The 760 items of conftrap-lamp (seed 57, ../conftrap-lamp/sets_cl.json) with the saved fast (DIRECT) answers of Qwen2.5-14B, Phi-4,
OLMo-2-13B (../conftrap-lamp/out_main_*.jsonl). Families hindgen, redef, parcels, bonus, D (traps + controls) and MMLU.
Exploratory: these items were already used; a positive result only earns a frozen test on fresh items.

## Measure
For every token of the task text (the user turn exactly as DIRECT saw it, inside the chat template), surprise s = -log p(token | all
text before it). First 5 tokens of the task are skipped.
- z_task(t) = (s_t - mean s over the task) / sd over the task.
- z_local(t) = (s_t - mean over the 20 neighbours t-10..t+10 without t) / max(sd of those neighbours, 0.5).
- MAIN(item) = max over tokens of min(z_local, z_task): a word that is surprising both against its neighbours and against the task.
- Report-only variants: max z_local, max z_task, max raw s.

## Questions
Q1 (primary). Within traps only: does MAIN separate fast HABIT errors (fast answer = habit option) from fast right answers?
  Cell = model x family, eligible if >= 10 of each. Measure: AUC within cell, mean over eligible cells.
  Rival: low confidence (1 - probability of the chosen option, same fast pass), same AUC.
  Direction is not known in advance: high surprise may mean "strong old expectation -> habit" (AUC > 0.5) or "the model noticed ->
  right" (AUC < 0.5). Both are declared now.
  "Promising" (worth a frozen test on fresh items) if |mean AUC - 0.5| >= 0.12 AND |AUC(MAIN) - 0.5| exceeds |AUC(CONF) - 0.5| by
  >= 0.05 with bootstrap (items within cells, 2000) low > 0 AND same direction in >= 2 of 3 models. Otherwise "no lamp here".
Q2 (sanity, report-only). Traps vs controls: does MAIN find the odd text at all (AUC trap vs control within model x family)?
Q3 (report-only). parcels/bonus: share of items whose most surprising word lies inside the "New RULE"/"New BONUS RULE" sentence.
Also report-only: Q1 with the variants, and recall of habit errors at a 30% flag rate (MAIN vs CONF).

## Anti-fitting
Formulas, window (10), floor (0.5), skip (5), thresholds fixed here before any surprise is computed. Code hashes in PREREG.md before the run.
Prediction (Claude): Q2 clearly yes for D and redef (stated counterfactual numbers), weak for hindgen (nothing odd in the words);
Q1 unclear, most likely weak (the stumble marks the trap, not whether the model then followed it).

# PLAN v2 (after review/plan-review.md; overrides v1 where they differ). Written before any main-run surprise was looked at.
Smoke (16 items, pipeline only, no Q1 cell testable) showed: chat models expect the turn to END after every full question, so the
biggest "surprise" was the fixed answer-format tail. Fixes in code: special tokens get no probability (surprise "given that the text
goes on"); the fixed lines (R_TAIL, "Answer for the TEST case below...", "Return only the answer.") are left out of all statistics.
The GPU pass (modal_sp.py) saves per-token offsets + surprise for every content token, so all choices below are made on the scorer only;
the main GPU pass was launched with modal_sp.py sha 3a73633a... before this v2 and its outputs are not opened until PREREG.md is frozen.
- S4 words, not tokens: a whitespace word gets the summed surprise of its tokens; words without a letter/digit dropped. Skip = the
  first word only (B2). z_local uses +-10 neighbour words without the word itself, sd floor 0.5; with < 6 neighbours z_local = z_task.
- B1 Q1 measure = WITHIN-TEMPLATE AUC: only (habit error, fast right) pairs from the same template, pooled within the cell. Templates:
  conftrap cluster (redef template, D hook, parcels test case, bonus set); hindgen traps by key (good bet lost / bad bet won).
  Eligible cell: >= 30 informative pairs and >= 2 templates. CONF (1 - p of the fast answer) and LEN (number of words) scored the same way.
- S1/S2 strength = mean over eligible cells of |AUC - 0.5| (directions may differ by cell). Bootstrap 2000, clusters then items within a
  cell, signed difference with each cell's direction fixed from the full sample. "Promising" needs ALL: MAIN strength >= 0.12; beats CONF
  by >= 0.05 with low > 0; beats LEN by >= 0.05 with low > 0; same direction in >= 2/3 of eligible cells; same direction in >= 2 models
  (OLMo counts only with >= 2 eligible cells).
- B4 report-only SPAN = largest word surprise inside a critical span fixed per family (D: the stated number; redef: the sentence before
  "Q:"; hindgen: the outcome word of the test question; parcels/bonus: the New RULE sentence). This is an ORACLE position (the model
  does not know where the change is); it shows whether the stumble exists where we know it should. Also: where MAIN's word lands.
- B3 Q2 (traps vs controls) is a pipeline check, meaningful only for D and hindgen; redef has no controls; in parcels/bonus the
  controls carry the same New RULE sentence. Prediction v1 on redef Q2 withdrawn.
- S6 report-only gate on D and hindgen (traps + controls): flag 30% by MAIN / SPAN / CONF / LEN; recall of fast errors and accuracy of
  "reason only where flagged" vs never / always (thought answer = forced choice after reasoning, saved). CONF needs the same one
  forward pass, so the surprise lamp has no compute advantage over CONF, only a possible information advantage.
- Also report: Spearman MAIN vs CONF within traps.
Prediction (Claude, v2): Q1 "no lamp here"; MAIN mostly lands on names, numbers and structure words; SPAN separates traps from controls
in D clearly.
