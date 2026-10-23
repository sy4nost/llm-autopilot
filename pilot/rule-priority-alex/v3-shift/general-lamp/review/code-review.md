# Code review: general-lamp (make_gl.py, modal_gl.py, analyze_gl.py) against PLAN.md v2.1 / PREREG.md

Reviewer: independent adversarial code review, 2026-10-18. I did not write this code. The files are frozen (sha256 values match PREREG.md) and none of them was modified. Analysis was local and read-only: `analyze_gl.py --selftest` plus small `python3 -I` scripts over sets_gl.json, meta_smoke.jsonl, scores_smoke.json, data/*.jsonl and ../indep/data/redefine-math.jsonl. Nothing was run on Modal.

Severity scale: **BUG-affects-verdict**, **BUG-report-only**, **NIT**. Line numbers are real file lines.

## Findings

### 1. BUG-affects-verdict (deviation from the plan). NeQA validity also strips "n't" contractions, so twice as many items are kept as the plan allows
- **Where:** modal_gl.py:32-44 (`unnegate`).
- **Plan v2.1:** "замена ровно одного « not »; если «not» не ровно одно, вопрос исключается".
- **Code:** counts `\bnot\b` and `\w+n't` together. When there is exactly one hit, it removes the "not" or rewrites isn't→is, aren't→are, and so on.
- **NeQA data** (data/NeQA.jsonl): 156 questions have exactly one "not" and no contraction. 144 have exactly one "n't" (isn't 106, aren't 29, wasn't 7, weren't 2) and no "not".
- **Effect on the test split:**
  - the plan-literal rule keeps **141** test items;
  - the code keeps **270**.
- **Why this can flip the verdict.**
  - P-neqa is a test family in both directions.
  - P-quote will almost certainly be non-informative (0/30 errors in the smoke).
  - So C->P is informative only if *both* P-redefine and P-neqa reach ≥30/≥30.
  - At the smoke error rate (7/29 ≈ 24%), the plan-literal subset expects about 34 errors, which is right at the threshold. The code's set expects about 65.
  - The extension is reasonable in spirit, but neither the plan nor the PREREG declares it.
- **Fix** (no code change needed for the main run; everything can be recomputed locally):
  - NeQA is test-only in both directions. It is a single cell ("all"), so the per-family centring constant does not change within-cell AUC.
  - Recompute the C->P and P->C tables and bootstrap with P-neqa restricted to the plan-literal `src_i` subset. Use per-item scores (`scores_main.json`) and labels (`meta_main.jsonl`), and reuse `fam_auc`/`boot_dir` from analyze_gl.py.
  - Report both. State which one is the verdict of record. The plan-literal one is the pre-registered rule. If the two verdicts differ, say so explicitly.

### 2. BUG-affects-verdict (deviation; can change the chosen layer). LOFO layer selection averages families without the C-rule/C-parcels 0.5 weights
- **Where:** analyze_gl.py:161, `res[L] = float(np.mean(folds))`.
- **Plan:** "Везде, где семьи усредняются (среднее П → С, критерий «нет слепой семьи»), каждая из них весит 0.5". The LOFO target is a mean over held-out families. In C->P that means C-rule 0.5, C-parcels 0.5, C-bonus 1.
- **Code:** the LOFO mean is unweighted. The fallback branch (164) does use `mean_fams`, so the two branches are inconsistent.
- **Effect:** the layer choice feeds every C->P criterion. The probability of a different layer is moderate when the per-layer curves are flat.
- **Fix** (for a sensitivity analysis): collect `{f: auc}` per layer and use `mean_fams`. Re-run with `--analyze-only 1` and a patched *copy* of the analysis source. The states are on the volume. Report whether the chosen layer, and hence the verdict, changes.

### 3. NIT (could move C2 vs B4 slightly). B4 is not "the same lamp": its classes are unweighted
- **Where:** analyze_gl.py:251, 254 (`np.ones(...)` as sample weights).
- **Plan:** "та же лампочка", and the lamp uses balanced classes. MMLU is about 77% correct, so B4 is fitted unbalanced.
- **Effect:** usually a small change in AUC, but it is a deviation in the baseline the lamp must beat by 0.05.
- **Fix:** use `fam_weights(mm_tr)` (a single family, so the classes become balanced) in a sensitivity run.

### 4. NIT. Within a bootstrap replicate, qualifying cells are re-derived and NaN families are silently dropped
- **Where:** analyze_gl.py:220 (`fam_auc(sub, ...)` with `cells=None`), 145-146 (`mean_fams` skips NaN), 305 (`nanpercentile`).
- **Effect:**
  - A replicate in which a borderline cell falls below 10/10 changes which cells count.
  - A replicate in which a family loses every qualifying cell is averaged over fewer families.
  - Both widen or shift the CIs, especially the lower bound of lamp−B differences near 0.
- **Fix:** fix the qualifying cell set from the point estimate. Count and report replicates in which any family is NaN.

### 5. NIT (deviation in an edge path). OOF and fallback folds group P-redefine by item, not by template
- **Where:** analyze_gl.py:172. `grp` for public items is the item index (modal_gl.py:68).
- **Plan:** "деление по наборам или шаблонам".
- **Effect:** only the 5-fold fallback and the criterion-4 OOF threshold in P->C. P->C will almost surely stop at the P-quote rule anyway.
- **Fix:** for P-redefine, use `cell` as the group.

### 6. NIT. Criterion-4 threshold pools positives across training families without family weights
- **Where:** analyze_gl.py:317.
- **Effect:** the 80% recall point is dominated by the family with the most habit errors (likely C-rule B/C arms). Positives in 100%-habit cells (e.g. C-parcels A60|T2) are centred to about 0 and pull the threshold down. This is consistent with a literal reading of the plan, but it is worth reporting the per-family recall at `thr`.

### 7. Design note, not a code bug. In C->P, criterion 4 is mostly a format-shift test
- Training rows are in chat format (last token "assistant\n"). MMLU is F3 raw text (last token of "Answer:").
- MMLU is centred by the training-pool mean, as the plan requires (analyze_gl.py:318-320, implemented correctly). So the lamp score on MMLU includes w·(format offset).
- The smoke shows this offset is large: `lamp_uncentred` ≈ +71 on F3 public items (scores_smoke.json).
- This is what the plan intended (the Zhao lesson). But if C->P fails only on C4, the report should say that the failure is foreign-format firing, not poor habit detection.

### 8. NIT (ambiguity to state in the report). How the "twins" count in P->C
- With weights of 0.5, C-rule and C-parcels count as one family in averages.
- The code still counts them as two for the "≥2 informative test families" rule (analyze_gl.py:298). It also applies criterion 3 to each of them separately (315).
- The plan does not say otherwise. If P->C ever becomes informative only through C-rule + C-parcels, flag it.

### 9. BUG-report-only. The PREREG promise "results without the 30 smoke items" is not implemented
- **Where:** PREREG.md ("В отчёте будут результаты без этих 30 вопросов"). analyze_gl.py has no such section.
- **Fix:** compute post hoc from scores_main.json. The smoke items are the first 30 non-aux test items of each public family in `src_i` order: P-redefine src_i 0..~33 in t0, and the first 30 test `src_i` of NeQA and quote.

### 10. BUG-report-only. The "without previously scored items" report is minimal
- **Where:** analyze_gl.py:326-328.
- **Plan:** "Результаты даются дважды: со всеми вопросами и без этих 200".
- **Code:** prints only the lamp AUC, and only for informative families. There are no baselines, CIs or criteria.
- **Fix:** rerun the direction table and criteria 1-3 on the subset with `seen_before=False`. Everything needed is in the per-item scores.

### 11. BUG-report-only. The P-native "twin pair" is trained on all three C families, not on C-rule
- **Where:** analyze_gl.py:341-345.
- **Plan:** "пара C-rule → P-native".
- **Fix:** train on C-rule only, or relabel the report line.

### 12. BUG-report-only. Within-kind results are missing
- The plan's report-only list includes "внутривидовые результаты" (C→C, P→P). Only the LOFO per-layer numbers are printed, which is partial.

### 13. BUG-report-only (plan item 7, reproducibility). Per-item output is missing what is needed to re-check criterion 4 and B4
- **Where:** analyze_gl.py:289.
- `per_item` holds only test-family scores. It does not include:
  - lamp scores on held-out MMLU;
  - the OOF training scores;
  - `thr` and `mu_pool`-centred MMLU scores;
  - B4 scores on MMLU.
- So C4 and B4 validity cannot be recomputed from the returned files.
- **Fix:** a Modal-CPU `--analyze-only` dump of those arrays.

### 14. NIT. The trap-situation control excludes C-rule and is unbalanced
- **Where:** analyze_gl.py:347-348.
- C-rule has no untouched cells, so excluding it is defensible. Say so in the report.

### 15. NIT. The aux validity checks ignore option mass on the aux prompt
- **Where:** analyze_gl.py:50-53.
- "Knows the quote" is decided by a pairwise preference even when both options are improbable. In the smoke, quote-aux mass has median 0.41 and minimum 0.015. NeQA-aux mass is about 1.
- The plan does not require a mass check. Report the aux-mass distribution among valid items.

### 16. NIT. modal_gl.py:215 has no row-count assertion
- The states are concatenated from whatever `states_*.npy` files are in `/cache/gl/{tag}`.
- Stale extra chunks from an earlier, longer run would be appended but never indexed, because `row_index` mapping is preserved. So it is harmless today.
- Add `assert len(S) == rows[-1]['n_rows']` for safety.

### 17. NIT. Other small items
- B3's layer-0 mean (modal_gl.py:140) includes the chat-template tokens for generated families. The offset is constant and harmless.
- `USED` contains 20261018, which is not in the plan's list. Harmless for S=88.
- `make_native` builds once with its default seed at import. That build is discarded, so it is harmless.

## Smoke check (meta_smoke.jsonl / report_smoke.txt / scores_smoke.json)
- **Consistent:**
  - 302 items, 242 state rows, `row_index` contiguous;
  - every NeQA/quote base row has a matching aux row;
  - `seen_before` ≈ 2/3 of the first 30, as expected for 200/300;
  - generated families have option mass of about 1.0 and use their own chat format;
  - MMLU accuracy 0.77 with mass about 1;
  - the cache-vs-full maximum difference is 0.10.
- **Worth noting:**
  - **B4 per-layer CV AUCs of 0.11–0.35** with n=30. This is a tiny-n leave-group-out artefact. In main, check that they are above 0.5. If they stay below 0.5, the subject-grouped CV is anti-learning.
  - **All 30 smoke redefine items come from template t0** (view-num). Under F3 that template gives 11/30 errors, against 0% under the old chat protocol. This shows F3 changes behaviour a lot.
  - **P-quote: 0/30 errors**, so P->C will be non-informative by rule.
  - **P-redefine in F3 keeps only 29% on dev** (mass ≥ 0.5). That leaves roughly 235 of 810 test items across 9 templates, so P-redefine itself may fail ≥30/≥30. C->P may then be non-informative regardless of findings 1-2.
  - smoke.log starts with a stray `collect 100 / 6750` block. That is the main item count, which also matches main.log. It does not affect main (tag `main`), but it is confusing.

## Checked and correct
- **Habit targets.**
  - C/P-native: picked `old` with `old != key`.
  - Public: pick ≠ gold, which is valid because every item has exactly 2 classes. Redefine's non-gold class is the standard-math answer on all 900. Quote has `answer_index=1` everywhere.
- **Validity checks.** NeQA-aux and quote-aux validity are in the right direction. Invalid items are excluded from both classes.
- **Option scoring.** Summed multi-token log-probs, with the cached continuation indexed correctly.
- **Format.** F3 was selected by the max-min median rule.
- **Seen-before marking** matches the recipe in public/ and conflict/ (`Random(0)` shuffle, first 200).
- **MMLU exclusions** match the blockex (first 10 per subject), watch_public (seed 185, 500) and conflict (seed 0, 200) recipes. The 1000/500 split is correct.
- **Dev/test split.** Redefine: 10 per template; the templates are contiguous blocks of 100 with a fixed answer_index (verified on ../indep/data/redefine-math.jsonl). NeQA/quote: 10%, seed 12345. Dev items are never scored in main.
- **Sizes.** The main item count is 6750 = 3360 generated + 810 + 540 + 540 + 1500. Families, cells and groups match the plan (90/80/160/90 sets).
- **Cell centring** on train and test, with standardisation from training data only.
- **Family/class balancing** in training.
- **FAM_W** in direction means, the B1 sign and the fallback.
- **Informative-family rule** (≥30/≥30 over qualifying cells) and the "fewer than 2 means non-informative" rule.
- **Bootstrap units** (sets / items within template / items), paired across detectors.
- **Criteria.**
  - C1-C3 are as specified.
  - C4 uses an OOF threshold, training-pool-mean centring of MMLU, and held-out correct MMLU only.
  - B4 layer is chosen by subject-grouped CV.
  - The P->C P-quote rule is correct.
  - The verdict table matches the plan.

## Bottom line
I found no bug that silently corrupts targets, splits, centring or statistics. Two undeclared deviations can change the verdict, and both can be recomputed after the run without touching the frozen files:
- NeQA keeps "n't" items: 270 against 141 plan-literal. Recompute locally from the per-item scores.
- LOFO layer choice ignores the 0.5 twin weights. Needs an analyze-only sensitivity run on Modal CPU.

Report both alongside the frozen verdict.
