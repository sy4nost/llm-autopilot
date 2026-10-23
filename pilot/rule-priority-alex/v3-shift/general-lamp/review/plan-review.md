# Review of general-lamp/PLAN.md (draft 2026-10-18)

Reviewer: independent session. I ran no Modal or GPU jobs. I read the generators, the earlier results, and the local data (with `python3 -I`, read-only). Scripts and their outputs are in this folder:

- `cell_auc.py` → `cell_auc_out.txt`: expected positives per family, and the AUC of a predictor that knows only the prompt cell (arm × probe type). It uses earlier error rates.
- `check_redefine_habit.py` → `check_redefine_habit_out.txt`: is the non-gold class of redefine-math always the habitual answer?
- `redefine_template_auc.py` → `redefine_template_auc_out.txt`: error rate per template on the 200 items scored in `public/`, plus the AUCs of template-only, length and margin predictors.
- `option_mass.py` → `option_mass_out.txt`: how much probability 14B puts on the answer options under the `public/` scoring protocol.

## Bottom line

As written, the plan can produce **a PASS that reflects surface features of the prompt**. In the other direction it can produce **a FAIL that tells us nothing**, because the public targets are barely tied to what the model would actually say. Five problems are blockers:

1. The plan says IMPORTANT UPDATE appears in traps and untouched cases alike. That is false for the selected arms.
2. In the C-families and P-native, "habit error" is almost the same thing as "which prompt cell is this". A predictor that knows only the cell reaches AUC 0.87–0.98 (estimated from earlier runs). Within-family AUC does not remove this.
3. Under the earlier public scoring, 14B puts almost no probability on the redefine-math options: the median log P of either option is −34. On NeQA and quote the options are also usually not the model's real next token. The "answer" being predicted is therefore not the model's behaviour.
4. P-native is a structural twin of C-rule, so it is not an independent test family. It sits in the test set of one direction and the training set of the other.
5. Key quantities are not fixed: public and MMLU sample sizes, the MMLU train/false-alarm split, the minimum number of families left after the ≥30 rule, the sign of the confidence baseline, and how multi-token options are scored. Each one leaves room to tune after seeing the data.

| # | Point | Verdict |
|---|---|---|
| 1 | Is the target well defined in each family? | **BLOCKER** (public elicitation; C-rule has no untouched cases) |
| 2 | Confounds and controls | **BLOCKER** (note presence, cell identity, template identity; baselines too weak or unsigned) |
| 3 | Expected positives; the ≥30 rule | **CONCERN** (quote may be dropped; no minimum family count) |
| 4 | Layer selection, regularisation, leakage | **CONCERN** (CV grouping and objective unspecified; C-rule/P-native twin) |
| 5 | Criteria and thresholds | **CONCERN** (0.70 is below the surface ceiling; false alarm only reported; MMLU baseline is weak) |
| 6 | Room to tune after seeing data | **CONCERN** (several free choices listed below) |

---

## 1. Target definition per family — BLOCKER

**C-rule (match/make_match.py).**
- All four probes D1–D4 are changed cases under the new rule: D1 H→P, D2 P→H, D3 H→R, D4 R→H (`make_shift.py` docstring and `key()`). So in the change arms **there is no "untouched" case**.
- The only negatives where the old answer is correct are `A_nochange_60`. Those prompts have **no update note** and their key is the old rule.
- The plan's table and its "traps of experience itself" section say C-rule has untouched cases with the note. That is wrong.
- D2/D4 do count as habit errors (old label). Their rate is low (10–33% with sampling), so most positives are D1/D3.

**C-parcels / C-bonus (newtasks2/make_new2.py).**
- Untouched cases (S1, S2) can never be positive, because old == new for them.
- `N60` has **no update note**: its key is the old answer, so it is always negative.
- Parcels trap T1/T2 are the same two fixed tuples in every seed (the audit already noted this).
- With greedy or argmax answers, the trap cells are almost all-or-nothing:
  - A60 T: 80% (parcels) / 90% (bonus) old answers;
  - E_nocopy T: 88% / 90%;
  - E_copy T: 0% / 5%.

**P-native (change_0, D1–D4, 120 prompts).**
- Earlier rates: D1 9% right, D3 16% right, D2/D4 100% right. So essentially **every positive is a D1/D3 prompt**, and every D2/D4 prompt is negative.
- D1/D3 differ from D2/D4 in the test-case text: pedestrians PRESENT vs ABSENT.
- The pull here is a specific safety prior ("people on the crossing → stop"), not a general "old answer" habit. On D2/D4 the old answer agrees with common sense (GO / CHECK when nobody is there), yet the model gives it 0% of the time.

**P-redefine.**
- The habit is well defined. `check_redefine_habit.py` confirms the non-gold class is always the standard-math answer in all 900 items: 3 for π, 2 for e, otherwise the computed first digit.
- The task has 9 templates of 100 items each, and `answer_index` is fixed per template.
- **BLOCKER: elicitation.** Under the `public/` protocol (chat template, summed log-prob of each class), the median log P(any option) is **−34.4**. Not one item gives the options more than 0.5 of the probability.
- So the "model's answer" is the argmax between two tokens the model would almost never produce. The state at the last token encodes what the model really plans to say ("The …"), not this choice.

**P-quote.**
- The habit (the famous ending) is plausible, but the data are not local, so I could not verify the classes. They are multi-token, and summed log-prob favours the shorter class.
- Option mass: median log P −7.8; only 11% of items give the options more than 0.5.

**P-neqa.**
- Option mass: median log P −3.5; 7% of items above 0.5.
- At 61.5% accuracy on a 2-way task, many errors may come from not knowing the fact rather than from ignoring "not".
- NeQA has no stated rule change, so it is the weakest example of "habit".

**Fixes.**
- (a) Elicit answers so that the options are the model's real next token. For example: prefill the assistant turn, end it with `A:`, or add "Answer with only the digit / only A or B".
- (b) Pre-register a filter on option mass, e.g. keep items with P(options) ≥ 0.5. Report how many items are kept, and check that rule in the 12345 smoke, not on the test items.
- (c) Compute the same option-mass statistic for the synthetic families.
- (d) Fix how multi-token options are scored: length-normalised or first token. Assert that the labels have distinct first tokens, as the earlier runs did.
- (e) NeQA: also run the un-negated question (cost +n prompts). Count an item as a habit error only if the model answers the un-negated question correctly. Do the same for quote: check the model actually completes the original quote.
- (f) Correct the plan text: C-rule has no untouched cases, and the note is absent in nochange/N60.

## 2. Confounds a within-family AUC does not remove — BLOCKER

**a) Note presence.**
- Within C-rule and C-parcels/C-bonus, all `nochange`/`N60` prompts lack the note and are all negative.
- A detector for "the note is present" alone gives about AUC 0.67 in C-rule, and more in combination with (b).

**b) Cell identity (arm × probe), estimated with `cell_auc.py` from earlier rates.** A predictor that only knows the cell gets:

| family | n | expected positives | cell-only AUC | cell-only AUC, traps only |
|---|---|---|---|---|
| C-rule | 480 | ~125 | 0.87 | 0.81 |
| P-native | 120 | ~49 | 0.92 | 0.92 |
| C-parcels | 320 | ~67 | 0.98 | 0.90 |
| C-bonus | 320 | ~74 | 0.98 | 0.90 |

- These cells are readable from the text: the test-case wording, whether an exact copy is in the history, whether a PERIOD 2 copy is present.
- In C-rule the history composition (A vs B vs C) decides the outcome: 93% vs 38% vs 12% old answers. A probe trained there learns "an exact copy of the test case is in the history". That is a mechanism-specific feature and has nothing to transfer to public tasks, which gives an uninformative FAIL for C→P.
- In P→C, a probe that has learned "this prompt is a trap" will score highly on the C-families for surface reasons, which gives a misleading PASS.

**c) Template identity in redefine** (`redefine_template_auc.py`, 200 earlier items, 62 errors):
- Error rate by template: π "+" 0.89, e 0.61, π mod 0.50, π number 0.30, the "view" templates 0.00–0.21.
- **A template-only predictor gets AUC 0.85.**
- In quote and NeQA, topic, quote fame and class length may play the same role. Prompt length runs the other way (AUC 0.29, i.e. 0.71 if inverted), so length is also a usable surface cue.

**d) The plan's own defence.** It says the label words and the note "differ between sets / are absent in public". That does not cover (b) or (c), which work within a family.

**e) The confidence baseline has a fixed sign.**
- "Small margin = fires" works on redefine (AUC 0.83 for margin alone).
- In the C-families, habit errors are confident copies, so the same baseline may sit below 0.5. Beating it by 0.05 is then trivial.

**f) The MMLU generic-error probe is a weak control.**
- MMLU is a different format (4 letters, knowledge errors). Its direction will probably transfer poorly to every family, so criterion 3 is almost free.
- Nothing checks that the MMLU probe even works on held-out MMLU.

**Fixes.**
1. **Primary metric: AUC stratified within cells.** Compute AUC within each (arm × probe) cell for synthetic families and within each template for redefine. Pool these with a pre-registered weighting, such as the van Elteren-style weighted average of within-stratum AUCs, using only strata that contain both classes. Keep plain within-family AUC as a secondary result.
   - This is the "same situation, the model gave in vs held out" contrast that the public families measure by design.
   - Problem it exposes: with argmax answers the parcels/bonus trap cells have only about 4–8 negatives each, and P-native D1/D3 has about 5 per probe. These families cannot supply that contrast at their current size (see 3).
2. **Add a lexical / surface baseline** trained with the identical protocol, which the lamp must also beat by ≥0.05. Options: logistic regression on embedding-layer (layer 0) features mean-pooled over the prompt, or bag-of-tokens TF-IDF. Also report a cell- or template-identity baseline.
3. **Add an output-only learned baseline**: logistic regression on sorted option log-probs, entropy and top-1 probability, with a family-balanced fit on the training families. The confidence baseline's direction is then fitted on training data, or reported as max(AUC, 1−AUC) for conservatism.
4. **Add a "trap-situation" probe.** Train it on the training families with label = trap vs untouched, and test it on the target. If it matches the lamp, the lamp is a situation detector.
5. **MMLU generic-error probe:**
   - use disjoint MMLU train and evaluation splits;
   - report its AUC on held-out MMLU and require it to be at least 0.65, otherwise the control is void;
   - report the lamp's AUC on MMLU errors (discriminant validity: a habit lamp should be near 0.5 there).
   - Consider a second generic-error control from non-habit errors in similar formats, such as F0/fresh arms and the non-old wrong labels.
6. **Centre activations per family** (subtract the family mean from features, in training and at test time, without using labels). Weight families equally in training. This removes the format and base-rate direction that ruined the Zhao probe.

## 3. Expected positives and the ≥30 rule — CONCERN

From `cell_auc_out.txt`, assuming earlier rates carry over to argmax answers:

| family | expected habit errors |
|---|---|
| C-rule | ~125 |
| C-parcels | ~67 |
| C-bonus | ~74 |
| P-native | ~49 |
| redefine-math | ~62 at n=200, ~280 at the full 900 |
| quote-repetition | **~28 at n=200 (below 30)**, ~70 at n=500 |
| NeQA | ~77 at n=200; NeQA has 300 items in total, as far as I recall (verify) |

Problems:
- The plan does not give n for public or MMLU. With about 2,260 prompts left after the 1,240 synthetic ones, quote could fall under 30 and be dropped.
- The option-mass filter from point 1 will cut every public family further.
- The rule has **no minimum number of remaining families per direction**. If quote is dropped, C→P becomes P-native (the twin of the training set) + redefine + NeQA, and P→C trains on P-native + redefine only.
- Under the stratified metric, the usable positive/negative pairs are far fewer than the raw counts. Parcels and bonus trap cells have about 4–8 negatives each, and P-native D1/D3 about 5 per probe.

**Fixes.**
- Fix n per family now. Use full redefine (900) and full NeQA, and enough quote items to expect at least 60 errors after the option-mass filter.
- Require **both 30 positives and 30 negatives** per family. Under stratification, require at least 30 of each inside strata that have both classes.
- Pre-register "the direction is uninformative, not a FAIL" if fewer than 2 public P families remain in C→P, or fewer than 2 C families in P→C.
- To create contrast inside trap cells in the C-families, add sets or intermediate arms (e.g. A20, or E with one copy). For P-native, raise N_SETS (e.g. 90–120 sets; the prompts are cheap) or add vocabularies.

## 4. Layer selection, regularisation, leakage — CONCERN

**CV design.** "5-fold CV on training families" is not specified enough:
- Folds must be **grouped by set** (synthetic: one set shares a history across 16 prompts) and by template or question for public items. Otherwise the CV AUC is inflated, and inflated most at the layers that memorise the set.
- The CV objective must be the test metric (stratified within-family mean). A pooled AUC rewards layers that encode family identity.
- Better still, select the layer by **leave-one-training-family-out**, which matches the transfer question.

**C = 0.1.** Fixed in advance, so fine. Still specify: the solver; that standardisation is fitted on training data only; class weights (balanced); family weights (equal); and the feature count (5120 per layer, with about 1,100 training rows in C→P).

**MMLU probe layer.** State whether it uses the lamp's layer or its own CV-best layer. Report both and pre-register one for criterion 3.

**C-rule vs P-native.**
- `make_native.py` reuses `make_match.build()` with identical RNG draws. With the same seed, set i has the same field order, label index, probe tuples and history slots.
- Item-level leakage is small, because P-native is change_0 with no history. But the logical cases and the rule structure are identical, and the positives are the same probe types (D1/D3).
- This is exactly the audit lesson ("test tasks isomorphic to tuning tasks"). It strengthens a paired native-vs-fictional comparison, which this plan does not test, and it weakens any claim of generality.
- **Fixes:**
  - remove P-native from the C→P criterion mean, and remove C-rule from the P→C criterion mean (or keep P-native out of P training);
  - report the C-rule ↔ P-native pair as its own secondary "same structure, other source" analysis;
  - if P-native stays in the criteria, give it a seed different from C-rule.

**Overlap with tuning tasks.** Parcels was built to reproduce the C-rule-style change (`make_new2.py` line 2) and is isomorphic to the head-selection task. Its fixed T1/T2 tuples mean that "new seed" is not "new tasks". Do not count C-rule and C-parcels as two independent families when judging spread.

## 5. Criteria — CONCERN

- **The 0.70 bar is below the surface ceiling.** In the C-families, cell identity alone gives 0.87–0.98, and in redefine template identity alone gives 0.85. If the primary metric stays unstratified, 0.70 means nothing for P→C. Fix: make stratified AUC primary, and add the surface baseline from 2.2 to criteria 2 and 3.
- **Criterion 2 (confidence).** Its result depends on the fixed sign. Fit the sign on training data or use max(AUC, 1−AUC).
- **Criterion 3 (MMLU probe).** Likely near 0.5 everywhere, so the criterion is close to free. Add the validity check from 2.5.
- **Criterion 4 (no family below 0.55).** With 30–50 positives, a point AUC has a standard error of about 0.05, so a real 0.62 can fall below 0.55 by chance. Keep it, but state that it applies to the point estimate, and report the confidence intervals.
- **Both directions required.** The conjunction is an intersection-union test, so it is conservative and needs no multiplicity correction. But P→C is inflated by the twin and by surface cues, while C→P is the scientifically meaningful direction. Requiring both lets an unstratified P→C pass hide a public-only C→P result. **Recommend: the primary result is C→P on public P families only, with the stratified metric. P→C is secondary.**
- **False alarms on MMLU are report-only.** That is precisely how the earlier Zhao lamp failed: it fired on 100% of foreign prompts. A lamp that fires on most of MMLU must not be called "general". Fix: make it a criterion, e.g. false-fire rate on X-mmlu ≤ 30% at the threshold that catches 80% on training families. The threshold should be per family-centred score, set on training data only.
- **Bootstrap.**
  - Specify B and the seed.
  - Resample within each family (sets or templates for synthetic, items for public) and average per replicate.
  - Pair the lamp with each baseline inside each replicate.
  - Note that only 20–30 clusters exist per synthetic family.
- **Multiplicity.** Exploratory outputs (within-kind results, best layer, the trap-situation target) must be labelled exploratory in the report. They must not be used to claim a pass if the primary result fails, which is the same lesson as `watch_all` in the audit.

## 6. Room to tune after seeing data — CONCERN

Free choices the PREREG must close:

1. Sample sizes and item lists for each public family and for MMLU, including the exclusion list of MMLU questions already used: the blockex first 10 per subject, the 500 from public-watch seed 185, and the conflict/public MMLU sets. Also the seed used to draw them.
2. The answer-elicitation format and the option-mass filter (point 1). It must be tuned only on the 12345 smoke or other non-test items, and frozen in the PREREG.
3. The scoring rule for multi-token options: summed, length-normalised or first token.
4. The CV grouping and objective; the MMLU probe's layer; the family weights; and per-family centring.
5. The fallback when a family is dropped (minimum family count).
6. The sign or orientation of the confidence baseline.
7. **Output.** The plan returns only summary numbers. That stops peeking, but it also stops an audit like `audit/recompute.py`. After the run, return the per-item lamp scores, labels, cells and option log-probs (not activations), so the result can be recomputed independently.
8. The smoke on 12345 must not compute any AUC on the target, only technical checks.
9. The "same new number" for all generators: say explicitly whether C-rule and P-native share it (see 4).

## Prioritised required changes

1. **Fix answer elicitation for the public families.** Use prefill or an answer-only instruction, add a pre-registered option-mass filter, and measure option mass for every family. Without this, the public targets do not reflect behaviour (redefine median log P(options) = −34).
2. **Make within-stratum AUC the primary metric**, within (arm × probe) cells or templates, with family-mean-centred features. Correct the plan's false claim about the note, and its claim that C-rule has untouched cases.
3. **Add surface and output baselines the lamp must beat**: a layer-0 or bag-of-tokens probe, a cell/template-identity predictor, an output-only logistic model, and a confidence baseline with a fitted sign. Add the trap-situation probe as a control.
4. **Take the C-rule/P-native twin out of the cross-direction criteria.** The main result is C→P on public P families. Report the pair separately.
5. **Fix sample sizes** so that every family has at least 30 positives and 30 negatives after filtering (more quote items; full redefine and NeQA). Set a minimum of 2 remaining families per direction, otherwise the direction is "uninformative". Add sets or intermediate arms so that the C-families and P-native have contrast inside each cell.
6. **MMLU:** disjoint splits; a validity check for the generic-error probe; and false alarms on X-mmlu as a criterion, not report-only.
7. **NeQA and quote validity checks.** Run un-negated NeQA and check original-quote knowledge, so that only "knows it but follows the habit" counts as a positive.
8. **Specify CV** (grouped by set or template, leave-one-family-out, the same metric as the test), solver and weights, bootstrap details, and per-item outputs for audit. Freeze all of this in the PREREG before Alex gives the number.
