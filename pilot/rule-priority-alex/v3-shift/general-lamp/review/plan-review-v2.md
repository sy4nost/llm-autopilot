# Review of general-lamp/PLAN.md v2 (2026-10-18)

Reviewer: the same independent session that wrote `plan-review.md`. I ran no Modal or GPU jobs. New local counts, all `python3 -I` and reading result files only:

- `cell_counts_v2.py` → `cell_counts_v2_out.txt`: which (arm × probe) cells reach ≥10 habit errors and ≥10 non-errors at the v2 sizes. It uses per-cell greedy/argmax rates from `match/results` and `newtasks2/new2_*.jsonl`.
- `redefine_cells_v2.py` → `redefine_cells_v2_out.txt`: the same question for redefine templates at 90 items per template, with and without losses to the filter.

## Verdict

**Ready to code. Before PREREG is frozen, fix the specification gaps below.** None of them needs a redesign. Of the five v1 blockers:

- three are resolved: the twin removed, within-cell AUC made primary, sizes and rules fixed;
- one is resolved in principle but may fail in practice: elicitation, because quote has multi-token options;
- one is half-resolved: cell confounding. The test metric is now within-cell, but the probe is still trained on labels that are mostly cell identity.

Three items matter most:

- **R1.** Train on cell-centred features. Otherwise C→P risks an uninformative FAIL.
- **R2.** Do not let MMLU be centred on its own mean for criterion 4, and set the threshold on out-of-fold scores. Otherwise criterion 4 is defined away, which is exactly the Zhao failure.
- **R3.** Do not let C-rule and C-parcels, an isomorphic pair, count as two families in the P→C mean.

## v1 blockers: are they resolved?

| v1 blocker | v2 | Status |
|---|---|---|
| Public options get no probability | Formats F1–F3 chosen on a dev split by median option mass; filter P(options) ≥ 0.5 | **Resolved in principle.** See N5 (quote) and N6 (readout position) |
| Cell or template identity gives AUC 0.87–0.98 | Within-cell AUC, cells with ≥10/10 | **Resolved for evaluation, not for training** (N1) |
| P-native twin in the criteria | Moved to report only | **Resolved** (but see N3: C-rule and C-parcels are also twins) |
| Unfixed sizes and rules | Sizes, splits, minimum families, MMLU false alarms as a criterion | **Mostly resolved**; gaps in N7–N12 |
| Baselines weak or unsigned | B1 with fitted sign, B2, B3, B4 with a validity check | **Resolved in principle**; B2/B3 definitions need fixing (N8) |

## Will enough cells qualify? (the ≥10 / ≥10 rule)

Estimated from earlier rates. The rates per cell come from only 20–30 sets, so treat them as rough.

**C-rule** (90 per cell, 16 cells): about **9.8 cells expected to qualify**.
- Nearly certain: all B_change_60 and C_change_60 cells except C D3 (3%).
- Likely: A_change_20 D2/D3 and A_change_60 D2.
- Almost never: A_change_20 D1 (93%), A_change_60 D1/D3 (93–97%), A_change_20/60 D4 (7%).
- **The v2 reason for adding A_change_20 is contradicted by the data.** A_change_20 is about as habitual as A_change_60 (D1 93% in both; D1+D3 88% vs 93% in `match/score_out.txt`). It adds about two marginal cells, D2 and D3. The cells with both outcomes come from B and C, not from the shorter history.
- Expected ≈ 280 errors and ≈ 600 non-errors in qualifying cells. Informative.

**C-parcels** (80 per cell): about **2.0 cells**, A60 T1 (60%) and E_nocopy T1 (80%).
- T2 is 95–100% habitual, and S1/S2 can never be positive.
- Non-errors in qualifying cells: about 32 + 16 ≈ 48. **Barely informative**; it can drop below 30 with a modest shift in rates.

**C-bonus** (80 per cell): about **1.8 cells**, A60 T2 (80%) and E_nocopy T2 (85%).
- Non-errors: about 16 + 12 ≈ 28, **below the 30 needed. Expect C-bonus to be non-informative** as a test family.
- As a left-out fold in the layer choice, C-bonus may also have no qualifying cell.

**P-redefine** (90 per template): **5.4 templates** at the old-protocol rates, 4.2 if the filter keeps 70%, 3.4 if it keeps 50%.
- The qualifying templates are almost all "Redefine π/e", plus "÷ instead". So the family AUC is mostly about the redefined-constant sub-task.
- Informative either way.

**P-neqa and P-quote** (one cell each): the counts depend on the filter and validity pass rates, which are unknown.
- NeQA: at a 60% filter pass and 85% validity, about 52 errors and 85 non-errors. Informative.
- Quote: see N5.

**Consequences.**
- P→C: the likely informative test families are C-rule, C-parcels (marginal) and NeQA. The ≥2 rule is probably met, but the C-side mean then rests largely on C-rule and its isomorphic sibling (N3).
- C→P: redefine and NeQA are likely; quote is uncertain. The ≥2 rule is probably met.
- The S cells never qualify, so the untouched probes contribute only to training.

## New issues in v2

**N1. Training labels still encode cell identity — CONCERN (R1).**
- The lamp is trained on all C-family prompts, labelled habit error vs not. In those data, most of the label variance is between cells: S vs T, arm, copy present. A predictor that knows only the cell gets about 0.87–0.98.
- The learned direction is therefore mostly "trap situation / exact copy in history". The test then asks only about variance inside a cell, and inside public families the situation is constant. That is a recipe for an uninformative FAIL in C→P.
- **Fix: make cell-centring the primary training rule.** Subtract each training cell's feature mean before fitting. This is label-free; the cell is design information. Then the probe learns only from within-cell variance, which matches the metric. Use only cells with both outcomes, or keep all and let centred one-class cells add nothing.
- Keep the unstratified training as a pre-registered secondary.
- In P→C, centre redefine by template; NeQA and quote are a single cell.

**N2. Family centring on test families — OK for AUC, harmful for criterion 4 (R2).**
- Within-cell AUC does not change when a constant is added to the scores. So test-family centring cannot change criteria 1–3; say so in the plan, so nobody thinks it helps.
- It does matter for criterion 4. If X-mmlu is centred on its own mean, the format offset that made the Zhao lamp fire on 100% of foreign prompts is removed by construction.
- **Fix:** compute criterion 4 with MMLU features centred on the *training-pool* mean, i.e. no MMLU-specific centring. Report the MMLU-centred value as secondary. Also say which items enter each centring mean: those after filter and validity exclusions.
- **Also: the 80% threshold must come from out-of-fold scores** (from the leave-one-family-out folds). Training-set scores are inflated, which lifts the threshold and makes criterion 4 too easy.

**N3. C-rule and C-parcels are an isomorphic pair in the P→C test mean — CONCERN (R3).**
- v2 says they "count as one group for spread", but never says where that applies.
- In the P→C mean the families have equal weight, so this pair gets double weight. Criterion 3 can pass or fail twice on the same structure.
- **Fix:** merge them into one family for the P→C mean and for criterion 3, each with weight 0.5, or define exactly what "одна группа" means in each computation.

**N4. Layer choice by leave-one-family-out — CONCERN.**
- With three C training families, one fold (likely C-bonus) may have no qualifying cell or fewer than 30/30. The plan does not say what happens then.
- **Fix:** pre-register that folds with no qualifying cell are skipped. If fewer than 2 folds remain, use a fixed fallback layer (e.g. 28), or pool folds' within-cell pairs.
- Specify that B1's sign, B2 and B3 are fitted with the same folds.

**N5. Quote: the filter may wipe it out, and its validity check is unspecified — CONCERN.**
- Quote options are multi-token continuations. Under the old protocol only 11% of items had P(options) ≥ 0.5 (`option_mass_out.txt`). Even a better format may keep few. The ones kept are likely the high-confidence repeats, which skews toward non-errors.
- "весь набор" is not a size. Write the N after fetching, and the expected errors after filtering on dev.
- The validity prompt "Complete the famous quote: <начало>" needs a decision rule. For example, score the famous and altered endings with the same option rules and count "knows" if the famous ending wins with mass ≥ 0.5. Also define how `<начало>` is cut from the prompt (the text after the last "Output:"?).
- Pre-register what happens if quote is non-informative. C→P then rests on redefine plus NeQA, and P→C trains on redefine alone, i.e. one sub-task family. Consider declaring P→C non-informative if only one P training family survives.

**N6. A different format per family means a different readout position — CONCERN.**
- F2 reads the state after "The answer is", while F1/F3 and the synthetic families read the end of the chat template or the raw prompt.
- If formats differ by family, readout position and format are confounded with family. Centring partly absorbs this, but the lamp direction must then transfer across positions too.
- **Fix:** choose one format for all public families: the one that maximises the *minimum* median option mass across families on dev. Measure option mass for the C families in that format. If it is fine, use it everywhere; otherwise report the readout position per family.
- Also state whether the ≥0.5 filter applies to the synthetic families and to MMLU, and which format MMLU uses (0-shot? letter? which F?).

**N7. NeQA validity check — mostly OK, two gaps.**
- Specify that "exactly one ' not '" is counted in the question stem only, not in the options or the instructions. Also say how "n't" and "Not" (capital, start of sentence) are treated.
- Apply the same format, filter and option rules to the un-negated prompt.
- Negatives are excluded on the same validity rule, which is consistent. Report how many items each step removes.

**N8. B2 and B3 definitions — CONCERN.**
- **B2:** "probabilities of options" is not transferable. Option order is arbitrary, and C families have 3 options while public ones have 2. Use order-free features: top-1 probability, the top-1 − top-2 margin, and entropy / log k.
- **B3:** layer 0 at the last token is the same token embedding for every prompt in a format, so it is constant and useless. The baseline is really the mean-pooled layer-0 vector over the prompt's own tokens. State that, exclude the chat-template tokens, and make sure it is saved.
- Fix the text "B4 weak, but **критерий 3** still applies": B4 is in criterion 2.

**N9. Bootstrap unit for redefine — minor.**
- "по наборам или шаблонам": with only 3–5 qualifying templates, resampling templates is degenerate.
- **Fix:** for redefine, resample items within each template, stratified. For C families resample sets; for NeQA and quote resample items.

**N10. Seeds — minor.**
- The generators use S, S+1, …, S+3. Make the PREREG require that this range avoid every seed already used, so no set seen before is regenerated: 25, 32, 185, 12345, and the generator defaults 20261018 and 20261020.
- `make_native.py` calls `mm.build()` with its default seed. Change it to S, or the "pair" is not the same sets.

**N11. MMLU exclusions — OK, write the exact recipes.** All three earlier samples can be reproduced:
- blockex: the first 10 per subject in test order (570 questions);
- watch_public: `random.Random(185)` shuffle, first 500;
- conflict: `random.Random(0)` shuffle, first 200 (`n_pub=200`).

`public/` used no MMLU, so remove it from the list. Put the code that applies the exclusions in the hashed script.

**N12. Earlier-seen public items — minor, disclose.**
- `public/` and `conflict/` both scored the `random.Random(0)` first 200 of redefine, quote and NeQA. Their per-item outcomes have been seen, including by me for template rates, and they informed the design.
- This is not a leak into the probe. Still, report test AUC with and without those items as a sensitivity check.

**N13. Verdict mapping — minor.** Spell out the remaining combinations:
- C→P passes and P→C is non-informative: "partial"?
- C→P fails and P→C is non-informative: "no"?

## Required changes before PREREG (prioritised)

1. **R1.** Train the lamp on cell-centred features (primary). Keep unstratified training as secondary.
2. **R2.** Criterion 4: no MMLU-specific centring (use the training-pool mean), and set the threshold on out-of-fold scores.
3. **R3.** Merge C-rule and C-parcels (or weight 0.5 each) in the P→C mean and in criterion 3.
4. Fix the quote specification: N, validity decision rule, how the start of the quote is extracted, the outcome if it is filtered out. Also decide whether P→C with a single surviving P training family counts as non-informative.
5. Use one common elicitation format and readout position across public families if possible. State the filter and format for the synthetic families and MMLU.
6. Fallback for layer-choice folds without qualifying cells; the same folds for B1–B3.
7. Order-free B2 features; B3 as mean-pooled layer-0 with the vector saved; fix the criterion-number typo.
8. Small items: bootstrap redefine within templates; seed exclusions and the `make_native` seed; the exact MMLU exclusion code; a sensitivity check without seen items; the full verdict mapping.
9. Expect C-bonus to be non-informative as a test family. Either accept that and say so in advance, or add sets (e.g. 160) so it reaches 30 non-errors within its trap cells. Correct the stated reason for A_change_20.
