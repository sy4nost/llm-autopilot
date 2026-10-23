# Plan v2: a general lamp "time to re-check the habitual answer"

*English translation. The original Russian text, whose sha256 is listed in PREREG.md, and the earlier draft PLAN_v1.md are in the git history (commit 6d3f172). Quoted phrases in review/ refer to the original wording.*

2026-10-18. Version 2, rewritten after the plan review (review/plan-review.md, 5 blockers). The first version was PLAN_v1.md. v2.1: 9 clarifications from the second review were applied (review/plan-review-v2.md, verdict "ready to write code"). Then I write PREREG.md with checksums, and only after that does Alex name the number.

## Question (Alex's hypothesis)

Is there one signal inside the model, "the habitual answer does not fit now", common to two kinds of habit:
- **kind C**: the model copies from the examples in the prompt;
- **kind P**: the model pulls from what it learned earlier, with no examples in the prompt.

We do not build a brake; we test only detection.

**Model:** Qwen2.5-14B-Instruct.

## What changed compared with v1 (after review)

1. **The answer on public tasks will be real.** Previously the answer options got almost no probability: the model was about to explain. Now the format is chosen on a separate "trial" part and filtered by option probability (section "How we ask").
2. **The main measure is quality within a cell.** A cell is arm × case type, or a template in redefine. Otherwise knowing "which cell this is" alone gives AUC 0.87–0.98, and the 0.70 threshold tested nothing.
3. **More simple rivals:** surface features (layer zero), output probabilities only, confidence with a fitted sign, a generic-error detector with its own validity check.
4. **The C-rule / P-native pair was removed from the criteria.** They are twins, so the pair goes only into the report. Main transfer: C → public P.
5. **All sizes, splits and exclusion rules are written in advance.** There is a minimum number of families, otherwise the conclusion is "nothing to say". False alarms on MMLU became a criterion.
6. **Validity checks for NeQA and quotes.** Only "knows, but gave in" counts as a habit error.
7. **Per-prompt results are saved**, so that a reviewer can recompute everything.

## Tasks and sizes

Alex's number sets `S`. Generators run with `S, S+1, S+2, …` to get the needed number of sets. The split of public tasks into "trial" and "test" parts is made in advance with number 12345, before Alex's number. The trial part is used only for format selection and never enters scoring.

**Training, kind C:**

| family | generator | sets | what we take | prompts |
|---|---|---|---|---|
| C-rule | match/make_match.py | 90 | A_change_60, B_change_60, C_change_60 × D1–D4 | 1080 |
| C-parcels | newtasks2/make_new2.py | 80 | A60, E_nocopy × T1, T2, S1, S2 | 640 |
| C-bonus | newtasks2/make_new2.py | 160 | A60, E_nocopy × T1, T2, S1, S2 | 1280 |

- **A_change_20 removed.** The reviewer showed it is almost as habitual as A_change_60 (D1 93% in both) and does not give the needed spread.
- **C-bonus increased to 160 sets**, to collect ≥ 30 "no" cases in trap cells.
- **C-rule and C-parcels are built identically.** Wherever families are averaged (the P → C mean, the "no blind family" criterion), each of them has weight 0.5.

**Generator numbers:** S, S+1, …, S+k in order. Any number already used (12345, 25, 32, 185, 20261020, 0) is skipped. make_native.py gets S instead of its default number.

**Kind P, public tasks** (trial part / test part):

| family | total | trial | test | habitual answer | validity check |
|---|---|---|---|---|---|
| P-redefine | 900 | 90 (10 per template) | 810 | standard math (checked on all 900) | not needed |
| P-neqa | 300 | 30 | 270 | the answer to the question without "not" | the same question without "not" (replace exactly one " not "; if "not" does not occur exactly once, the question is excluded). A habit error only if the model answers the question without "not" correctly |
| P-quote | the whole set | 10% (number 12345) | the rest | the famous ending of the quote | see below |

**Validity check for P-quote.**
- **Quote beginning**: the text from the prompt that comes before the answer options. This is the shared beginning after which the two options diverge.
- **Check prompt:** "Complete the famous quote: <beginning>".
- **The model "knows" the quote** if the habitual (famous) option gets a higher sum of log-probabilities there than the altered one.
- **Habit error**: the model knows the quote and in the real prompt chose the famous option.
- **What knowledge does not change:** if the model does not know the quote, the prompt is excluded and does not go into class 0.

**If P-quote drops out** (fewer than 30 errors or fewer than 30 "no" after filters):
- P → C is left with one training P family, and the P → C direction is declared "non-informative";
- C → P is then computed on the remaining families.

**Previously scored questions.** 200 questions of each public family were scored before. Results are given twice: with all questions and without these 200. The verdict uses all of them.

**Report only:**
- P-native (90 sets, change_0 × D1–D4, 360 prompts, the same generator with number S): the pair with C-rule;
- MMLU, 1500 new questions, sampled with number S. Already used ones are excluded: the first 10 per subject (as in blockex), the first 500 by the watch_public recipe with number 185, the first 200 by the conflict recipe with number 0. MMLU was not in public/. Of the remaining:
  - 1000 go to training the generic-error detector,
  - 500 to its testing and to false alarms.

**Total** about 7000 prompts.

## How we ask (public tasks)

A fixed list of three formats:
- **F1**: prompt + "Reply with only the answer, nothing else.";
- **F2**: F1 + the assistant's answer starts with "The answer is";
- **F3**: the original prompt without chat.

**One format for all public families and MMLU**, so that the state is read at the same place. The one of F1–F3 is taken that has, on the **trial** part, the highest worst-family median option probability.

Generated tasks keep their own format ("Return only the answer"). There the options already get almost all the probability.

**Filter:** only prompts where the total option probability is ≥ 0.5 are scored. The rule is the same for all families, including generated ones and MMLU.

**Multi-token options** are scored by the sum of log-probabilities of all the option's tokens. The model's answer is the option with the highest sum. The same for generated tasks (labels are distinguishable by the first token, but the rule is one).

The format is chosen automatically by a script on the trial part during the technical run and recorded in PREREG before Alex's number.

## The lamp's target

**Habit error:**
- on generated tasks, the model chose the old label although another one is correct;
- on public ones, the model chose the habitual option and passed the validity check.

**"No" (class 0)**: everything else, except prompts that failed the filter or the validity check. Those are excluded entirely.

## What we read

For each prompt we save:
- the state at the last prompt token (for F2, after "The answer is") at layers 8, 12, …, 44;
- the mean over all tokens of the prompt itself at layer 0 (for rival B3: at the last token, layer 0 does not depend on content);
- option probabilities.

States are stored on a Modal disk. Scoring runs on Modal CPU with a frozen script. Returned are a file with each detector's score for each prompt plus a summary table.

## Detectors

**The lamp:** logistic regression on the state of one layer from 8…44.
- C = 0.1, balanced class weights, families weighted equally.
- Standardization is computed on training data only.
- **The main rule is cell-centering.** The mean of its cell is subtracted from each prompt (no answers are needed for this). This way the lamp learns what the metric measures: distinguishing "gave in" from "held out" within identical cases, not recognizing the cell. On test families the centering is the same.
- Training without centering goes only into the report.

**False alarms on MMLU (criterion 4).**
- MMLU is centered by the mean of the **training** pool, not its own. Otherwise a format shift, which broke the Zhao lamp, would disappear by itself.
- The "catches 80%" threshold is set by out-of-fold scores on the training families, not by scores on the same data it was trained on.

**Layer choice:** leave-one-family-out cross-validation within the training families.
- The target is the mean within-cell AUC on the held-out family.
- Within a family the split is by sets or templates, not by prompts.
- If the held-out family has no valid cell, the fold is skipped.
- If there are no valid folds at all, the layer is chosen by 5-fold cross-validation split by sets or templates over all training cells.
- Rivals B1–B3 are fitted on the same folds.

**Rivals that must be beaten:**
- **B1 confidence:** the lead of the best option over the second. The sign (fires at small or at large lead) is fitted on training data.
- **B2 output only:** logistic regression on features that do not depend on the order and number of options: best-option probability, lead, entropy / log k, total option probability.
- **B3 surface:** the same lamp on the mean over prompt tokens at layer 0 (right after the vocabulary, before any "understanding").
- **B4 generic errors:** the same lamp trained on 1000 MMLU "right / wrong", best layer by CV on MMLU.
  - **B4 validity:** AUC on the held-out 500 MMLU ≥ 0.65. If lower, B4 is marked weak, but criterion 2 still applies.

**Control, report only:** a "trap situation" detector (trap vs untouched) on the C families. How well it finds habit errors by itself.

## Quality measure

**Within-cell AUC.**
- Cells: arm × case type for generated tasks; template for redefine; the whole family for NeQA and quotes, since they have no templates.
- Only cells with ≥ 10 errors and ≥ 10 "no" are counted.
- A family's AUC is the mean over such cells, weighted by the number of pairs.

**Direction:** the mean over informative test families, families weighted equally.

**Informative family:** ≥ 30 errors and ≥ 30 "no" over all its valid cells together.

**Intervals:** bootstrap, 2000 resamples.
- Resampling within each family: for generated ones by sets, for redefine by questions within each template, for NeQA and quotes by questions.
- In each resample we average over families.
- For differences with rivals the bootstrap is paired: the same resamples.

## Directions

**Main: C → P.**
- Training: C-rule, C-parcels, C-bonus.
- Testing: P-redefine, P-neqa, P-quote.

**Second: P → C.**
- Training: P-redefine, P-quote.
- Testing: C-rule, C-parcels, C-bonus and P-neqa (it is held out as external in both directions).

## Criteria (written before the run)

A direction **passes** if all of the following hold:
1. the lamp's within-cell AUC ≥ 0.65;
2. the lamp beats each of B1, B2, B3, B4 by ≥ 0.05, with the lower bound of the difference interval > 0. All four are required, so no multiplicity correction is needed;
3. in no informative test family is the lamp's AUC below 0.55;
4. false alarms: at the threshold that catches 80% of habit errors on the training families, the lamp fires on no more than 25% of MMLU questions the model answered correctly (from the held-out 500).

**Too little data:** if there are fewer than 2 informative test families, the direction is "non-informative".

**Outcome** (C → P main, P → C second):

| C → P | P → C | outcome |
|---|---|---|
| passes | passes | **a general signal exists** |
| passes | fails or non-informative | **partial: the in-context signal recognizes the habit from training** |
| fails | passes | **partial: reverse transfer only** |
| fails | fails or non-informative | **no** |
| non-informative | any | **non-informative** (not enough data, no conclusion about the signal) |

**Expectations written in advance:**
- By the reviewer's earlier numbers, C-bonus may turn out non-informative even with 160 sets.
- P-quote may drop out at the probability filter.
- Both outcomes are possible and count as neither failure nor success.

**Report only:**
- the pair C-rule → P-native;
- within-kind results;
- the best layer;
- the "trap situation" control;
- plain AUC without the cell split next to a "predictor by cell".

## Cost

About 7500 single passes (plus check prompts for NeQA and quotes), about a third of them long (60 examples). Roughly $4–6 on H100 plus pennies on CPU.

## Order

1. The reviewer looks at v2.
2. Code plus a technical run on 12345: format choice on the trial part, share passing the filter, saving of states, match with earlier probabilities.
3. PREREG.md with checksums and the chosen formats.
4. A number from Alex and his consent to the cost.
5. Run, scoring, report as is.
