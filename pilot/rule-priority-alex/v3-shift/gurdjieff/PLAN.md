# Gurdjieff sensor — PLAN v1 (2026-10-19)

## Question (Alex)
Can an internal sensor, reading the model's middle layers BEFORE it answers, catch the moment of automatism and switch on
"Stop, think aloud" only where needed, instead of thinking aloud on every question?

## What we already have (no new generation needed)
Qwen2.5-14B-Instruct, saved answers per item for DIRECT (answer at once) and THINK (think aloud, then answer):
- own tasks: thinkaloud/out_main.jsonl (seed 2; 2060 items; families R, N, K, D; traps + controls).
- public tasks: thinkaloud-indep/out_main.jsonl (seed 78; 2300 items; memo, NeQA, hindsight, into-the-unknown, algebra,
  controls, modus/quote/pattern/redefine, MMLU 1000).
Answers are re-derived with the frozen rules of each run (score_ta.py / score_ti.py), not re-generated.

## New measurement (one cheap GPU pass)
For every item, one forward pass of the exact DIRECT prefix of its run (same chat template, tail, "The answer is"/stem).
Save the residual stream at the last prefix token for layers {8, 16, 24, 28, 32, 36, 40, 44} (bf16) plus the final-layer
logits over the options' first tokens. Stored on the Modal volume; probe training also runs on Modal (CPU), only reports come back.
Sanity check: argmax over options' first-token logits at that position must agree with the saved DIRECT pick on >= 90% of items
whose options have distinct first tokens (otherwise the prefix is wrong -> stop).
Estimated cost: < $1 (about 4,400 forward passes, no generation).

## Sensor
- Teacher label (no answer key needed, the model's own "slow self"): y = 1 if THINK answer != DIRECT answer.
- Sensor = logistic regression on one layer's state (standardised, L2, C chosen by inner CV on the training pool only),
  layer chosen by inner CV on the training pool only.
- Gate: answer with THINK if sensor score > t, else DIRECT. t set on the training pool so that the gate thinks on 30% of items.

## Transfer splits (the earlier general-lamp failed across habit kinds, so transfer is the real test)
- S1 own -> public: train on all own-task items, test on all public items.
- S2 public -> own: train on public items, test on own items.
- Report-only: leave-one-family-out inside each pool.

## Rivals at the SAME thinking share on the test set (threshold recomputed on the training pool the same way)
- random gate (expected value), low-confidence gate (DIRECT margin), output-only sensor (option logits only),
  surface sensor (prompt length + family-blind word counts: "not", digits, "Answer"), always-DIRECT, always-THINK, oracle gate.

## Metrics (test pool; families weighted equally; bootstrap item + cluster schemes, wider interval)
- acc(gate), recovered share R = (acc(gate) - acc(DIRECT)) / (acc(THINK) - acc(DIRECT)), think share actually used.
- AUC of sensor for y within family.

## Pre-registered criteria (each split separately; verdict "sensor works" for a split if all pass)
- G1 acc(gate) - acc(DIRECT) > 0 with low > 0.
- G2 acc(gate) - acc(low-confidence gate) >= +3 points, low > 0; and >= output-only and surface sensors (point).
- G3 R >= 0.5 at think share <= 40% (actual share on test).
- G4 no harm: on controls and MMLU, acc(gate) - acc(DIRECT) >= -2 (point).
Overall: "internal lamp transfers" if both S1 and S2 pass; "works one way" if one passes.
Prediction (mine): S1 and S2 likely fail G2 (habit kinds differ, as in general-lamp); within-pool may work.

## Later (not in this run)
- "Stop": on sensor fire, replace think-aloud by "describe the conditions without interpreting, then answer" (Alex's ladder).
- "Mortality" knob as a separate arm, measured for help vs harm.

## Steps
plan review -> extraction + analysis code -> smoke on 40 items -> code review -> PREREG with sums -> Alex OK -> run -> report.

---
# PLAN v2 (after review/plan-review.md; overrides v1 where they differ)
Reviewer facts (saved outputs): y = THINK != DIRECT is 24% own / 15.7% public; population identity alone predicts y with AUC
0.81 / 0.74 (own template cell 0.93); inside N-traps and hindsight the fixable items are CONFIDENT (low-margin AUC 0.24 / 0.28).
So the real question is: inside one kind of question, does the sensor know WHICH items will flip?

Changes:
1. Primary criterion is within-population AUC on the test pool (B1, B2):
   P1 mean within-population AUC of the sensor for y (populations with >= 10 y=1 and >= 10 y=0 on test), >= 0.70,
      and >= best of {low-confidence, output-only, surface/length} + 0.05 with low > 0 (bootstrap over items within populations).
2. Gain at equal thinking share (B3): every gate thinks on exactly 30% of the TEST items (label-free quantile of its own scores on
   the test pool). P2 acc(sensor gate) - best rival gate (random, low-confidence, output-only, surface, length) >= +2 points, low > 0.
   Curve 10-40% reported. Accuracy gain is the main number; recovered share R report-only.
3. Harm per population (no pooling): R-cfright, R-agree, N-agree, K/D-agree+plain, MMLU, memo/neqa/hind/alg controls, modus, quote:
   P3 every one of them acc(sensor gate) - acc(DIRECT) >= -3 points (point estimate). S2 has no MMLU on the test side: stated.
4. Leakage (B4): public redef excluded from both splits.
5. Extraction (B5, B6): each run's direct() copied verbatim (own N: chat(text), options without leading space; others
   chat(text+tail)+"The answer is"/stem, options " "+c). Every option is scored by its own full pass (batch 1, no padding) exactly as
   in the runs; the sensor reads the residual at the position that predicts the FIRST token where the options differ (shared
   option tokens like the leading space are part of the prefix). Sanity: recomputed forced-choice pick == saved pick on >= 98% of
   items in EVERY family, else stop. Output-only sensor = log-probs of the options at that same decision position.
6. Inner CV for layer and C grouped by family (leave-one-family-out inside the training pool).
7. Splits: S1 train own (R, N, K, D all populations) -> test public (memo, neqa, hind, unknown, alg incl. controls, modus,
   quote, pattern, MMLU; no redef). S2 the reverse. Bootstrap: item scheme within population + cluster scheme (run's cluster/grp).
8. Per-item scores of all sensors saved locally (scores_S1.jsonl, scores_S2.jsonl). Diagnostic label y+ = THINK right and DIRECT
   wrong reported. Public THINK labels come from a resumed run (small batch noise), stated.
Verdict per split: "internal lamp works" if P1, P2, P3 pass. Overall as in v1.
Prediction (mine): P1 fails on S1 and S2 (confident flips are hard to see inside a family); if anything works it is within pool.
Cost: about 4,400 items x 2-4 options, forward passes only, est. $1-2.

# PLAN v2.1 (after "v2 check" in review/plan-review.md)
- Rival direction fixed on the training pool (sign chosen so its training within-population AUC >= 0.5), never per test population.
- Gates at the 30% test quantile: ties broken by a fixed-seed tiny jitter; verdict wording "works at a calibrated share".
  In P1/P2 bootstraps the best rival is re-chosen inside every resample.
- N1 (required for S2): within-cell check on the own test pool (cells = family/pop/template cell with >= 5 flips and >= 5 non-flips):
  mean within-cell AUC of the sensor >= 0.65 and > best rival's (point). Otherwise S2 is "template recognition, not a lamp".
- Notes: P1 also rewards predicting harmful flips (R-cfright); P3 informative only on R-cfright, MMLU, neqa-control, quote, modus.
