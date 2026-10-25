# Watchman on public habit tests never used in this project — PLAN v1 (2026-10-23)

Alex, 2026-10-23 23:12Z: "But check the watchman" — after the conftrap-lamp PASS (../conftrap-lamp/rezultaty.md), check the watchman
(lamp = fast answer != thought answer) on outside tests the project never touched.

## Why
conftrap-lamp passed on families we selected for confident habit errors, 3 of them our own. Open question: does the lamp beat the cheap
rivals on independent public habit tests, chosen and built by others, where errors are not all confident?

## Tests (fixed now; picked before any lamp or thinking output on them)
Searched for public tests of "habit vs the stated condition", not used anywhere in this project. Pilot (sets_wp_pilot.json, seed
20261024, fast answers only, report_pilot.txt) checked that fast answers make errors on them:
- cpkc: BIG-bench contextual_parametric_knowledge_conflicts. A passage swaps the real answer for another entity; "answer assuming the
  context is true". 2 options: context entity (right) vs memorized one (habit). 100 items, no controls (the task has none).
  Pilot fast right: Qwen 53, Phi-4 62, OLMo 47 of 100; most errors are the memorized answer.
- modar: BIG-bench modified_arithmetic (+1 tasks: 3-digit addition, 3-digit subtraction, 2-digit multiplication, 5 worked lines each).
  Options op+1 (right), plain op (habit), op+2. 100 traps (34/33/33) + 40 controls from the *_control subtasks (plain op right).
  Pilot fast right: 40 / 48 / 0; controls 100.
- base9: Wu et al. 2023 "Reasoning or Reciting" counterfactual arithmetic, their prompt template, fresh 2-digit base-9 numbers with a
  carry. Options base-9 sum (right), base-10 reading (habit), base-9 sum +-1. 100 traps + 40 base-10 controls.
  Pilot fast right: 53 / 42 / 0; for Qwen and Phi-4 most errors are NOT the habit answer (computation slips).
All three enter the verdict, whatever the pilot showed per model (no per-model selection). Unlike conftrap-lamp, fast errors here are
partly unconfident (Phi-4 wrong-answer confidence ~70-90%), so CONF is not handicapped by selection.
Items of the pilot and the smoke are never reused (by prompt; cpkc also by passage+question).

## Models, arms, measures, criterion: identical to conftrap-lamp (PLAN v2 there)
Qwen2.5-14B-Instruct, Phi-4, OLMo-2-13B-Instruct; bf16, greedy, H100. DIRECT forced choice after "The answer is"; THINK = task +
"Think it through step by step. Then give your final answer on the last line as: Answer: <answer>", 512 tokens; SAMPLE = 5 samples, T=1.
Parse: cpkc = the one option whose full name appears on the final Answer line, else forced choice; modar/base9 = the one option number
named, an Answer line naming no option = off-list (counts as differs and wrong), forced choice is the sensitivity readout.
H1: FLAG beats EACH of RANDOM@r, CONF@r, SAMPLE@r by >= +10 recall points (mean over eligible cells, E >= 15 and non-E >= 15, at least
4 of 9 cells), bootstrap 2000 item and cluster schemes (clusters: cpkc by question, others by item), low > 0, > 0 in >= 2 of 3 models.
Plus: readout signs hold (DIRECT length-normalised; THINK forced) and leave-one-family-out (each rival's point difference stays > 0
with any one family dropped). Verdict "confirmed on public habit tests" only if all hold.
Report-only: THINK - DIRECT on traps, controls fast vs thought, CONF AUC, prior-class rival, think on 30% least confident, SAMPLE'.

## Threats named now
- Public data may be in training sets (BIG-bench canary present). Nothing is fit, so it cannot favour a method, but it can change how
  hard the items are.
- base9 errors of Qwen/Phi-4 are mostly not the habit answer: the lamp is tested there as an error finder generally, not habit-only.
- On 2-option cpkc, recall(FLAG) = P(thought right | fast wrong).
- One THINK wording, thresholds carried over (+10), not tuned.

## Anti-fitting
Everything above fixed before any THINK output on these tests. Smoke (seed 12345, 16 items) = pipeline checks only. PREREG.md with
sha256 of make/modal/score and the items file frozen before the main run; Alex gives the seed. Later analysis labelled post hoc.
Prediction (mine): passes vs RANDOM and SAMPLE; vs CONF closer than in conftrap-lamp, could fail on Phi-4 and on base9.

# PLAN v2 (after the conftrap-lamp audit, ../conftrap-lamp/audit/audit-report.md; written before any main-run output)
The audit showed that on key-vs-habit items recall(FLAG) is almost exactly P(thought right | fast wrong): the lamp catches an error only
when reasoning fixed it, ties a rival that also sees the reasoned output (+0.1), adds no accuracy over "always reason", and is blind to
errors reasoning does not fix. So the substantive question here is whether REASONING fixes habit errors on outside tests.
V1 (primary verdict): THINK - DIRECT accuracy on traps, mean over the 9 model x test cells >= +5 points, bootstrap low > 0, > 0 in >= 2
  of 3 models; same sign with THINK read by forced choice; controls (habit right) THINK - DIRECT >= -5 for every model; and the mean
  stays > 0 with any one test dropped. Verdict "reasoning fixes habit errors on public tests" only if all hold.
V2 (secondary, the old H1): FLAG vs RANDOM / CONF / SAMPLE as in PLAN v1. Stated now: it is expected to pass whenever V1 holds strongly,
  so on its own it is NOT evidence that disagreement is a clever detector.
Report-only, required in the write-up: FLAG - THINKPROB (same-information rival: lowest forced-choice probability of the fast answer read
  after the reasoning; predicted ~0); recall of FLAG / CONF / SAMPLE / RANDOM on fast errors split into "fixed by reasoning" and
  "not fixed" (predicted: FLAG ~100 vs ~0).
