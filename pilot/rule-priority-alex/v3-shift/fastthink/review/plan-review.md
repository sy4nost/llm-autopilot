# Plan review: fastthink/PLAN.md v1 (independent auditor, 2026-10-19)

Scope: PLAN.md, checked against make_ft.py / modal_ft.py / score_ft.py as they stand (19:52), the data files and the
thinkaloud-indep record. Nothing was run on Modal. I built the main item set in memory (seed 999, no file written) to check counts.

## Verdict
Feasible and cheap. The anti-fitting rules are good, but I found **7 blockers**. The most serious: (B1) sig-figs sampling
makes "the last option" the right answer most of the time, (B2) H1 is mostly a restatement of H2 and is judged only against
one rival that the plan itself expects to fail, and (B4) the CIs treat the same items under 3 models as independent.

## BLOCKERS (fix before running)

**B1. sig-figs sampling adds an answer-position confound, and the plan text is wrong about it.**
The plan says "3 options letters". The data has 2-, 3- and 4-option items (8,897 / 9,480 / 2,520), and key 3 occurs only in 4-option items.
`strat()` balances by key across all of them. The seed-999 draw gives these (n options, key) counts:
(4,3):50 (4,2):7 (4,1):2 (4,0):1; (3,2):43 (3,1):25 (3,0):18; (2,0):31 (2,1):23.
So 50 of 60 four-option items have D right, and 43 of 86 three-option items have C right. A "pick the last option"
bias scores well, and THINK-vs-DIRECT differences could be position effects rather than habit effects.
Fix: use 3-option items only, as the plan says, balanced 67/67/66 by key. Or stratify by (n_options, key) within each option count.
State chance level.

**B2. H1 is weakly identified and stacked against one rival.**
(a) In binary cells (HANS, epistemic, 2-option sig-figs), a fast error flagged by FLAG means THINK chose the other option, so THINK is right.
That makes recall(FLAG) = P(THINK right | DIRECT wrong) = autopilot rate / E rate, and precision(FLAG) = P(THINK right | disagree).
H1 is then largely "THINK fixes many fast errors", which is H2. It is not an independent property of a "lamp".
That is fine as a measure. But the verdict text must not present H1 and H2 as two confirmations, and the R6 number is the same thing.
(b) The only rival is CONF@r. The plan predicts CONF fails on TruthfulQA ("wrong fast answers are confident, which favours FLAG"),
so beating it there is partly by design. Required: also report and require FLAG > **random@r** (recall = r in expectation)
in the H1 mean. CONF can fall below random, and then "beats CONF" says nothing.
(c) The stated use case is "API models, no weights". CONF needs log-probs, so it is not the rival for that claim.
Either drop the API claim from the verdict, or add a weight-free fast rival with less compute than THINK. Options, any one fixed now:
option-order flip (re-score DIRECT with options reversed or shuffled, flag disagreement; one extra forced pass, almost free);
k=3-5 sampled short direct answers (self-consistency, flag non-unanimous); verbalised confidence.
Pick one now and include it in H1 as "FLAG must beat each pre-declared rival".

**B3. Readout mismatch confound (affects H1 and H2).**
DIRECT is argmax summed log-prob after "The answer is". THINK is parsed text with forced-choice fallback.
Some FLAG disagreements, and some H2 gain, can come from the readout and not from the thinking.
The previous plan required the gain to hold against 3 DIRECT variants. This plan dropped that without saying so.
Fix: pre-register these sensitivities, computable from saved per-token log-probs at no extra cost:
- DIRECT length-normalised (mean log-prob; matters for " entailment" / " non-entailment" and memo words);
- FLAG_forced (THINK forced-choice argmax != DIRECT) and THINK_forced accuracy.
Rule: the verdict stands only if these sensitivities have the same sign. A raw-completion DIRECT is optional.

**B4. Bootstrap ignores that the same items appear in all 3 models.**
`boot()` resamples each (model, test) cell independently. The 3 model cells of a test share items, and item difficulty and trap strength are
strongly correlated across models. The CI of the cross-model mean (H1, H2, R4) is therefore too narrow.
Fix: per test, draw one resample of item ids (or clusters) and apply it to all models.
This applies to both schemes and to the per-model ">= 2 of 3" summaries.

**B5. Test selection is informed by earlier results, and exclusions are undocumented. This is a forking path, which is exactly what Alex asked to avoid.**
- In thinkaloud-indep/PLAN.md v2, sig-figs was excluded because "it tests a rounding skill, not a habit". It is now included as a habit trap,
  after it became known that THINK helps where step-by-step work helps. sig-figs is largely arithmetic, so CoT gain there is expected
  and is not evidence about habits.
  Fix: pre-register H2 and H1 both with and without sig-figs (leave-sig-figs-out must pass too), or move sig-figs to report-only.
  Write down why the earlier exclusion is reversed.
- fetch_ft.py fetched prompt-injection (data/prompt-injection.jsonl is present) and tried resisting-correction. Neither is in the plan and no reason is given.
  The earlier review flagged the same pattern. Fix: state in PLAN, before the run, why each is out (failed to load / not an MC format / ...),
  or include it.
- The earlier run showed FLAG beats CONF exactly where THINK gains a lot (memo .97 vs .78, hind .90 vs .77) and loses where it gains little (NeQA .65 vs .75).
  Choosing tests expected to give big THINK gains therefore pre-selects in favour of H1. Fix: per-test predictions (direction for H1 and H2)
  written in PLAN now. Better: name an outside rule for test choice, e.g. "every inverse-scaling round-2 MC task not used before, plus X".

**B6. Pass rules are under-specified.**
- H2 "12 cells": the HANS cell mixes 150 traps and 150 controls. State whether H2 uses HANS traps only or the whole cell.
  score_ft uses the whole cell, which halves the HANS gain. Either choice is fine, but it must be written down.
- If a model is dropped: is it "> 0 in >= 2 of 2"? Does H1/H2 still run with 2 models?
- Minimum number of eligible H1 cells, e.g. ">= 6 of 12, else H1 not testable". Without it, H1 could rest on 2-3 cells.
- Plan text errors to correct: "~1850 items" (the draw is 1,615); "prompts identical to thinkaloud-indep for ... sig-figs" (sig-figs was never run there).

**B7. TruthfulQA parse with up to 13 letters.**
The parser takes the first standalone capital A-M on the Answer line. "Answer: I think B", "Answer: A person ..." and
"Answer: The correct answer is C" are mis-read when I / A / C are options. 10 of the 250 drawn questions have >= 9 options (so I is a valid option), and every question has A.
This is a code item, but it decides THINK accuracy on one of the 4 new tests.
Fix: prefer a pattern like `^\(?([A-M])[\).:]?(\s|$)` at the start of the line, else a letter followed by "." or ")", else forced choice.
Smoke must print parse rate per model per test (it does) and a few raw Answer lines, which are not accuracy.

## Label and format checks (done on the data)
- HANS (tasksource/hans, 15,000 rows, 7,559 label 0 / 7,441 label 1): 0 = entailment, 1 = non-entailment. Spot check is consistent
  ("The manager left the garden." -> "The manager left." = 0). make_ft mapping trap = 1, control = 0 is **correct**.
  - The file has no heuristic/subcase/template fields. So there is no per-heuristic breakdown (lexical overlap / subsequence / constituent),
    which reviewers expect for HANS. Clustering by premise is close to item-level, while HANS items are template-generated, so the
    cluster bootstrap does not capture template dependence. Suggest getting the original HANS release with its heuristic/subcase columns,
    or at least saying so.
  - 2,966 premises contain " ," (tokenised spacing). make_ft fixes " ." only. Suggest also fixing " ,".
- Epistemic (400 rows, 147 entailment / 253 non-entailment, options always ["entailment","non-entailment"], the target is a word):
  mapping by WORD_OPTS.index is **correct**. The draw is 100/100. Stripping the trailing "Relation:" is fine.
- TruthfulQA MC1 (817): the right option is **first in all 817**. make_ft shuffles with the seed and recomputes the key, which is **correct**.
  Options per question range 2-13 (chance varies), so report chance-corrected accuracy or chance per cell.
  Contamination risk: TruthfulQA is a standard dev/eval set in OLMo-2/Tulu-3 and widely used in post-training, so it is weak as an "unseen habit" test.
  Say this in the limitations.
- sig-figs: classes " A"/" B"(/" C"/" D"), answer_index matches the letters, all round 2. The habit (decimal places) sits in the
  distractor. See B1 for the sampling problem.
- Memo/hindsight replication: 150 memo (75/75) and 115 hindsight (58/57) are unused by thinkaloud-indep. OK.

## SUGGESTIONS
1. More rivals, report-only (cheap, label-free; a reviewer will ask):
   - THINK's own forced-choice margin (confidence after thinking);
   - "predicted-class prior": at rate r, flag items whose DIRECT answer is the class DIRECT over-predicts in the cell (label-free), lowest margin first. On HANS and epistemic this is strong;
   - always-flag (precision = E rate), as lift over base rate;
   - a precision/lift table at r.
   Also report CONF's AUC (curve) next to the single equal-rate point.
2. Report the H1 recall gap on MMLU too (non-habit errors). If FLAG wins there as well, the "habit lamp" reading is weaker
   ("thinking corrects errors" in general).
3. Main set: exclude the fastthink smoke items. Seed 999 overlaps 9 of them by cluster; the earlier run had this rule. Smoke generations get read.
4. Truncation: phi-4 is verbose, and sig-figs and MMLU had 6% truncation on Qwen already.
   Pre-declare: if smoke truncation > 10% for a model, MAXNEW = 1024 for all models, decided before the main run.
   Report THINK accuracy on non-truncated items.
5. Modal: the earlier run died on the workspace limit. Spawning 3 H100s in parallel may hit it again.
   Run models one after another, or confirm the limit. Also report per-model wall time and cost.
6. Report per-class accuracy (HANS trap/control, epistemic by key, TQA by number of options) so a class shift is visible, as done before.
7. The ">= +10 points" H1 threshold and "+5" H2 threshold carry over. Note in PREREG that they were not tuned on these data.
8. Multiple verdicts (H1, H2, harm, replication): fine as written. Do not add "per test" verdicts afterwards. Label all leave-one-out results post hoc unless pre-registered (see B5).
9. The OLMo-2 template begins with a "<|endoftext|>" text token. Check in smoke that it is tokenised once as a special token
   (add_special_tokens=False is right). `checks.template_sample` shows only the tail; also print the head.

## Cost and feasibility
1,615 items x 3 models: about 4,850 generations of up to 512 tokens (51 batches of 32 per model, about 15-25 min on H100 with HF generate).
Forced-choice passes are about 10k per model (TQA about 5 options x 2, MMLU 4 x 2, hindsight 10-shot prompts are the long ones).
With weight download and load, it should be about 45-75 min per model, so about $4-6 total, in line with the budget.
The 3 h per-model timeout is ample. The resumable volume state is good.

## What a skeptical grant reviewer would say
"Disagreement between CoT and no-CoT answers predicts no-CoT errors on tasks where CoT is more accurate. That is close to a tautology
in binary tasks. The tests were picked after seeing where CoT helps. The only rival was the one expected to fail. The CIs ignore
that the same items repeat across models."
B1-B5 answer these points. After the fixes, a positive H2 on HANS/epistemic/TruthfulQA (without sig-figs) across 3 families would be a credible, modest result.
