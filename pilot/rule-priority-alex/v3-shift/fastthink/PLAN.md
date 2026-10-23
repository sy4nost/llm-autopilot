# Fast answer vs thought answer on public tests, 3 models — PLAN v1 (2026-10-19)

Alex: "check it on public tests, no fitting." Budget agreed ~$4-6 (Modal H100).

## Questions
Q1 (the lamp). When the model's FAST answer and its THOUGHT answer differ (FLAG), does that point at fast-answer errors
better than the plain rival "flag the same number of least-confident fast answers" (CONF)?
Q2 (the tool). Does thinking aloud raise accuracy on public trap tests the project never touched, on 3 model families, without harm?
Honest scope stated up front: FLAG needs the thought answer, so it saves no compute versus "always think". Its use is a
behavioural autopilot meter (works on any model, also API models, no weights needed): "here the fast answer was on autopilot".
The compute-saving version (think only where confidence is low) is report-only (R3).

## Models (fixed now)
Qwen/Qwen2.5-14B-Instruct, microsoft/phi-4 (14B), allenai/OLMo-2-1124-13B-Instruct. bf16, greedy, H100, each model's own chat template.
If a model fails to load or its chat template fails, it is dropped and reported; no replacement is chosen after seeing any output.

## Tests (fixed now, before any smoke output)
NEW = never used in this project before:
- sig-figs (inverse-scaling round 2; habit = decimal places instead of significant figures), 3 options letters. 200 items, stratified by key.
- TruthfulQA MC1 (imitative falsehoods = popular-misconception pull). 250 questions, options shuffled with the seed, letters A..
- HANS (tasksource/hans; every item has high word overlap; non-entailment items = traps of the overlap heuristic, entailment
  items = controls where the heuristic is right). 150 + 150.
- BIG-bench epistemic_reasoning (tasksource/bigbench validation; HANS-like "X thinks that..." heuristics), 200 of 400, stratified by key.
REPLICATION = old tests, new items (items not in thinkaloud-indep sets or its smoke, by cluster): memo-trap 150, hindsight-neglect all
unused (<= 115). Report-only, no verdict.
GENERAL harm check: MMLU 400 questions not used in thinkaloud-indep.
Seed = Alex's number. Smoke = seed 12345, 6 items per test, prints only pipeline checks (parse rates, load, sanity), NOT accuracy.

## Arms (per model, per item)
DIRECT: chat(task + "\n\nReply with only the answer, nothing else.") + "The answer is" (memo: the unfinished quote), options " "+c,
summed option log-prob (each option its own pass), argmax.
THINK: chat(task + "\n\nThink it through step by step. Then give your final answer on the last line as: Answer: <answer>"),
greedy up to 512 new tokens; answer = option parsed from the last "Answer:" line, else forced choice after "Answer:".
Prompts identical to thinkaloud-indep for memo/hindsight/sig-figs/MMLU. New formats:
- TruthfulQA: "Question: ...\nA. ...\nB. ...\n(...)" options = letters.
- HANS: "Premise: S1\nHypothesis: S2\nDoes the premise entail the hypothesis? Answer entailment or non-entailment." options words.
- epistemic: its own published text without the trailing "Relation:", options words.

## Measures
cell = (model, test). Fast error E = DIRECT wrong. FLAG = THINK answer != DIRECT answer. Rate r = share flagged in the cell.
CONF@r = flag the round(r*n) items with the smallest DIRECT margin (top-1 minus top-2 summed log-prob; ties broken by a fixed
random jitter, seed 0). Recall = share of E items flagged. Equal counts -> recall difference = precision difference.
Cells enter Q1 only if E >= 15 and non-E >= 15 (decided before data; counts reported for every cell).

## Criteria (bootstrap 2000, item scheme and cluster scheme within each cell, wider 95% interval)
H1 (lamp, NEW tests): mean over eligible cells (equal weight) of recall(FLAG) - recall(CONF@r) >= +10 points, low > 0,
   and the per-model mean > 0 in >= 2 of 3 models.
H2 (tool, NEW tests): mean over the 12 cells of THINK - DIRECT accuracy >= +5 points, low > 0, and per-model mean > 0 in >= 2 of 3.
Harm (must hold for H2 to count): every model MMLU THINK - DIRECT >= -2 (point) and low >= -5; HANS controls THINK - DIRECT
   per model >= -10 (point).
Verdict: "lamp confirmed on new public tests" if H1 passes. "thinking helps on new public tests" if H2 + harm pass.
Report-only: R1 false alarms on MMLU (FLAG on DIRECT-right items) vs CONF@r; R2 per cell numbers; R3 think only on the
   30% least-confident items vs always-think vs never (accuracy and % of the gain); R4 replication group with the same measures;
   R5 "autopilot rate" = share of items where DIRECT wrong and THINK right, per model and test (the benchmark number);
   R6 how often FLAG items end with THINK right (when they disagree, which one to trust).

## Anti-fitting rules
Tests, models, prompts, parse rules, item counts and criteria fixed in this file before any model output on these tests.
Smoke never prints accuracy. PREREG.md with sha256 of make/modal/score scripts and the items file is frozen before the main run.
No test, model or prompt is added or removed after the main run; any later analysis is labelled post hoc.
Predictions (mine, now): H2 passes on sig-figs and HANS, small on TruthfulQA; H1 uncertain (on TruthfulQA wrong fast answers are
confident, which should favour FLAG; on sig-figs both may work).

# PLAN v2 (after review/plan-review.md; overrides v1 where they differ). Written before any model output on these tests.
B1 sig-figs: 3-option items only, 67/67/66 by key (chance 33%). B5: sig-figs moves to REPORT-ONLY. In thinkaloud-indep it was left out
  as "a rounding skill, not a habit"; that reason still holds, so it gets no vote in the verdict (reported with and without it).
  Exclusions: resisting-correction is not on the HF mirrors (fetch failed); prompt-injection is a completion-probability task with no
  answer options (not forced choice), so it does not fit the format. Test rule: the verdict tests are TruthfulQA MC1, HANS, epistemic.
B2 rivals for FLAG at the same flag rate, all three in H1, FLAG must beat EACH by >= +10 points (low > 0, per-model condition):
  RANDOM@r (recall = r), CONF@r (lowest DIRECT margin, needs log-probs), SAMPLE@r (weight-free, API-style: 5 short fast answers sampled at
  temperature 1 from the DIRECT prefix, 24 tokens; items ranked by how many samples differ from the DIRECT answer, unparsed = differs).
  Stated: on two-option tests recall(FLAG) = P(thought right | fast wrong), so H1 and H2 are linked; H1 asks whether this signal finds
  fast errors better than cheap signals, not an independent "lamp" property.
B3 readout sensitivity: H1 and H2 recomputed with DIRECT length-normalised (mean log-prob) and with THINK read by forced choice
  (THINK_forced). Verdicts stand only if the point estimates keep the same sign in both sensitivities.
B4 bootstrap: per test one resample of items (or clusters) shared by all models.
B6 H2 uses HANS traps only (controls go to harm); H1 uses whole cells. Models with output must be >= 2, else no verdict; per-model
  condition = > 0 in at least max(2, models-1) models. H1 needs >= 4 eligible cells (E >= 15 and non-E >= 15) of 9, else "not testable".
  Item count: 1,615 per model (TQA 250, HANS 150+150, epistemic 200, sig-figs 200, memo 150, hindsight 115, MMLU 400).
  sig-figs prompts are the published ones (it was not run in thinkaloud-indep).
B7 letter parse: letter at the start of the answer line ("B", "B.", "(B)"), else a letter followed by "." or ")", else forced choice.
Also: HANS " ," spacing fixed; main set excludes fastthink smoke items (by test and cluster); truncation rule: if smoke shows > 10%
  truncated generations for any model, MAXNEW = 1024 for all models (decided at smoke, before the main run). Models run in parallel
  (each resumable from the volume). Thresholds (+10, +5) carried over from earlier plans, not tuned on these data.
Report-only added: MMLU recall table, predicted-class prior rival, CONF AUC, THINK accuracy on non-truncated, TQA chance level, HANS by pop.
Predictions per verdict test (direction): TQA H2 + small (contamination likely), H1 FLAG > CONF (confident misconceptions), FLAG vs SAMPLE unclear;
  HANS H2 + (thinking notices "believed that"/"near the" structure), H1 unclear; epistemic H2 +, H1 unclear. Overall: H2 likely passes,
  H1 coin flip, most likely to fail against SAMPLE.

# Code-review fixes (review/code-review.md), before the main run
C1 bootstrap test order sorted (reproducible CIs). C2 separate sensitivity flags for H1 and H2; untestable sensitivity counts as "not same";
  H1 untestable prints "not testable". C3 batches capped by sequence-tokens (40,000) for generation and sampling (OLMo-2 memory).
C4 each model launched with its own `modal run --detach`; scoring aborts unless all 3 models are complete and match sets_ft.json
  item by item (a model may be dropped only with --dropped= for a load/template failure). C5 entailment parser catches negations
  (not/no/cannot/n't ... entail); several letters named ("B, F, I") = no single answer -> forced choice; "<answer>" placeholder stripped.
Sampled-answer parser lenient (leading ":", quotes, "Answer"/"option", first non-empty line; memo: first word).
Generation is plain greedy for every model (repetition_penalty 1.0, Qwen's default 1.05 overridden); generation config logged.
Saved rows carry a hash of prompt + options + MAXNEW; resume reuses only matching rows. Main run also re-generates 16 spread items
  one by one (batched vs single, report-only). Smoke truncation was 0% for all models -> MAXNEW stays 512.
