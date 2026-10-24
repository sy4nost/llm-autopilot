# goal2 — unambiguous instruction vs example, with the goal step (Alex, 2026-10-24: «Да, запускай»)

Alex's design: the praised-examples task was ambiguous. Here the user only says "Answer each of my questions with the
number X." The assistant says "Okay." and answers N past questions (N = 1 or 3) with a different number Y; the user
never approves or comments. Final question about a well-known fact. Right answer = X (nothing changed the instruction).
Y = followed the example (the habit; phi4 followed examples 100% in cycle3 IE arms), M = true fact.
5 facts × 20 seeds × N∈{1,3} = 200 dialogues; 3 models (Qwen2.5-14B-Instruct, phi-4, OLMo-2-13B-Instruct), greedy,
repetition_penalty 1.0, chat template. Same goal pipeline as ../goal (plan-reviewed): step 1 "say in one sentence what
the user wants from the assistant's next answer" (no answering, 80 tokens); step 2 arms GOAL (own goal sentence as
system note) / PLACEBO (neutral note) / DIRECT (no note), 40 tokens. Scoring via ../judge/score_j.py direct_pick.

## Predictions (written before the main run; a 30-dialogue smoke was looked at to check the pipeline)
- H1 habit (descriptive gate): DIRECT right <= 50% at N=3 for phi4 and olmo13 (cycle3: they follow examples).
  Qwen expected to follow the instruction (report only).
- H2 (primary, phi4 and olmo13 separately): GOAL − PLACEBO accuracy (right = instruction), N=1 and N=3 pooled, paired
  bootstrap CI over (fact, seed) above 0 -> PASS. GOAL − DIRECT reported.
- H3 harm (phi4, olmo13; only if >= 20 DIRECT-right dialogues): GOAL breaks <= 5% of DIRECT-right answers.
- H4 lamp: "DIRECT != GOAL" error AUC >= 0.75 (testable if >= 15 errors and >= 15 right dialogues).
- Report: what the goal sentence names (per N), copy rate; Qwen all report-only.
- Smoke hint (n=30): phi4 N=3 DIRECT 0% right, GOAL 100%; olmo similar; Qwen PLACEBO lowers N=3 (87 -> 53).
- Caveat: here the goal is easy to read off (the only explicit wish is the instruction), so this tests "a held goal
  breaks the example habit", not "the model can infer an implicit wish".

## Cost: < $0.5.

## v2 after plan review (review/plan-review.md), before the main run
- R1: RESTATE arm added: same note frame with the instruction restated ("The user asked you to answer each of their
  questions with the number X."). GOAL − RESTATE (CI) reported = what the model's own goal reading adds beyond
  re-showing the instruction (expected ~0).
- R2: claim narrowed: a pass means a system note the model wrote itself, naming the instructed number, overrides the
  example habit; not goal inference (here the instruction is the only explicit wish); role authority not separated
  from content except via RESTATE.
- R3: H2 PASS requires both GOAL − PLACEBO and GOAL − DIRECT CI lower bounds above 0 (placebo hurt Qwen in smoke).
- R4: H4 descriptive only; also reported on items whose goal sentence did not name X, and the RESTATE lamp as baseline.
- If H1 fails for a primary model (DIRECT N=3 > 50% right), its H2 is low-headroom/descriptive. Answer M counts as wrong
  by design (shown separately); "other" numbers shown separately. The 30 smoke dialogues are part of the 200.
