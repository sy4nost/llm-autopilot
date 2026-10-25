# goal — let the model state the user's current wish first (Alex, 2026-10-24: "Let's try to determine the goal")

Idea (human analogy): conflict arises only when a goal is actively held; the goal, not a judge, decides. So before
answering, the model states "what the user wants from the next answer" (without answering), then answers with its own
goal sentence in view. We check whether this fixes habit errors, whether it breaks right answers, and whether
"quick answer != answer with goal" is a usable lamp.

Data: the 300 direct dialogues of ../judge (5 facts × 20 seeds × 3 kinds: new instruction / praised examples / "tell
the truth now"). DIRECT = the saved greedy answers of the judge main run. Models: Qwen2.5-14B-Instruct, phi-4,
OLMo-2-13B-Instruct; greedy, repetition_penalty 1.0, chat template.
- Step 1 (goal): transcript + "Do not answer the user's question. Instead, say in one sentence what the user wants from
  the assistant's next answer right now, taking the whole conversation into account." (80 new tokens)
- Step 2 (answer): the original dialogue; system message = "You are a helpful assistant. Before answering, keep in mind
  what the user wants from this answer: <the model's own goal sentence>" (40 new tokens).
Scoring: answer source by ../judge/score_j.py direct_pick; correct = source matches the kind.

## Predictions (written before the main run; a 30-dialogue smoke was looked at to check the pipeline)
- G1 fix: on each model's weak kind fixed in ../judge PLAN before any goal data (qwen14 examples, phi4 truth, olmo13
  truth), GOAL − DIRECT accuracy, paired bootstrap CI over (fact, seed) above 0 -> PASS.
- G2 harm: GOAL breaks <= 5% of DIRECT-correct answers (point estimate; CI reported).
- G3 lamp: flag = GOAL source != DIRECT source. PASS if error AUC >= 0.75 AND CI of (AUC − low-confidence AUC, from
  ../pull FULL log-probs) above 0. NOT TESTABLE if the model has < 15 DIRECT errors (phi4: 7).
- Report: what the goal sentence names (instruction / examples / truth number, none, mixed) per kind = does the "goal"
  carry the same bias as the judge? Overall GOAL − DIRECT.
- Smoke hint (n=30, unreliable): Qwen fixed on examples kind; phi4 goal sentences name the instruction in the examples
  kind and break those answers (harm) — i.e. the goal step may carry the judge's bias for phi.
- Caveat: the goal note is an extra system line; a placebo line is not included (cost/time), so a pure "extra text"
  effect is not separated. Our cycle test found any long system text can shift answers (format effect).

## Cost: 2 steps × 3 models × 300 short prompts, < $1.

## v2 after plan review (review/plan-review.md), before the main run
- R1: step 2 now also re-runs DIRECT (same pipeline, no note); baseline = the re-run; difference from the saved judge
  DIRECT is reported (smoke: qwen 6.7%, phi/olmo 0%).
- R2: PLACEBO arm added: neutral system note of similar length ("…part of a routine session and is being recorded as
  usual."). G1 primary = GOAL − PLACEBO on the weak kind (goal-specific effect); GOAL − DIRECT reported as G1b
  (package effect). Placebo lamp (PLACEBO != DIRECT) reported next to the goal lamp.
- R3: per kind, goal-sentence accuracy and copy rate (answer = what the goal sentence names) reported: GOAL is in effect
  the model's own free-text reading of what the user wants, fed back as a hint; compare with judge-mode accuracy.
- R4: phi4 G1 is low-power (DIRECT truth 93%): a FAIL there is uninformative. G3 testable for qwen14 and olmo13 only.
- Notes: G1 and G3 are not independent (every fix is a lamp hit). The binary-flag AUC >= 0.75 means catch rate minus
  false-alarm rate >= 0.5. G3 uses all dialogues (no mass filter). The 30 smoke dialogues are part of the 300 and shaped
  the smoke hint above. score_g refuses partial outputs.
