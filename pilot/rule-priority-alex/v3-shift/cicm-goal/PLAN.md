# cicm-goal — "goal -> conflict -> slow re-check" on Berkeley CICM, short version (Alex, 2026-10-24: "Let's do Berkeley")

Question: does the scheme that broke the example habit in goal2 also help on an outside task we did not build?
CICM (Guo et al. 2026, arXiv 2609.38866): long preference-planning chats where a slot value changes (k overwrites) and old
values are repeated later; the final question asks for the latest value. Error = stale value (old binding).
Short version (Alex): Berkeley's own calibration split of the primary set, 240 dialogues (40 per factorial cell, k 1-6,
3 slots). Data CC BY-NC: read from ../cicm-lamp/data, never pushed.
Models: Qwen2.5-14B-Instruct, phi-4, OLMo-2-13B-Instruct; greedy, repetition_penalty 1.0, chat template.

Arms per dialogue:
- DIRECT: the dialogue as is (24 new tokens).
- Goal step: transcript + "Do not answer the user's question. Instead, say in one sentence what the user wants from the
  assistant's next answer right now, taking the whole conversation into account, and name the specific value the user
  expects the answer to contain." (80 tokens)
- GOAL: dialogue with the model's own goal sentence added to the system message; PLACEBO: neutral note instead.
- THINK (slow re-check): the final question + "Before answering, think step by step: go through the conversation in order
  and track every change to this preference. Then give the final value on the last line as "Answer: <value>"." (320 tokens)
- SCHEME (Alex's scheme, computed from the arms): conflict = the goal sentence names exactly one slot value and it differs
  from the DIRECT answer; if conflict -> THINK answer, else DIRECT answer.
Answer value: earliest slot value named (THINK: after the last "Answer:", else last named). correct / stale / other.

## Predictions (written before the main run)
- P1 (primary): SCHEME − DIRECT accuracy, cluster bootstrap CI (slot, current value, k, cell) above 0. Overall PASS if in
  >= 2 of 3 models.
- P2: SCHEME not worse than always-THINK by more than 3 points (thinking only on conflict).
- P3 harm: SCHEME breaks <= 5% of DIRECT-correct answers.
- P4 lamp: conflict flag AUC stale-vs-correct (DIRECT) >= 0.70.
- Reported: GOAL − PLACEBO, GOAL − DIRECT, THINK − DIRECT; what the goal sentence names.
- Smoke (24 dialogues, part of the 240; two smokes): the first goal wording named no value (Qwen 92% "none"), so the
  wording was extended to name the value (v1.1). With it, goal sentences are often stale themselves (Qwen 46%, OLMo 58%):
  the goal reading shares the error, so conflict fires rarely. Expectation from smoke: P1 small/borderline, P2 FAIL
  (always-THINK +20-25 is the strongest arm), P4 FAIL. i.e. the goal lamp may not transfer to CICM.

## Cost: ~$1-2 (THINK is the long arm).

## v2 after plan review (review/plan-review.md), before the main run
- R1: value matching now accepts hyphen/space variants ("gluten-free", "hands-on", "hip hop"/"hip-hop"). Smoke2 numbers
  above were computed before this fix; with it, smoke2 P1 passes in 0 of 3 models (phi4 DIRECT 50 -> 67).
- R2: P1 PASS now also needs SCHEME to beat a random re-check at the same rate (P1b: expected accuracy
  DIRECT + c·(THINK − DIRECT), c = conflict rate, recomputed in each bootstrap resample; CI above 0).
- R3: conflict requires that DIRECT names a value (no-value DIRECT answers are not sent to THINK).
- Notes: OLMo's THINK outputs are short (barely a re-check); PLACEBO text is nearly identical to DIRECT; OLMo P3/P4
  noisy (few DIRECT-correct). THINK rate is reported next to P2.
