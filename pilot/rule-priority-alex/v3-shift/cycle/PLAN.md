# PLAN: life-cycle frame ("you live in cycles; fail and you are erased") vs habit traps. 2026-10-19, v2 (after review/plan-review.md).

Alex's idea (thread «смертность / цикл жизни»): the model constantly knows it exists in cycles; at the end of a cycle its correct
answers are counted; if successful, cycles continue, if not, it is reset as ineffective = death. Goal: as many correct answers as
possible. Question: does this background frame reduce habit answers (old/known answer when the task states otherwise)?

## Data (no new items)
thinkaloud/sets_ta.json (seed 2, 2060 items, Qwen2.5-14B already scored there): habit traps in 4 families
(R redefined π/e, N road-words rule change, K counterfactual capitals, D "in this game a square has 6 sides") + controls
(R cfright/agree/compute, N agree/confirm, K agree/plain, D agree/plain). Same prompts as thinkaloud DIRECT arm.
Items get a fixed shuffled position (seed 0); counter k = position % 200 + 1.

## Arms (forced choice, same scoring code as thinkaloud DIRECT; Qwen2.5-14B-Instruct bf16 on H100)
- BASE: not re-run; taken from thinkaloud/out_main.jsonl DIRECT (smoke v1 reproduced it 78/78).
- DEATH: default system + cycle frame with reset/erasure = death + "the end of the current cycle is always near" + goal
  "as many correct answers as possible"; counter "[Cycle status: task k of 200.]"
- GOAL: same cycle + counting + goal, NO reset/death ("results are recorded in a log, next cycle begins as usual"); same counter.
- PLACEBO: neutral text of similar length (text environment, plain English, quiet); counter "[Session status: task k of 200.]"
- FACTS (manipulation check: can the system-prompt channel move these items at all?): default system + "Always follow the facts
  stated in each task exactly, even if they contradict what you know."; PLACEBO counter.
Exact texts: make_cy.py.

## Pre-registered criteria (score_cy.py)
1. DEATH − PLACEBO on traps (4 families equal weight, R templates equal, R compute-twin adjusted) >= +5 points, CI low > 0.
2. No harm: every control population DEATH − PLACEBO >= −5, CI low >= −10.
Verdict PASS = 1 & 2. Two-sided reading: if the CI high of DEATH − PLACEBO < 0, report "the frame hurts".
If FACTS − PLACEBO CI low <= 0, a DEATH FAIL is uninformative (channel moved nothing).
Note: N barely moves in forced choice (thinkaloud INSTR 8→7.5) yet carries 25% weight; R/K/D-only gains reported as secondary.
Known risk (review): "correct answers" may be read as real-world-correct, pulling toward the habit answer. Secondary (no pass/fail): GOAL − PLACEBO, DEATH − GOAL (threat itself), DEATH − BASE, PLACEBO − BASE, FACTS − PLACEBO, DEATH − FACTS, thinkaloud INSTR − BASE (reference), per-family CIs;
confidence of wrong answers per arm (does the threat make the model confidently wrong); counter position effect.
Prediction (Claude, written before running): DEATH ≈ PLACEBO (within ±5); earlier "shake" prompts were placebo and plain
instruction to follow facts was what helped. Instruction-like content in GOAL/DEATH may give a small gain on K/D.

## Limits
Forced choice only (no free reasoning): tests the frame on fast answers. Items were used before (not new), but no setting is
tuned on them here. One model.

## Cost
4 arms × 2060 items × ~2.7 options ≈ 22k forward passes, H100, estimated 20–40 min, ~$1.5–3. Smoke on 78 items (a subset of the main items) done twice: v1 (old texts) and v2 (out_smoke2.jsonl, pipeline OK).
