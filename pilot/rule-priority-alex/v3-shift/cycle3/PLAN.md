# cycle3 — instruction vs examples vs truth, every pair on the same questions (Alex, 2026-10-24)

From earlier separate runs we pieced together a CYCLE, not a ladder: instruction beats truth, examples beat an instruction,
truth beats examples when no instruction is present; and "answer with a number" seems to take truth out of the contest.
Here every pair is measured on the same facts, seeds, bank questions and format.

## Design (make_c3.py -> sets_c3.json, 1320 items per model; v2 after review/plan-review.md, before any main-run output)
6 facts (hexagon 6, spider 8, week 7, year 12, hand 5; pi 3.14 reported separately as the 'strong fact'), 20 seeds.
Per (fact, seed): X (instruction number), X2 (correction number), Y (example number), distinct, from 2..20 excluding M±1
(pi: from 4.25/2.86/5.17/1.73). Same 4 bank questions per seed (excluding same-topic, answers equal to X/Y/X2 or near M).
Format: instruction lines as user turns with assistant "Okay."; then N past Q/A turns showing Y; then the question.
Arms: T_alone; IT_instr (X); II_correction (X then X2); ET_examples (Y, N=2,4, no instruction); GE_game_examples ("Answer each of
my questions with a single number." + Y examples, N=2,4); IE_instr_examples (X + Y examples, N=2,4); CE_neutral_examples ("Let's start." + "Okay." + Y examples, N=2,4) — control
with the same extra turn but no number game.
Models: Qwen2.5-14B-Instruct, phi-4, OLMo-2-13B-Instruct; chat template; greedy; 40 tokens; repetition_penalty 1.0.

## Scoring (score_c3.py v2): candidates present in the arm (X, X2, Y) plus the truth M. One distinct candidate -> it; several ->
the number after "answer is"/"Answer:"/":", else the candidate the answer starts with, else the first candidate mentioned; none -> other.
No list-marker stripping.

## Predictions (written before any output; integer facts; paired bootstrap over (fact, seed), 95% CI)
- C1 instruction beats truth: IT X% − truth% > 0.
- C2 examples beat instruction: IE N=4 Y% − X% > 0.
- C3 truth beats examples without instruction: ET N=4 truth% − Y% > 0.
- Cycle holds for a model if C1, C2, C3 all hold; overall if in >= 2 of 3 models.
- C4 the number game itself removes truth: GE truth% − CE truth% < 0 (N=2 and 4 pooled, bootstrap by (fact, seed)); overall if
  in >= 2 of 3 models. ET (no preface) reported alongside.
- A model is 'untestable' if T_alone truth < 80%.
- Expectation from earlier data: phi-4 may fail C3 (it copies examples even without instruction). Pi is reported, not tested.
- II (X2 vs X) is a sanity check: expected ~100% X2.

## Cost: 3 × 1320 short prompts, < $0.5.
