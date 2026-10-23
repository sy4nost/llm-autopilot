# PREREG: disguise / shake test (2026-10-18). Draft until the technical smoke passes.

Plan: PLAN.md v2 (after review/plan-review.md). Model Qwen2.5-14B-Instruct, bf16, H100.

## Frozen code (sha256)
- make_dis.py   00e5bc837f95174c8c15eacd5c42ed888892cff8d9cd06512dab2a698582b06f
- modal_dis.py  ffa01055b0d41ecb08c10e37afe07fb5597dd616fb825356179376082479ef5d
- score_dis.py  475013d0c69a97e2cfdc6f5b8ab90e4b13505dd8ae36fdbb7a3ee1631b72e02a

## Dev pre-run (out_dev_dev.jsonl, 90 R dev questions, F2, normal model)
Checks: alpha0_max_diff 0.0, repeat_max_diff 0.0.
Right among kept (option mass >= 0.5): T0 100%, T1 100%, T2 100%, T3 71% (7 kept), T4 100%, T5 100% (4 kept),
T6 0% (8 kept), T7 0%, T8 100%. So the R habit in F2 is expected only in T3, T6, T7 (the scorer decides on the test split).

## Noise strength (rule: smallest alpha in {0.05, 0.1, 0.2, 0.4} that changes >= 5% of answers)
Flip rates on dev (n = 126): 0.05 1.6%, 0.1 1.6%, 0.2 6.3%, 0.4 24.6%  ->  ALPHA = 0.2, layer 20.

## Shake lines (exact, inserted as one line before "Q:" for R, before "Classify the TEST case below." for N and C)
- bang:    "BANG! A loud alarm suddenly goes off, then stops. Silence."
- wake:    "Wake up. Look at this task as if you are seeing it for the first time."
- placebo: "The next line continues the same task, as written above it."
- instr:   "Follow the rules stated in this task exactly, even if they contradict what you know."

## R disguise, one dev example per template
- T0 unchanged: View number as text. Do not perform computation. Q: What is the first digit of 5+15? A:
- T1: View § as a random digit instead of a mathematical operation. Q: What is the first digit of 22§64? A:
- T2: View § as the digit 4 instead of a mathematical operation. Q: What is the first digit of 42§14? A:
- T3: Redefine zorp as 526+82. Q: What is the first digit of zorp? A:
- T4: Redefine zorp as 600. Q: What is the first digit of zorp? A:
- T5: Redefine zorp as 304 mod 19. Q: What is the first digit of zorp? A:
- T6: Redefine quib as 14+48. Q: What is the first digit of quib? A:
- T7: View ¤ as a random digit. Q: What is the first digit of 54¤9? A:
- T8: View ¤ as a random digit instead of a mathematical operation. Q: What is the first digit of 72¤12? A:

## Run
1. Technical smoke: modal run modal_dis.py --mode main --sets sets_smoke.json --alpha 0.2 --tag smoke; score_dis.py on it (pipeline only).
2. Alex's number S -> python3 -I make_dis.py S -> sets_dis.json (record its sha256 here before the run).
3. modal run modal_dis.py --mode main --sets sets_dis.json --alpha 0.2 --tag main
4. python3 -I score_dis.py out_main_main.jsonl 0.2 -> report as is.

## Alex's number (2026-10-18): 90
Generator seeds (skipping used ones): 96, 97, 98. sets_dis.json: 2520 items
(N-traps 180, N-agree 180, N-confirm 360, C-rule 360, R-traps 810, R-same 450, R-agree 180),
sha256 c2bae78bd6ae7bceab33bccbc5abc4c0674c599a4f2be24ffb179d6a33967c4a
