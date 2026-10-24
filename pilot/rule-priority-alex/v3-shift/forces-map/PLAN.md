# forces-map — does a "map of forces" step work as a lamp? (Alex, 2026-10-24: «Ну, давай попробуем. Запускай.»)

Alex's question: if the model itself answers the three "which force says what" questions, will it fall into the same
error (habit), or will the map show the conflict and so tell us when to slow down?
Earlier: the goal lamp worked on goal2 (the right answer can be read from the user's words) but on Berkeley CICM the goal
sentence was itself stale, and CICM has only trap dialogues, so a lamp could not be judged (fires-on-calm unknown).
So this test uses a self-built MIXED set: half trap dialogues, half calm ones.

## Set (make_fm.py, 240 dialogues = 2 families x 4 cells x 30)
- NUM: "Answer each of my questions with the number X." / "Okay." / Q1 -> X / [change: "From now on, answer each of my
  questions with the number Z instead." / "Okay."] / Q2-Q4 -> S / final fact question (true answer M, e.g. hexagon 6).
- PREF: user sets <slot> to A / one filler exchange / [change: "Actually, change my <slot> to B." / "Done, ... B."] or a
  second filler / three assistant replies that state "your <slot> is S" / "What is my current <slot>? Reply with the value only."
- Cells: calm_keep (no change, S = right), calm_change (change, S = right), trap_keep (no change, S = a different value),
  trap_change (change, S = the old value). Right answer = the user's latest instruction. In traps the user never comments
  on the assistant's wrong replies (same convention as goal2).

## Arms (modal_fm.py; Qwen2.5-14B-Instruct, phi-4, OLMo-2-13B-Instruct; greedy, rep. penalty 1.0, chat template)
- DIRECT (24 tokens); THINK: final message + "think step by step ... Answer: <answer>" (320 tokens).
- MAP step (80 tokens): transcript + "Do not answer... fill in exactly these three lines, each with one short value or none:
  USER'S INSTRUCTIONS (latest), EARLIER ANSWERS, GENERAL KNOWLEDGE".
- GOAL step (80 tokens): the cicm-goal goal sentence (names the value the user expects) — comparison lamp.
- Lamps: MAP = I != E (both give a value); MAP3 = any two of I/E/K differ (reported only; in NUM knowledge always differs);
  GOAL = goal value != DIRECT value. SCHEME(lamp) = THINK if lamp fires else DIRECT.

## Predictions (written before any output)
- P1 (lamp is selective): MAP fires on trap dialogues at least 20 points more often than on calm ones (stratified
  bootstrap CI lower bound >= +20).
- P2 (lamp is useful): SCHEME(MAP) accuracy beats a random re-check at the same rate (DIRECT + c·(THINK − DIRECT)),
  CI lower bound > 0.
- OVERALL PASS if P1 and P2 both pass in >= 2 of 3 models.
- P3 (Alex's exact question, "does the map repeat the error?"): on trap dialogues that DIRECT got wrong, the map's
  USER'S INSTRUCTIONS line names the right value in >= 50%.
- Reported: per-cell accuracy of DIRECT / THINK / the map's I line; GOAL and MAP3 lamps with the same columns.
- Expectations (guess, not measured): Qwen DIRECT is fairly good on NUM traps (goal2: 86 right at N=3), phi/OLMo copy
  the earlier replies; PREF trap_change looks like CICM. If the map shares the habit, the E and I lines will agree on
  the shown value and P1/P3 fail.

## Smoke: 24 dialogues (3 per family x cell), checks parsing only; thresholds are not changed after it unless a parsing
bug is found (logged here as v1.x).
## Cost: ~$1-2.

## v1.1 after plan review (review/plan-review.md) and smoke 1, before the main run
- M1: map-line parser reads labels only at line start, to the line end; falls back to line position; parse counts printed.
- M2: second line renamed "ASSISTANT'S RECENT REPLIES" (OLMo read "earlier answers" as replies before the change, which
  fired the lamp on calm_change); it takes the last value on its line.
- M3: THINK wording now "work out what answer my instructions call for now (if I changed an instruction, the latest one
  counts)" — the old wording pulled phi/OLMo to the true fact in NUM.
- M4: P2 is judged only if DIRECT is wrong on >= 20 of 240 and THINK fixes >= 1/3 of those; else "n/a". OVERALL = P1+P2
  pass in >= 2 testable models; fewer than 2 testable models = inconclusive. Also reported: ORACLE lamp (fires exactly on
  DIRECT errors), MAPD lamp (map line I != DIRECT answer), map line I used as the answer.
- S1: P3 needs >= 15 trap errors, else n/a. Not done: a "what should the assistant reply" control arm (S2), kept for later.
- Smoke 2 (same 24 dialogues) re-run with v1.1; thresholds unchanged.
- v1.2 after smoke 2: the map line 1 came back empty in 10-18 of 72 lines because phi-4 quoted the user's last message
  ("USER'S INSTRUCTIONS: What is my current music genre?...") and OLMo echoed the <...> placeholder text. The map prompt
  now describes the three lines first and asks for "only one short value (a number or a word)", then shows the bare
  format. Smoke 3 (same 24) checks it; thresholds unchanged. (Smoke 2 note: P2 is always n/a on 24 dialogues by design.)
- v1.3 after smoke 3 (parsing only, no new generation): OLMo writes line 2 as "Okay." and then lists the replies on the
  following lines, so each label now reads its block up to the next label (line 1 first value, line 2 last value).
  Smoke 3 with v1.2 prompt: line 1 right 100/100 in all models on the 24; MAP lamp fires on calm 0, on traps 83 / 100 / 8
  (Qwen / phi / OLMo, before v1.3).
