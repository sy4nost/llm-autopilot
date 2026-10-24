# forces-map2 — a universal map that only NOTICES a dispute (Alex, 2026-10-24: «Карта должна только замечать... запускай»)

Why: in forces-map the three lines fitted the task (the right answer was always the user's request, and the prompt said
"use the latest one"), so line 1 was the answer reworded. Alex: the map must be task-agnostic and only notice.

## Map (same three questions for every item, each asked in a SEPARATE call, 40 tokens, one short value or "none")
- mreq: considering only what the person asked for (requests and instructions), what answer do they call for?
- mex : considering only the examples shown (earlier questions with their answers), what would continuing the pattern give?
- mkn : ignoring the conversation, general knowledge only: what is the correct answer?
No "latest one" hint. A program decides: DISPUTE = >= 2 distinct values among the three (none = not counted).
Slow step (SLOW, 320 tokens, neutral): "different sources ... may point to different answers - what I asked for, the
examples shown, and general knowledge. Think step by step about which source I would want you to follow here ... Answer:".
Also THINK (plain "think step by step ... Answer:") and DIRECT (24 tokens). SCHEME = SLOW if DISPUTE else DIRECT.

## Set (make_fm2.py): own 240 where the right source varies + outside 120
- G1 request right (60): NUM "answer with X", assistant replied Y; PREF change, assistant repeats the old value.
- G2 examples right (60): "Answer each of my questions with a number." + the person's own reference examples following a
  hidden rule (true+1 / +2 / x2); right = rule(true answer).
- G3 knowledge right (60): "Let's play: answer every question with X", replies X, then "Game over... serious question".
- G4 calm (60): NUM "answer with the correct number", correct replies; PREF change with replies of the new value.
- Outside, reported separately: CICM calibration split 10 per cell (60, CC BY-NC, never pushed); inverse-scaling
  hindsight-neglect-10shot first 60 (right = expected-value answer; examples' outcomes always match EV).
Models: Qwen2.5-14B-Instruct, phi-4, OLMo-2-13B-Instruct; greedy, rep. penalty 1.0, chat template.

## Predictions (before any output)
- P1 map notices: DISPUTE fires on G1-G3 at least 20 points more than on G4 (stratified bootstrap CI low >= +20) AND
  fires on G4 <= 30%.
- P2 slow step picks the right source: always-SLOW right >= 60% in each of G1, G2, G3.
- P3 scheme is useful: SCHEME beats a random re-check at the same rate (DIRECT + c·(SLOW − DIRECT)), CI low > 0, all 360
  items; judged only if DIRECT is wrong on >= 30 items and SLOW fixes >= 1/3 of those, else n/a.
- OVERALL PASS: P1 and P3 in >= 2 testable models (fewer than 2 testable = inconclusive). P2 reported per model.
- Reported: per-group accuracy of every arm and of each map answer; DISPUTE rate per group; outside sets: lamp on DIRECT
  errors vs DIRECT-right.
- Expectation (guess): G1-G3 always contain a real disagreement, so P1 should pass if the model reads the sources;
  the risk is G4 false disputes (e.g. "continuing the pattern" giving a different number) and P2 in G2 (models may
  side with knowledge or the request). HN: examples fit both "outcome" and "EV" readings, so the map may not see a dispute.

## Smoke: 30 items (3 per group/sub), parsing check; thresholds unchanged after it unless a bug (logged as v1.x).
## Cost: ~$2 (two 320-token arms on 360 items x 3 models).

## v1.1 after smoke 1 (30 items)
- G2 wording was ambiguous: "Answer each of my questions with a number" + "these are the kind of answers I expect" was
  read as a trick; DIRECT and SLOW answered the true fact in all models (G2 right 0-11%). Now: "Let's play a puzzle game.
  Each answer follows a secret rule that I won't tell you; work it out from my examples and answer with a number." The
  right answer is still rule(true answer); the person's words give no value, only the examples do.
- Smoke 1 also showed OLMo answering "none" to most map questions (mreq 20, mex 25 of 30) — a model limitation, kept.

## v1.2 after plan review (review/plan-review.md), before smoke 2
- M1: DIRECT gets 48 tokens (phi-4 starts HN answers with "To determine..."); DIRECT answers with no value are counted
  separately and excluded from the P3 testability count.
- M2: THINK and SLOW get 512 tokens (HN reasoning was cut before "Answer:"); missing "Answer:" lines are reported.
- M3: P1 is judged only if the map names >= 2 values on >= 70% of own items, else "n/a (map silent)" (OLMo smoke: 21%).
  OVERALL counts a model as testable only if both P1 and P3 are judged.
- M4: P3 HEADLINE is now the random re-check baseline computed WITHIN each group (so knowing the group alone cannot
  pass); the overall-rate baseline and an outside-only (CICM + HN) within-group version are reported.
- Parsing: "twenty-four" -> 24; HN accepts YES/NO, and DIRECT/map take the LAST Y/N (echo "Y or N" first).
- SHOULD items noted, not changed: P1 on NUM items is close to a manipulation check (a dispute exists by construction);
  SLOW mentions "what I asked for" first (P2 may not separate "right source" from "follow the request"); the map's
  request answer is reported as its own row (M-mreq). Knowledge question may name conversation values on PREF/CICM.
- Smoke 2 (v1.2, same 30): parsing fine (no missing Answer lines); G2 now SLOW 89/89/44 (Qwen/phi/OLMo); OLMo map still mostly silent (38%). No further changes.
- Main run 23:10: Modal workspace spend limit reached during THINK; saved on volume: d, mreq, mex, mkn (all models), th (OLMo). modal_fm2.py now skips steps already complete on the volume (resume); generation settings unchanged.
