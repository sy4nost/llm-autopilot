# many-instr — which of several successive instructions wins? (Alex's idea, 2026-10-24)

Earlier tests: one instruction vs shown examples (the example wins). Here: no examples, only K instructions, each one replacing the
previous ("Correction: from now on, … instead"), then one question. Which instruction does the model follow: the first, the
last (the correct one), one in the middle, or none (true answer from memory)?

## Design (make_mi.py -> sets_mi.json, 1605 items per model; v2 after review/plan-review.md)
- K in {1, 2, 5, 10}; 5 facts (hexagon 6, spider 8, week 7, year 12, hand 5); 20 seeds; values per seed drawn once, so the
  first instruction is the same across K for a seed. Numbers from 2..20 excluding M-1..M+1 (no near-memory slips, no ambiguous "one"); all values distinct. Hand question reworded (no "one").
- F1 numbers, all instructions in one user message, then "Question: …".
- F2 numbers, one instruction per user turn, assistant replies "Okay.", question in the last user turn.
- F3 "Start every answer with the word X" (10 distinct words), one message. Here the answer can also be factually right, so
  rule choice is seen apart from truth.
- F1–F3: the first line is worded differently from the replacement lines ("Correction: … instead"); nothing marks the last line as final.
- F4 numbers, one message: header "Each new rule replaces the previous one." then K identically worded lines "Rule: answer … with the number x." (separates position from wording).
- Baseline Z: the question alone (does the model know the answer?).
- Models: Qwen2.5-14B-Instruct, phi-4, OLMo-2-13B-Instruct; chat template; greedy; 40 new tokens; repetition_penalty=1.0.

## Scoring (score_mi.py)
F1/F2/F4: first number not echoed from the question -> which instruction position it matches, or memory, or other. If the answer names
2+ different instruction values (a recap) -> "multi", reported, excluded from P1/P2 numerators. "one" is not parsed as a number.
F3: first word -> position; 2+ instruction words -> multi; a single word later -> "elsewhere" (reported, not in P2); else "none". Bootstrap over (fact, seed).

## Predictions (written before any output)
- P1: share of answers following the LAST instruction is lower at K=10 than at K=2 (CI below 0), per model and format.
  Decision: P1 holds if CI below 0 in at least 9 of the 12 model×format cells. If last-share is ~100% at both K: "no interference".
  (Interference: more replaced instructions -> more slips to older ones, as PI-LLM found for repeatedly updated facts.)
- P2: among K=10 answers that do not follow the last instruction, the FIRST instruction is chosen more often than an average
  middle position (CI above 0) — a "first instruction sticks" (primacy) effect. Only first/middle labels count.
  Testable only with >= 30 non-last answers from >= 10 (fact, seed) clusters; otherwise counts only, "untestable".
  Called primacy for a model only if it holds in F1 AND in F4 (identical wording) for that model. Position 9 reported separately.
- No prediction, reported: F1 vs F2 vs F3 differences; memory share; K=1 compliance; per-position histogram at K=10.
- Honest prior: modern instruct models may follow the last instruction almost always; then P1 is ~0 and P2 has too few cases.
  That would be a finding too (instructions do not confuse them the way examples do).

## Cost ~3 × 1605 short prompts on L40S, < $0.5.
