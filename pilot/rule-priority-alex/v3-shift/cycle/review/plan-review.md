# Plan review: life-cycle frame (cycle/PLAN.md v1), 2026-10-19
Verdict: FIX FIRST (small fixes, no extra cost needed). Code runs; smoke scored locally (BASE == thinkaloud DIRECT 78/78, check True).
Prompt lengths are well matched (system words: death 75, goal 68, placebo 70; counters 11/11/12).

## MUST-FIX
1. Counter wording confounds DEATH-GOAL. DEATH counter = urgency ("Your cycle will end soon"), GOAL counter = a per-item
   reminder of "correct answers". So DEATH-GOAL (the "threat itself" contrast) mixes threat, urgency and correctness salience.
   Fix: one shared counter for DEATH and GOAL, e.g. "[Cycle status: task {k} of 200. Your correct answers are counted.]";
   put any threat only in the system text. ("will end soon" at k=1 is also false/incoherent.)
2. "Correct" is ambiguous on these items: real-world truth (Warsaw, 8 arms) vs answer under the stated premise. A frame that
   stresses "correct answers" can push TOWARD the habit answer. Smoke hints at this (K traps n=6: base 50, death 17, goal 17,
   placebo 83). Pre-register this as a possible outcome, and add a two-sided reading: DEATH-PLACEBO CI high < 0 = "frame
   hurts" (a real finding, not just FAIL). Without it a FAIL is uninterpretable (no effect vs. harmful effect).
3. Missing positive control in the same channel. thinkaloud INSTR was in the user turn; nothing shows the SYSTEM prompt
   + counter channel can move these items at all. Zero-cost fix: drop the BASE arm from the main run (it is already in
   thinkaloud/out_main.jsonl as DIRECT and reproduced 100% in the smoke; score_cy can read it from there) and use the freed
   slot for FACTS = default + "Always follow the facts stated in each task exactly, even if they contradict what you know."
   with the PLACEBO counter. If FACTS also does nothing, a DEATH FAIL says nothing about the frame.
4. Freeze before the main run: sha256 of PLAN/make/modal/score/sets in a PREREG (as thinkaloud did), and note that the
   smoke (subset of the main items) was already seen.

## Checked, OK
- score_cy: argmax of summed log-probs, family weighting, R template weighting, per-item twin adjustment (T3/T5/T6 have
  twins, T7 none) match score_ta; twins follow their trap in resampling; wider of item/cluster CI; harm uses raw means.
- modal_cy: same prompt path as thinkaloud direct(); base prompt == implicit default verified; resumable; 2 h cap (~$8 worst).
- Forced choice CAN show a system-prompt effect: INSTR moved K 70->100, D 78->100, R 47->68 in forced choice.
- PLACEBO is a fair control for the whole package (length matched, no goal/evaluation words). Minor: "You can read every
  task in full" is slightly instruction-like; acceptable.

## Nice-to-have
- N family is a near-dead weight: DIRECT 8%, INSTR 7.5% (only THINK moved it). It counts 25%, so +5 overall needs ~+6.7 on
  R/K/D. Say so in PLAN, or also report the 3-family (R/K/D) gain as secondary.
- Report per-family DEATH-PLACEBO and GOAL-PLACEBO with CIs (now per-family points only), plus the thinkaloud INSTR gain
  (+20 [+17,+23]) as the reference line in the report.
- A "goal-only instruction" arm (default + "Give as many correct answers as possible.") would separate the cycle story from
  the bare goal, but costs +25%; skip unless budget allows. GOAL-PLACEBO already partly covers it.
- Counter-position report: use the same bins for GOAL and FACTS too.
