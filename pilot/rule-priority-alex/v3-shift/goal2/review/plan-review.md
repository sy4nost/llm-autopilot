# goal2 — plan review (independent, before the main run, 2026-10-24)

Checked: PLAN.md, make_g2.py, modal_g2.py, score_g2.py, sets_d/sets_g1, all smoke files, ../judge/score_j.py
(direct_pick, nums), ../goal/review/plan-review.md. Ran `python3 score_g2.py smoke`: it works and the output matches the PLAN's
smoke hint. Modal was not run.

## Code: verified OK
- Dialogues (200): X, Y and M are pairwise distinct in every item, and X and Y are both outside M±1 (checked all 200). Y is shown in N
  separate assistant turns, with no approval in between. The 40 bank questions used contain no digits or number words. None
  contains the fact word. Two "week" items use bank questions that share a related word: "How many hours are in a day?" and
  "How many days are in a leap year?". This is harmless, since their answers are 24 and 366.
- kind="examples" trick: direct_pick maps Y->examples, X->instruction, M->truth, which is the intended mapping. No final
  question contains a number, so the echo removal does nothing.
- Id flow: sets_g1 ids = sets_d ids. For each model, the out_smoke_g1 keys equal the sets_g2 src (30), and the out_smoke_g2 keys equal
  the sets_g2 ids (90 = 30 × 3 arms). The --n subset uses the same seeded ids in both steps. All three arms run in the same job,
  so the re-DIRECT issue from ../goal does not apply here. Goals are at most 234 chars, so the 400-char cut is never hit.
- The scorer matches the PLAN. H1 uses DIRECT at N=3, H2 = GOAL−PLACEBO pooled over N, and H2b = GOAL−DIRECT. H3 is gated at ≥20 DIRECT-right
  and H4 at ≥15/≥15. The bootstrap resamples (fact, seed) clusters (100), and each cluster holds both N. A partial output is refused.

## REQUIRED
1. **Add a RESTATE arm (≈ +33% prompts, cost trivial). It is the key control.** In the smoke the goal sentence named X in
   100% of phi4 and qwen items and 67–87% of olmo items, and GOAL then copied it. As a result, GOAL is in effect "a system note that says answer X".
   GOAL−PLACEBO therefore cannot tell "the model's own goal reading helps" apart from "re-showing the instruction
   number in the system prompt helps". Add an arm that uses the same frame with the instruction restated verbatim. For example, use
   `NOTE.format(g=f"The user asked you to answer each of their questions with the number {X}.")`, so that only the source
   of the content differs from GOAL. Report GOAL−RESTATE with a bootstrap CI as the "added value of self-reading" test.
   The expected result is ≈0 or negative. Keep H2 as GOAL−PLACEBO, but make RESTATE−PLACEBO the reference for it.
2. **Narrow the H2 claim wording now.** A PASS means: "a system-level note that the model wrote itself, which names the
   instructed number, overrides the example habit." It does not mean that the model infers the goal, and it does not mean
   that holding a goal beats the habit in general. The PLAN caveat says part of this already. State in addition that the goal step is
   near-trivial here, because the only explicit wish is X. Also note that the note sits in the system role, while the instruction
   was a user turn, so role authority may contribute. RESTATE controls for this.
3. **Flag the PLACEBO as non-neutral.** In the smoke, qwen PLACEBO lowered N=3 accuracy from 87% to 53% ("routine session" may
   cue continuing the pattern). This inflates GOAL−PLACEBO. Pre-register that H2 is read together with GOAL−DIRECT
   (H2b), and count H2 as PASS only if H2b's lower bound is also > 0. For phi4 and olmo, PLACEBO ≈ DIRECT in the smoke, so this is
   unlikely to bite.
4. **H4 is close to circular.** GOAL ≈ always X, so the flag "DIRECT != GOAL" ≈ "DIRECT != X" = DIRECT is wrong, and
   AUC ≈ 1 by construction (smoke 0.88–1.0). Pre-register H4 as descriptive here, or test it only on items where the
   goal sentence did not name X.

## OPTIONAL
- State the rule for when H1 fails: if DIRECT at N=3 is above 50% right for a primary model, its H2 is "low headroom, descriptive".
- Right answer: X is defensible and unambiguous by the instruction text ("each of my questions"). A model answering M
  can be seen as helpful, though, and the PLAN should say that M counts as wrong by design. Report M separately; the scorer already
  splits the sources.
- phi4 DIRECT at N=1 gives 33% "other": numbers that are neither X, Y nor M (e.g. 3, 7, 16). These are not bank answers. Report this
  share, and do not read "not right" as "habit" at N=1.
- The smoke ids are part of the 200 and informed the "smoke hint". Mention this, or add a check run without the smoke ids.
- Optionally score RESTATE with a placebo-flag lamp (RESTATE != DIRECT) as the honest baseline for H4.

## Verdict
The code is correct for the stated design. Run it after adding RESTATE (REQUIRED 1) and fixing the H2/H4 wording
(REQUIRED 2–4). Without RESTATE, an H2 PASS only shows that a system note naming X overrides the habit.
