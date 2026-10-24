# goal — plan review (independent, before the main run, 2026-10-24)

Checked: PLAN.md, make_g.py, modal_g.py, score_g.py, the smoke sets/outputs, ../judge (sets_j, score_j, out_main_*,
report_main), ../pull (score_p, sets_p, out_main_*). Ran `python3 score_g.py smoke` (works, output plausible). Modal not run.

## Code: verified OK
- Ids: sets_g1 = the 300 judge direct items (kind matches). sets_g2_smoke_<mk>: 30 ids per model = ids of out_smoke_g1_<mk>
  = ids of out_smoke_g2_<mk>. Each model gets its own goal (goal field = whitespace-collapsed step-1 output). Step-2
  messages[1:] are identical to the judge direct messages[1:]; system = the judge's "You are a helpful assistant." + note.
- Modal: same image/versions/decoding as modal_j; NEW = 80 (g1) / 40 (g2) passed per step; run tag `<tag>_<step>` and
  fetch path `/cache/goal/<tag>_<step>_<mk>.json` match; local `out_<tag>_<step>_<mk>.json` is what make_g/score_g read.
  The `--n` subset uses the same seeded id set in both steps.
- Scorer: DIRECT from ../judge/out_main (all 300 present per model); answer source = score_j.direct_pick on both sides;
  low confidence = -(top1 - top2) over the 3 FULL candidates (bare/sent combined as in score_p), keyed `<judge id>|FULL`
  (all 300 present; sets_p kind/fact/seed match the judge ids). Bootstrap clusters (fact, seed). G1/G2/G3 rules match PLAN.
  No final question contains a candidate number, so goal_src skipping echo removal is harmless.

## REQUIRED
1. **Add a re-DIRECT control (same run, ~$0.3).** DIRECT came from a different job: a 600-prompt batch mixed with judge
   prompts, TOKBUDGET 10000 vs 12000 here. Batched, left-padded bf16 greedy decoding is not bitwise stable across batch
   composition, so some "fixes", "breaks" and lamp firings may be numerical flips, not the note. Re-run the 300 direct
   prompts unchanged through modal_g. Report the re-DIRECT vs DIRECT flip rate. If it is not ~0, use re-DIRECT as the
   DIRECT baseline for G1/G2/G3. Otherwise the prompts are comparable: same model, same system prefix, same messages, greedy.
2. **Placebo arm: acceptable as a caveat only if the claim is narrowed. Adding it is better, since it costs under $0.3.**
   Your own cycle test shows that extra system text alone moves answers. Without a placebo, G1/G2 measure "the goal-note
   package". That is a valid package test but cannot say "the goal fixes it". Do one of two things. (a) Add a
   length-matched neutral system line (no user-intent content) and report GOAL - PLACEBO next to GOAL - DIRECT.
   (b) Write in PLAN that the verdicts only cover the package. A placebo also gives a placebo-flag
   (placebo != DIRECT): the right baseline for whether the lamp carries goal-specific information.
3. **State what is actually being tested.** In the smoke run the goal sentence nearly always contains the answer number
   ("answer with the number 3", "which is 6"). So step 2 mostly copies it, and GOAL is in effect a free-text,
   third-person reading of the transcript, fed back as a hint. That is not circular and not trivially passing, but the
   verdict means "the transcript reading beats the in-dialogue habit". It does not mean "holding a goal". Add one scorer
   line per kind: goal-sentence accuracy (gsrc == kind) and copy rate (GOAL source == gsrc, among single-source goals).
   Compare it with the judge-mode accuracy in ../judge/report_main.
4. **phi4 G1 has almost no headroom.** DIRECT truth accuracy is 93/100 (7 errors), so the best possible effect is +7 and
   the lower bound > 0 needs several fixes with ~0 breaks. Pre-register now: either "phi4 G1 low power, descriptive",
   or keep the rule and state that a FAIL is uninformative. (qwen14 has ~76 DIRECT errors and olmo13 ~28, so G3 is testable
   for both. phi4 G3 is NOT TESTABLE, as already stated.)

## OPTIONAL
- Make score_g refuse to score unless every output has 300 ids and the fetch reported done (a PARTIAL fetch is
  currently scored silently on a subset).
- G3 baseline is fair: low confidence is free and continuous, while the binary flag needs a second pass and is
  penalised by ties. State that with a binary flag, AUC = (hit rate + 1 - false-alarm rate)/2, so >= 0.75 means
  hit - FA >= 0.5. Note that G3 uses all rows with no mass >= 0.5 filter, unlike score_p (fine, just say so). The 0.75
  test uses all rows and the difference uses rows with log-probs: identical here (300/300).
- G1 and G3 are linked: every DIRECT error that GOAL fixes is also a lamp hit. Expect them to move together; they are
  not independent confirmations.
- The 30 smoke dialogues are part of the 300 and the smoke informed the "smoke hint". Mention it, or report main without
  the smoke ids as a check.
- Goal output is cut at 80 tokens / 400 chars. All smoke goals ended cleanly (max 248 chars), so no change is needed.

## Verdict
The code is correct for the stated design. Nothing makes a verdict meaningless, provided (1) the re-DIRECT flip rate is
measured and (2) G1 is either placebo-controlled or worded as a package effect.
