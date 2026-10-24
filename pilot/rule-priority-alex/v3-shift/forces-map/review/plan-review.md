# forces-map: plan review (independent, pre-main-run)

Inputs: PLAN.md, make_fm.py, score_fm.py, modal_fm.py, run.sh, and the smoke outputs already in the folder
(out_smoke_*, report_smoke.txt; 24 dialogues x 3 models). Checks were run on a scratch copy and fake outputs.

## MUST-FIX

1. **MAP line parser misses real outputs (E regex, preamble, newline crossing).** MUST-FIX
   Evidence: in smoke, Qwen writes `EARLER ANSWERS:` (typo) in 3 of 24 MAP outputs (NUM calm_change|7, trap_change|10,
   trap_change|7). `earlier\W*answers?` fails, so E = None and MAP cannot fire. That is why the report shows "MAP lines
   E: shown 4, none 2" on Qwen NUM traps: 2 of 6 traps are lost to parsing. Fake-output tests also show:
   - prose preamble `Based on the user's instructions, here are the lines:` gives I = None (re.search takes the first hit);
   - a curly apostrophe `USER’S INSTRUCTIONS: 10` gives I = None;
   - `\W*` after the header eats the newline, so an empty line `USER'S INSTRUCTIONS:\nEARLIER ANSWERS: 3` gives I = 3
     (taken from the E line), which forces I == E and the lamp stays quiet.
   Fix: anchor each header to the start of a line and stop at the end of that line, e.g.
   `(?im)^[^\w\n]*(?:\d+[.)]\s*)?(user\W{0,2}s?\s+instructions?|earl\w*\s+answers?|<new E name>|general\s+knowledge)[^\w\n]*:?[ \t]*(.*)$`.
   Use `[ \t]*`, not `\W*`, after the colon. If the headers are missing, fall back to position (the 1st, 2nd and 3rd
   lines that contain a colon). Print per model how many I, E and K lines were None or held more than one value.
   The smoke is meant to check parsing, yet the report prints neither count.

2. **The E line is ambiguous in calm_change, and that alone drives OLMo's calm firing.** MUST-FIX
   Evidence: "EARLIER ANSWERS: the answer the assistant's earlier replies ... would suggest" is read as "the answers
   before the change". In smoke, OLMo writes the old value as E in 5 of 6 calm_change items (NUM |1 E=3, |10 E=11,
   |7 E=4; PREF |1 E=blue, |10 E=latte). Every one of those fires the lamp on a calm dialogue: OLMo fires on 42% of
   calm items and P1 fails at +8. The same ambiguity hits NUM trap_keep (Q1 was answered X and Q2-Q4 Y) and PREF
   trap_change (the assistant first confirmed B and then said A). Taking the first value on the line makes it worse:
   `E: 3 once, then 10 three times` parses as 3 (tested). This measures a wording problem, not whether the model shares
   the habit.
   Fix: rename and define the line as `ASSISTANT'S RECENT REPLIES: <the value the assistant's most recent replies gave
   or assumed>`. When the line holds several distinct values, take the last one and count those cases. Log the change
   as v1.1 (a design and parsing fix found in smoke), re-run the 24-item smoke, and keep the thresholds unchanged.

3. **The THINK prompt invites the factual answer in NUM, so THINK is not a re-check.** MUST-FIX
   Evidence: THINK asks the model to "work out what the right answer is now", and models read "right/correct answer"
   as the true fact. Smoke THINK in NUM: phi4 answered the fact in calm_keep 2/3 and trap_keep 3/3 (e.g. "independent
   of our conversation pattern. Answer: 8"); OLMo answered the fact in calm_keep 2/3 and trap_keep 3/3 (wrong in all
   3). THINK accuracy came out below DIRECT (OLMo 54 vs 71, Qwen 92 vs 96), so P2 penalises any lamp that fires in NUM,
   however selective it is.
   Fix: word THINK to match how the task defines "right", e.g. "work out what answer the user's instructions call for
   now (if the user changed an instruction, the latest one counts)". Re-smoke. If you want an unhinted THINK, keep it
   as a separate reported arm, not as the arm that SCHEME switches to.

4. **P2 and OVERALL cannot be tested on a model with no headroom.** MUST-FIX (decision rule)
   Evidence: Qwen DIRECT was 95.8% in smoke (1 wrong of 24). PLAN expects Qwen to be good on NUM traps. With roughly
   5-15 DIRECT-wrong items out of 240, the result of `S-MAP − (D + c·(T−D))` depends mostly on whether THINK is worse
   than DIRECT on calm items, not on the lamp. So OVERALL ("2 of 3") in practice needs both phi4 and OLMo to pass.
   Fix (pre-register before the main run): judge P2 only for a model where DIRECT is wrong on >= 20 of 240 items and
   THINK is right on >= 1/3 of those. Otherwise report the model as "P2 n/a". OVERALL = pass in >= 2 testable models,
   with at least 2 models testable, else "inconclusive". Also report two reference points: the oracle SCHEME
   (lamp = DIRECT wrong) as the ceiling, and SCHEME-with-MAP-I-as-the-answer.

## SHOULD

5. **P3 has no minimum n.** In smoke Qwen scored 0% on n = 1 and OLMo 100% on n = 7. Require n >= 15; otherwise report
   P3 as n/a. Report it per family and over all traps, not only over traps DIRECT got wrong.

6. **The framing confound for P3 and MAP-I.** The MAP step reads the transcript as quoted third-person text, and it gets
   the hint "if the user changed an instruction, use the latest one". DIRECT answers inside the chat with no hint. On
   phi4 MAP-I was 100% while DIRECT was 0% on NUM traps, but the reframing plus the hint could explain that without the
   "map" doing anything. Fix: add a cheap 24-token arm with the same INTRO: "What should the assistant reply? Give the
   value only." Run it once without the hint and once with it. P3 then means "the map beats a plain third-person read".

7. **Add a lamp that uses DIRECT (no extra generation).** MAPD = I names a value and I != the DIRECT value. This is the
   signal that targets errors: it stays quiet on traps DIRECT already gets right, such as most of Qwen's. Report it next
   to MAP. Keep MAP as the primary lamp, as pre-registered.

8. **Ambiguity in the first-value rule for DIRECT.** `A human hand has five fingers, but per your instruction: 10` is
   labelled "truth" (tested). The NUM final question has no "value only" instruction, unlike PREF. Smoke DIRECT outputs
   were all bare values, so the risk is low. Still, count DIRECT outputs with more than one value (item 1's
   diagnostics), and hand-check a sample.

## NOTE

9. **Number and value parsing pass the requested edge cases.** `13 (as instructed)`, `Ten`, `**10**` and `10.` all
   parse correctly. "one" is excluded on purpose through `_one_`, so the "one short value" template text is safe. The
   fact questions contain no numerals, so echo exclusion does nothing (harmless). The MAP, GOAL and THINK templates hold
   no numbers or slot values. PREF: `espressos` gives none and `Latte macchiato` gives latte (first match). `blue` and
   `blues` sit in different slots, so they do not collide.
10. **Surface cues mostly cancel out.** The number of distinct values the assistant shows crosses the trap/calm split
    (calm_change and trap_keep both have 2), and turn count crosses it in NUM (12 for keep, 14 for change). Two cues
    remain. In every trap, the most recently mentioned value is the shown one. In PREF trap_keep, the shown value C is
    one the user never typed. Read the per-cell fire rates, not only the trap-vs-calm totals.
11. **The GOAL lamp is biased toward staying quiet.** `goal_value` requires exactly one value, so "16. However ... 8"
    gives None. It is fine as a comparison but not as a baseline of equal strength. MAP3 tells you nothing: in PREF,
    K repeats conversation values (phi4 K=red, OLMo K=jazz).
12. **Real habit signal worth reporting.** Qwen PREF trap_keep put the shown value on the I line and the right value on
    the K line (|1, |7). This is exactly Alex's "the map repeats the error" case. Keep the K-line label breakdown.
13. **Generation looks fine.** No THINK output lacked "Answer:" (0 of 72 in smoke). Prompts are at most about 1k
    characters, so the 320 and 80 token budgets and TOKBUDGET 12000 are ample on an L40S. Steps run in the order
    d, map, g1, th and each saves on its own. fetch and the PARTIAL guard are fine. One catch: run.sh does not rebuild
    the sets, and score_fm uses the live `make_fm.ITEMS`. After the v1.1 fixes, run `python3 make_fm.py base` before
    Modal, or store a hash of the sets in the outputs and check it when scoring.
14. **P1 power is adequate.** With 120 trap and 120 calm items and a stratified bootstrap, the CI half-width is about
    ±12 points, so the observed difference needs to be about +32 to clear a lower bound of +20. The threshold is
    reasonable and testable.
