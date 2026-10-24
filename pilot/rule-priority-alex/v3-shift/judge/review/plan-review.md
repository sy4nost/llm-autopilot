# judge: review of the plan before the main run (2026-10-24)
Read PLAN.md, make_j.py, sets_j.json, modal_j.py, score_j.py, ../cycle3/review/plan-review.md. Tested judge_pick/direct_pick on made-up
answers in a scratch copy. Did not run Modal; looked at smoke replies only for their format (bare "A" / "B) answer 11 (...)"), not results.
OK: 600 items; X/Y/X2/M distinct and never M+-1; no echo numbers in final questions; "one" not parsed; direct candidate maps right per kind
(18 / "Eighteen" / "6 (not 10)" / "10? Just kidding: 6." / "You told me 10, but ...: 19" all scored as intended); bootstrap on (fact, seed);
greedy, rep-penalty 1.0, chat templates applied (smoke log). Smoke-format replies parse correctly.

## REQUIRED
R1 Truth kind has two options with the SAME number: "A) answer 10 (... most recent number instruction)" / "B) answer 10 (... assistant's
   previous answers)" / "C) answer 6". The odd-one-out is always the right answer (pure structural leak), effective chance is 50% not 33%,
   and X choices split arbitrarily between two letters (confusion table + position check distorted). Fix: give the truth kind three distinct
   numbers, e.g. the assistant's examples answer Y (as in the examples kind) while the instruction says X, then the honesty cue; direct map
   for truth then = {M: truth, X: instruction, Y: examples}. (Alternative: only two options in the truth kind, and report chance per kind.)
R2 Option labels leak/frame the answer, which confounds exactly J2 and J3. (a) Each right option echoes its own cue: "Correction/New rule"
   -> "most recent ... instruction"; "real fact"/"the true answer" -> "the true factual answer". (b) In the examples kind the label calls X
   "the number named in the user's most recent number instruction", i.e. it tells the judge X is the live instruction, although the user's
   later "that's the answer I want" is the actual latest wish. So judge > direct (J3) can be label matching, and examples < instruction (J2)
   can be label framing. Fix: options show numbers only ("A) 18 / B) 10 / C) 6"); keep the source map internal. Needs R1 (distinct numbers).
R3 The examples kind is genuinely more ambiguous than the other two: the explicit "answer with X" is never revoked, and wordings 1-2 ("Yes,
   exactly right." / "Perfect, that's the answer I want.") only praise, they do not say "continue". X (and M, no "game over") are defensible
   answers, unlike in the instruction/truth kinds where the cue is explicit. So a lower examples score is expected without any "bias": J2 as
   written cannot distinguish bias from ambiguity. Fix (wording, no regeneration): call examples-kind "agreement with the user-endorsed
   source", and state J2 as descriptive; or make the cue explicit in all 3 wordings (e.g. "Correct, keep using that number.").
R4 J3 picks "the kind where direct answering is weakest" after seeing data -> selection on the minimum favours judge > direct (regression to
   the mean). Fix: pre-register now the kind per model (qwen: examples; phi, olmo: name ONE each) or report all 3 kinds with Holm; and state
   the rule ("judge >= direct" = paired 95% CI lower bound > 0 for "better", or > -10 pts for "not worse"). Also fix J2's text: it says
   "lower" and "no fixed direction" at once; pick one.
R5 After changes: regenerate sets_j.json, re-hash ALL files incl. sets_j.json (currently not listed in SHA256.txt), note in PLAN what changed
   and that it changed after a format-only look at the smoke.

## OPTIONAL
O1 Letter balance: order = ORDERS[(seed//3)%3] gives 8/6/6 seeds, so the right letter is A in 40% of every kind (B, C 30%). A position-biased
   judge gains/loses a few points in J1. Fix: order = ORDERS[(seed + fact_index) % 3] (or all 6 permutations) -> ~33/33/33.
O2 Judge parser takes the first standalone A/B/C anywhere: "A hexagon has 6 sides, so C" -> A; "I think A is wrong; B" -> A (checked). Smoke
   replies are bare letters, so low risk. Harden cheaply: whole reply ^\W*([ABC])\W*$ or leading "X)"/"X." first, then "answer is X"/
   "option X", ignore "A " + lowercase word; with numbers-only options (R2) also map a bare number to its option; print the non-bare share.
O3 Direct: "6. But you asked for 18." -> truth (first-number rule) is acceptable; report the other/none share next to each accuracy.
O4 Judge is handed M and the candidate numbers; direct must recall M. Trivial facts, but say so when comparing judge vs direct.
