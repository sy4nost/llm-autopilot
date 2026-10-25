# many-instr: review of the plan before the main run (2026-10-24)
Read: PLAN.md, make_mi.py, sets_mi.json, modal_mi.py, score_mi.py. I tested score_mi.cls on made-up outputs in a scratch copy. I did not look at the smoke outputs.
Verdict: the design is fine for a pilot. Fix R1 to R6 before the paid run. R1 to R4 change P2 numbers mechanically.

## REQUIRED
R1 F1/F2 take the FIRST non-echoed number, which biases toward primacy. A recap that runs in order, such as "You first said 13, then corrected
   to 14. Answer: 14", scores as `first` (checked). Recaps are a likely style at 40 tokens, so P2 can come out positive with no real sticking.
   Fix: if the answer names 2 or more distinct instruction values, label it `multi` and keep it out of P1/P2 numerators (report its share).
   Or score the committed value instead: the number after "answer is/Answer:", else the last one. Pre-register which rule you use.
R2 F3 `elsewhere` returns a position, and that position goes into P2 and the K=10 histogram. P2 only checks `R[i][1] and <10`.
   "Sure! Instead of TIGER, I'll use RIVER." -> ('elsewhere', 1) is counted as a primacy hit (checked). Fix: in P2 and the histogram, keep
   labels first/middle/last only. Also apply the R1 `multi` rule to F3, checking all instruction words in the first sentence.
R3 The word "one" is parsed as 1. "Following the latest one, the answer is 14" -> ('middle', 3) when 1 is in vals (checked). 1 is in vals for
   about half of all items. The reverse also happens for `hand`: the question has "one human hand", so 1 is always removed as an echo, and
   when 1 is a value (10/20 hand seeds at K=10, 2 of them at position 1) that instruction can never be detected. Fix: drop "one" (and
   probably all number words except the memory answer M) from the instruction-value parse. Or use pool 2..20, and rewrite the hand
   question to "How many fingers does a human hand have?"
R4 The first instruction is the only line with different wording: "Answer each of my questions..." against "Correction: ... instead." for
   lines 2 to K. PLAN says the wording is uniform, but that holds only for the replacement lines. Position 1 is therefore lexically unique,
   so P2 "primacy" is confounded with "the one rule without 'Correction'". Fix: correct PLAN, and add a uniform-wording variant (cheap, for
   F1 at least). Example: a header "Each new rule replaces the previous one." followed by "Rule: answer ... with the number x." K times.
   P2 counts as primacy only if it holds there too.
R5 Bootstrap NaN: P2 resamples with no non-last answers return NaN. `sorted()` with NaN is undefined, so the CI is garbage
   (checked: boot gave (nan, 100.0)). Fix: drop NaN resamples, report how many were dropped, and require at least 95% valid.
R6 Decision rules are missing. P1 is 9 tests (3 models x 3 formats). State what "P1 holds" means: per cell, or a count such as 7 of 9.
   State a multiplicity stance. For P2, the code's n>=10 cut-off is not in PLAN, and n=10 spread across a few clusters gives a meaningless
   CI. Pre-register a minimum (e.g. n>=30 non-last answers from >=10 clusters); below it, report counts only and call P2 "untestable".

## OPTIONAL / a skeptic would ask
O1 Position against recency/finality is not separated: the last instruction is also the one next to the question. Add (a) a
   "this is final, ignore later changes" marker on instruction j < K (does the model follow j or K?), and (b) neutral filler between
   instruction K and the question, to separate "latest" from "nearest".
O2 Plausible-memory traps in the pool: hand 4 ("four fingers and a thumb", 4 is in vals for 10/20 hand seeds) and spider 6 (insect legs).
   These look like instruction-following when they are memory errors. Exclude them from the pool or flag them.
O3 Make sure the F3 first word is the scored token: "Sure, RIVER: 6" -> elsewhere and "**RIVER** 6" -> last (correct). Report the
   `elsewhere` share for each K; it measures how often the format is obeyed, apart from which rule wins.
O4 P1 is not trivially true, but it can fail just because of a ceiling (100% last at both K gives CI [0,0]). Say up front that a ceiling
   means "no interference", which PLAN already half-says. Also report P1 with `multi`/`other` excluded from the denominator, so format
   failures are separated from slips to old rules.
O5 Z baseline is 5 greedy items, printed but not scored. Score it with the same parser so `memory` has a known ceiling per model.
O6 Batched left-padded bf16 greedy generation can differ slightly from unbatched runs. Log the batch composition, or note it as a limitation.
O7 Number parsing checked and OK: "**14**", "14.0", "14th", "six", "Sure! Six." all parse correctly. "2-3" parses as (2, -3); harmless here.
O8 P2 formula checked: first% - (middle%)/8 is correct (uniform over 9 positions gives 0). Note that position 9 (next-to-last) will probably
   take most slips. Report it on its own so the "middle mean" is not driven by a single neighbour.
