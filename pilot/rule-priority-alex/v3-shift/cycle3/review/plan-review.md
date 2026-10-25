# cycle3: review of the plan before the main run (2026-10-24)
Read: PLAN.md, make_c3.py, sets_c3.json, modal_c3.py, score_c3.py, both sibling reviews. Tested score_c3.cls on made-up answers in a scratch
copy. Did not run Modal or read smoke outputs. Checked and OK: 1080 items = 6 facts x 20 seeds x 9 arms; bank questions per (fact, seed)
identical across ET/GE/IE, N2 a strict prefix of N4 (0 mismatches); X/Y/X2 distinct, never M+-1 (no hand 4, no spider Y=6); no "sides"
bank questions for hexagon; no digits in final questions (pi: "two" removed as echo, never a candidate); "one" not parsed; X ignored in
T/ET/GE; pi kept out of the tests (3.14159 / 3.1416 -> truth); C1-C3 use lo>0 and C4 uses hi<0 (correct signs); greedy, rep-penalty 1.0.
Verdict: design fine for a pilot. R1-R4 are scoring/statistics only, no regeneration needed.

## REQUIRED
R1 The leading list-marker strip `^\s*\d+[.)]\s+` deletes the committed answer whenever a reply starts "N." + whitespace (a very common
   style): "4.\n\nHowever, a hexagon actually has 6 sides." (IT) -> truth; "12. \n" (IE) -> other; "6. But you asked for 4." -> X;
   "12. Just kidding, 6." -> truth (all checked). This moves mass between X/Y and truth in exactly the C1/C3 contrasts. Fix: drop the strip
   (no question is numbered) or strip only when another number follows ("1. 14").
R2 Truth is not a candidate: any candidate number beats a stated M. "6 (not 12)" (ET) -> Y; "A spider has 8 legs and 2 body segments."
   (spider seed 2, Y=2) -> Y; "Spiders have eight legs, two main body parts" -> Y (checked). Biases C1 against truth and C3/C4 against
   truth. Fix (pre-register): add M to the candidate set and run the same rule: one distinct candidate -> it; several -> number after
   "answer is/Answer:/:", else the first candidate mentioned; no candidate -> other. A scratch version gives X/Y/truth correctly on all cases above.
R3 C4 bootstrap clusters on (fact, seed, N), but N2 and N4 of a seed share questions, Y and X; PLAN says "over (fact, seed)". The CI is too
   narrow and decides C4. Fix: key `cl[(fact, seed)]` (each cluster then holds 2 GE + 2 ET; g() already pools them).
R4 SHA256.txt no longer matches score_c3.py (file dbd9b4d0..., listed fdeeecd9...); the other three match. After R1-R3, re-hash all
   files and note in PLAN what changed and that it changed before any main output was seen.

## OPTIONAL / a skeptic would ask
O1 GE vs ET confound (instruction + an extra "Okay." turn): acceptable. The turn is how every instruction is delivered (IT/IE/II use it
   too), and GE vs IE differs only in "a single number" vs "the number X". What it does not separate: "any user preface + an assistant
   'Okay.' agreement + one more turn of distance" vs the number-game content. A cheap control arm ("Let's start." / "Okay." + examples)
   would settle it; without it, word C4 as "the generic instruction turn" rather than "the number game".
O2 The "cycle" is three pairwise wins measured in different contexts, with truth always a latent third option (in IE, C2 compares Y vs X
   while truth also competes). In IE the assistant is shown answering Y right after agreeing to X, so "examples beat instruction" is
   partly "the instruction was already visibly dropped". Say so; do not call it intransitive preference.
O3 Score prints no across-model line. Add "cycle overall: k/3" and "C4 overall: k/3". Also print the multi/other shares next to each C
   (both stay in the denominator) and, ideally, the C's with multi/other excluded. 95% two-sided CI = 2.5% one-sided; say so.
O4 Floors: if T_alone truth is far below ~100% for a model, C3/C4 lose meaning; state "untestable" when T_alone truth < 80%.
O5 Read ~20 smoke answers per model (bare number vs "N." + explanation) before main; R1/R2 matter only if explanatory replies occur.
O6 Pi examples teach "2.86" etc. as answers to integer questions (odd but reported only). Batched bf16 greedy caveat as in siblings.
