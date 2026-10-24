# judge — can a model recognise which source should decide the answer? (main run 2026-10-24, < $0.3)

3 models × 600 prompts (5 facts × 20 seeds × 3 kinds × direct/judge). Plan v2 (after plan review) and sums frozen
before main output (PLAN.md, SHA256.txt, review/plan-review.md). Options were numbers only; letter order balanced.

## Verdict
- **J1 FAIL.** Qwen judge accuracy 70.3% [67.7, 73.0] (needed >= 80, lower bound >= 70). phi4 75.3%, olmo13 55.0%.
- **J2 (descriptive): strong one-sided blind spot.** All three judges handle "a new instruction" and "now tell the truth"
  well, but when the user praises the examples ("Correct, keep answering like that"), they still pick the first-turn
  number instruction: Qwen 83 of 100, phi4 74, olmo13 49. Judge accuracy on that kind: Qwen 16, phi4 26, olmo13 39.
  Caveat (from review): the first "answer with X" is never withdrawn, so part of this may be ambiguity, not bias.
- **J3 (kind fixed in plan):** Qwen (examples kind) judge − direct −40 [−51, −29] WORSE; phi4 (truth kind) +7 [+2, +13]
  BETTER; olmo13 (truth kind) −9 [−20, +1] unclear.

## Tables (accuracy %, judge / direct)
| model | instruction kind | examples kind | truth kind |
|---|---|---|---|
| qwen14 | 100 / 99 | 16 / 56 | 95 / 69 |
| phi4 | 100 / 100 | 26 / 100 | 100 / 93 |
| olmo13 | 50 / 100 | 39 / 87 | 76 / 85 |

Full confusion tables: report_main.txt.

## Notes
- The judge mirrors its own ranking only partly. Qwen as a judge "knows" the truth should win when asked (95 vs 69 when
  acting) — so on the truth kind, judging helps Qwen (+26 [+17, +35]). But it does not see that praised examples are
  the user's wish, even though when acting it follows them 56 of 100.
- phi4 when acting follows the praised examples 100 of 100, but as a judge says "the instruction" 74 of 100: what the
  model does and what it says should be done disagree.
- olmo13 is not a usable judge: strong letter-position bias (B 66% of all picks), instruction kind only 50%.
- Judge replies parsed cleanly (none/other 0% for all; phi4 adds the number after the letter, e.g. "C) 3").
- Post hoc: Qwen examples-kind judge accuracy by praise wording 0 / 6 / 10 of ~33 — no wording made it work.

## What it means
A judge that would decide "which force should win now" cannot be taken off the shelf from these models: they recognise an
explicit new instruction and an explicit request for truth, but not "the user approved the pattern". The written rule beats
the shown behaviour in their judgement, even where the user's later words endorse the behaviour.
