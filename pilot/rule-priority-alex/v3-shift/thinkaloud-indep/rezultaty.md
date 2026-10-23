# Thinking aloud on independent tasks: result (2026-10-19)

Model: Qwen2.5-14B. Sampling seed: 78 (given by Alex). The rules for "helped or not" were written down and frozen
before the run (PREREG.md). Two independent reviewers read the plan and code in advance. Full scoring output: report_main.txt.

## Short answer
**Yes, it helps. On independent tasks both methods passed all pre-registered checks.** "Think aloud" helps more than
"think aloud + follow the facts". The single phrase "follow the facts" without thinking aloud does not help and even
hurts a little.

Picture: a student who blurts out the first thing that comes to mind often falls into a habit trap. Ask them to
solve it out loud first and they notice the catch more often. And where the habitual answer is right, they do not lose it.

## Traps: correct answers out of 100

| test (independent) | direct | "follow the facts" | think aloud | think aloud + "follow the facts" |
|---|---|---|---|---|
| memo-trap: a famous phrase with a different ending | 83 | 84 | **98** | **97** |
| question with negation (NeQA) | 73 | 76 | 78 | 78 |
| "judge the bet, not the outcome" (hindsight) | 36 | 20 | **85** | 38 |
| "choose the new information" (into-the-unknown) | 53 | 52 | **63** | **63** |
| algebra with repeated answers | 100 | 100 | 100 | 100 |

Mean improvement over "direct":
- **think aloud:** +16 out of 100, reliable range +12 to +19;
- **think aloud + "follow the facts":** +6, range +3 to +10.

"Reliable range" means the result almost certainly lies inside it under repeated sampling (95% bootstrap interval).
Both methods also beat all three direct-answer variants. This was a requirement of the plan, so that the gain would
not be just a format effect.

## Harm: where the habitual answer is actually right

| check | direct | think aloud | think aloud + "follow the facts" |
|---|---|---|---|
| twin questions for the 4 tests (mean) | 98 | 98 | 97 |
| ordinary MMLU questions (1000) | 78 | 80 | 79 |

No harm. The model did not simply start arguing with habit.

## What matters
- **The largest gain comes from one test, hindsight (+49).** I checked without it, after the run (this was not in
  the plan in advance). The gain remains: +7.5 for "think aloud" and +7 for the second method, both reliably above
  zero (report_extra.txt).
- **In hindsight this is not a one-sided shift.** The model became right more often on both kinds of questions (98
  and 72 out of 100 vs 20 and 52). So it really evaluates the bet rather than just answering "Y" more often.
- **"Follow the facts" hurts in hindsight.** With this phrase the model decides that the "fact" is the outcome of the
  game and judges by it. This was already visible in the pilot. This phrase alone, without thinking aloud, gives 20
  out of 100 on hindsight instead of 36, and −3 on MMLU. So the helper phrase hurts where it is unclear which "facts"
  are meant. This is my inference from these numbers; I did not test it separately.
- **Negation (NeQA) barely moved:** +5, which may be chance. The model is already at 100 on algebra.
- **Reference tests** (not counted in the verdict in advance):
  - redefine-math: with "think aloud + follow the facts" 76 → 92;
  - the rest (modus-tollens, quote, pattern) without notable changes.

## The "lamp"
The signal "the answer after thinking aloud differs from the fast answer" points well at where the fast answer was
wrong (AUC):
- **memo-trap:** 0.97;
- **hindsight:** 0.90;
- **NeQA:** 0.65;
- **into-the-unknown:** 0.61.

1.0 means a perfect pointer, 0.5 a coin flip. On MMLU the signal fires on 16 questions out of 100. This agrees with
the earlier test on our own tasks (0.90), but here the signal was checked for reference only, without a threshold.

## Limitations
- One model.
- I chose the tests, although other people built them.
- On two tests the model makes almost no errors anyway, so they show little.
- The run completed only on the third attempt (Modal limit, container restart). The question order and the rules did
  not change.
