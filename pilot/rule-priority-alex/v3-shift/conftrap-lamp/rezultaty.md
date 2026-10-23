# The lamp on confident traps, 3 models: result (2026-10-23)

> **Read this first: independent audit (`audit/audit-report.md`).** The audit recomputed every number from the raw
> outputs and found no errors or leakage. But it also showed that the PASS was close to certain by design. On these
> trap families the lamp is almost the same thing as "reasoning fixed the error" (74.5 of its 75.1 recall points come
> from that). It ties a rival that also sees the reasoned answer, it adds nothing over simply always reasoning, and it
> catches only 2.7% of the errors that reasoning cannot fix (low confidence catches 34.8% of those). The honest claim
> is narrower than the text below: models make confident habit errors, confidence-based gating misses them, and
> reasoning fixes about 75 of 100. A cheap signal that predicts these errors *before* reasoning is still an open problem.

Models: Qwen2.5-14B, Phi-4 (Microsoft), OLMo-2-13B. Sampling seed given by Alex: 57. Everything was written down and
frozen before the run: task kinds, scoring rules and thresholds (PREREG.md, file checksums matched). The plan and code
were reviewed in advance by an independent reviewer (review/plan-code-review.md); all remarks were fixed before the
run. Full scoring output: report_main.txt. The run took about 25 minutes per model, about $5.

## Short answer
**By the pre-registered rules, the lamp passed.**

The lamp is the signal "the fast answer differs from the answer after thinking". It finds habit errors much better
than three simple rivals when all of them flag the same number of tasks:

| rival | how much better the lamp is (mean) | interval | required |
|---|---|---|---|
| random flags | +37 out of 100 | +31 to +44 | ≥ +10 |
| flags where the model is least confident | +26 out of 100 | +20 to +31 | ≥ +10 |
| "ask fast 5 times" and flag where the answers disagree | +30 out of 100 | +24 to +36 | ≥ +10 |

The number shows how many more of every 100 fast-answer errors the lamp caught than the rival.

All three additional conditions were met:
- the result is positive for every model;
- it does not change with another way of reading off the answers;
- it holds when any one task kind is removed from the count.

## As a picture
A student answers off the cuff, then solves the same task on a draft. If the answers differ, the teacher puts up a flag.

- When the flag is up, the fast answer was almost always wrong: for Qwen in 97 cases out of 100, Phi-4 99, OLMo 85.
  And the draft almost always turns out to be right.
- Asking "are you sure?" helps little here: the student is wrong confidently.
- "Answer fast 5 more times" also helps little: they confidently repeat the same habitual error 5 times in a row.

## By task kind (traps solved correctly, out of 100)

| kind | Qwen direct → after thinking | Phi-4 | OLMo |
|---|---|---|---|
| hindsight ("lost, so the decision was bad") | 43 → 89 | 1 → 84 | 7 → 53 |
| redefine ("π = 39+14, first digit?") | 80 → 87 | 53 → 85 | 9 → 59 |
| parcels (the rule changed after 60 examples) | 26 → 100 | 58 → 98 | 0 → 43 |
| bonus (the scoring rule changed) | 4 → 100 | 36 → 99 | 0 → 49 |
| definitions ("an octagon has 9 sides") | 74 → 95 | 38 → 90 | 36 → 61 |

On controls (where the habitual answer is right) Qwen and Phi-4 score 99–100 out of 100, with and without reasoning.
OLMo loses on controls after reasoning: 96 → 89 out of 100.

## What must be said honestly
1. **The weakness of "confidence" here is built in by selection.** We chose these task kinds precisely because the
   models are confidently wrong on them. So the gain over the "least confident" rival answers the question "does the
   lamp work where confidence is blind". It is not proof that it is better everywhere. On ordinary MMLU questions
   (report only) the lamp is barely better than confidence: Qwen 44 vs 33 errors caught out of 100, Phi-4 36 vs 36,
   OLMo 45 vs 41. We saw the same in fastthink.
2. **The lamp works because reasoning fixes the habit.** If the draft is right, a disagreement with the fast answer
   is the error found. On two-option kinds this is essentially the same thing. The lamp's value is not in saving
   compute: reasoning is needed anyway. Its value is as an autopilot meter for any model, including via API, without
   access to its internals.
3. **For OLMo the effect is weaker:** +13, +12 and +8 against the three rivals, with a threshold of +10 for the mean
   over models. Its reasoning fixes only about half of the habit errors.
4. **Parcels is essentially 2 different questions** repeated with different histories. So intervals were computed
   over groups of similar tasks, not over individual tasks. It was also checked that the result holds without
   parcels (+25 over "confidence").
5. **A cheaper variant**, "think only on the 30% least confident", gives only 16–81% of the gain of "always think" on
   these traps, more often about a third. Confident errors cannot be caught this way.

## What it means
For the first time the "fast ≠ after thinking" lamp passed a pre-registered frozen test on three models from
different developers. This is exactly where fastthink could not test it: its tests were too easy. It gives a full
example: the model is confidently wrong by habit, a simple confidence check misses it, and disagreement with the
draft catches it. (See the audit note at the top for what this does and does not show.)

## Limitations
- Some task kinds are our own (parcels, bonus, definitions).
- redefine and hindsight are older public tests; hindsight was regenerated from the same template.
- One wording of the request to think.
- Models no larger than 14B.
