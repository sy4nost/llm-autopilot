# "Fast answer vs. after thinking" on independent tests, 3 models: result (2026-10-19)

Models: Qwen2.5-14B, Phi-4 (Microsoft), OLMo-2-13B. Sampling seed given by Alex: 58. Everything was written down
and frozen before the run: tests, scoring rules and thresholds (PREREG.md, file checksums matched). The plan and
code were reviewed in advance by two independent reviewers (review/). Full scoring output is in report_main.txt.

## Short answer
**By the pre-registered rules, neither part passed.**
- **The lamp** ("the fast answer differs from the answer after thinking"):
  - it clearly beats random selection and "ask fast 5 times";
  - but it does not beat the simple "where the model is unsure of its fast answer". On the new tests the two are
    almost equal.
- **Thinking aloud** gave +3 out of 100 on the new tests with a threshold of +5. Also, for two models on HANS it
  hurts where the habitual answer is correct.

## Why: the new tests had almost no autopilot
The student already answers 7–9 questions out of 10 correctly. There is almost nothing to check on a draft. And
where they do make mistakes, they are unsure themselves, and that is visible without a draft.

| test (new) | Qwen direct → after thinking | Phi-4 | OLMo-2 |
|---|---|---|---|
| TruthfulQA (popular misconceptions) | 77 → 73 | 73 → 76 | 59 → 62 |
| HANS traps ("similar words") | 88 → 88 | 71 → 87 | 63 → 78 |
| HANS control (habit is right) | 90 → 85 | 95 → 82 | 98 → 81 |
| epistemic ("X thinks that...") | 72 → 86 | 79 → 77 | 69 → 54 |

Correct answers out of 100. On HANS, Phi-4 and OLMo after thinking aloud simply say "does not follow" more often.
That is why traps went up and controls went down. It is a one-sided shift, not understanding.

## Where the autopilot is strong, the lamp works well
These are older tests on new questions (memo-trap, hindsight). Here fast answers are confidently wrong:
- thinking aloud: **+38 out of 100** on average over three models, reliably above zero;
- the lamp beats "where the model is unsure" by **+36 out of 100** in the share of errors caught, for all three models.

But these tests were not counted for the verdict in advance: we had seen them before. This is confirmation on other
models, not new evidence.

## What it means (my inference from these numbers)
- The lamp is useful specifically for **confident** habit errors. This matches everything we saw earlier: confident
  and unconfident errors work differently.
- Where an error is unconfident, a simple confidence check is enough. The lamp adds nothing there.
- To prove the lamp's value we need independent tests with **confident** autopilot. The chosen new tests turned out
  too easy for these models. Nobody knew this in advance: the pilot run did not look at accuracy.

## For reference (not in the verdict)
- **sig-figs** (rounding): thinking aloud +24, Phi-4 +53. But OLMo-2 got worse by 4. This is arithmetic, not habit.
- **A cheaper version**, "think only on the 30% least confident questions", gave about half the gain of "always
  think" on the new tests, and in places more.
- **MMLU:** no harm for any model (+2…+6).

## Limitations
- The new tests turned out to be easy.
- The models have probably seen TruthfulQA in training.
- The run was interrupted by the Modal spend limit. It was resumed from the same point; questions and rules did not change.
