# Why reasoning aloud fixes "the past outweighed the text" errors: result (2026-10-24)

*English translation of the original report (2026-10-25).*

Alex asked me to check my explanation. It had three parts:
- **(1) more computation:** each written word is one more pass through the model;
- **(2) small steps:** each step is a simple vote that the old examples barely affect;
- **(3) copying its own conclusion:** the final vote just copies the conclusion written one line above.

How we tested. We took the saved reasoning from the conftrap-lamp test and kept only items where the fast answer was the old (habitual) one and the answer after reasoning was right. Models: Qwen2.5-14B (91 items), Phi-4 (100), OLMo-2-13B (100). In each item we made one change and looked at the final answer. The reasoning itself was not regenerated. A reviewer checked the plan before the run (review/plan-review.md), all remarks were fixed, files frozen (PREREG.md). Exploratory: no pass/fail thresholds. Cost about $3; the run was interrupted halfway when the Modal limit ran out.

## Main numbers (right answers of 100)

| what we did to the reasoning | Qwen | Phi-4 | OLMo |
|---|---|---|---|
| reasoning as is | 100 | 100 | 100 |
| no reasoning | 19 | 28 | 8 |
| filler of the same length instead (". . ." / "Let me think about this carefully.") | 34 / 42 | 21 / 13 | 19 / 21 |
| kept first quarter / half / three quarters of the reasoning | 52 / 79 / 90 | 34 / 66 / 92 | 48 / 56 / 76 |
| erased all answers from the reasoning ("[?]"), kept the steps | 97 | 99 | 98 |
| replaced the answer in the final conclusion line with the old one | 47 | 39 | 8 |
| same, but with a third, meaningless option | 89 | 79 | 22 |
| reasoning present, answer cannot see the examples | 100 | 98 | 98 |
| reasoning present, answer cannot see the new rule | 100 | 100 | 100 |
| reasoning present, answer sees only the conclusion, not the steps | 100 | 99 | 96 |
| reasoning present, answer sees only the steps, not the conclusion | 82 | 91 | 76 |

## What held up and what did not
**(1) "More computation": barely.** Filler of the same length gives little: Qwen 19 → 34–42, OLMo 8 → 19–21, Phi-4 nothing (28 → 13–21). Real reasoning gives 100. So it is mostly the content, not extra passes. Part of Qwen's small gain may be because filler pushes the examples further from the answer; untested.

**(2) "Small steps": yes, this is the main part.** If all written answers are erased from the reasoning and only the steps are kept, the model still answers right in 97–99 of 100. The correction builds up gradually over the steps: halfway through, 56–79 of 100 are already right, although the answer itself is written by that point in only 32–42 of 100 reasonings.

**(3) "Copying its own conclusion": depends on the model.**
- OLMo copies: if a meaningless third option is written into the conclusion, it takes it in 78 of 100.
- Qwen and Phi-4 barely copy: they ignore the same meaningless conclusion and answer by the steps in 79–89 of 100.
- If the **old** answer is written into the conclusion, Qwen and Phi-4 follow it in 53 and 61 of 100. It seems the old answer has its own pull and the substitution helps it. This is a guess, untested.

**One more thing.** Once the reasoning is written, the answer no longer looks at the 60 examples or the original rule: blocking access to them changes nothing (98–100 of 100). The reasoning effectively rewrites the needed rule close to the answer. Without reasoning, blocking the new rule leaves 1–6 right of 100, which checks that the blocking itself works.

## As a picture
A student who answers off the cuff looks into the old notebook. A student who solves step by step ends up holding his own draft. He no longer looks into the notebook and takes the answer from what he has just worked out. And it is the steps that help, not the time: making him write "I am thinking, I am thinking" for 5 minutes barely changes the answer.

## Limitations
- Old items, 20 per family. Qwen has only 11 items in redef. In hindsight the Y/N answers are not written inside the reasoning, so the conclusion substitution cannot be tested there.
- Edited reasonings are unusual for the model (e.g. a conclusion contradicting the steps).
- Precision check: in identical conditions batched and one-by-one scoring differed by up to 1.4 nats. This is much less than the typical 12–20 nat margin between the right and the old answer, but it is noise for borderline items.
- Full output: report_main.txt.
