# The "surprising word" lamp: result (2026-10-24)

*English translation of the original report (2026-10-25).*

Alex's idea: the model reads the task and keeps predicting the next word. If a word turns out quite different from what it expected, that is a "stumble". Word surprise was measured two ways: relative to neighbouring words and relative to the whole task. The question: can a stumble, before the answer, signal that the model is about to answer from habit?

How we tested. We took the same 760 items from the earlier conftrap-lamp test, for which fast answers were already saved. Same models: Qwen2.5-14B, Phi-4 and OLMo-2-13B. For each item the model read the text once, and for each word we recorded how much it surprised the model. An independent reviewer checked the plan before the run (review/plan-review.md) and found 4 serious errors, all fixed. Scoring rules were frozen before we looked at results (PREREG.md, file checksums matched). Cost about $0.3. Exploratory test: the items had been used before.

## Short answer
**There is no lamp.** None of the main pre-registered conditions was met:

| | signal strength (0 = coin flip) | required |
|---|---|---|
| most surprising word in the task | 0.105 | ≥ 0.12 |
| plain low confidence in the answer (rival) | 0.132 | lamp must beat it by at least 0.05 |
| task length (rival) | 0.054 | lamp must beat it by at least 0.05 (got +0.05, but the interval goes below zero) |

The direction does not hold either: in 6 trap families of 10 a strong stumble went with the right answer, in 4 of 10 with the error.

## As a picture
A student reads the task. Asked on which word he "tripped" the most, he almost never points at the changed rule. He points at the word "and," in the middle of a long rule, at the heading "IMPORTANT", at a name like "Igor", or at "Q:". The most surprising word in the text is almost always noise: formatting, names, random numbers. It lands on the changed spot rarely: in 0–42 items of 100, depending on model and family.

## What we did find (report only, not the main question)
If we know in advance where the changed spot is and look at the stumble exactly there, the model **does** stumble:
- in definitions ("a hexagon has 4 sides") a trap can be told from an ordinary item by this spot in 86–98 cases of 100;
- in "lost, so the decision was bad" items, in 78–99 cases of 100.

But **how hard** the model stumbled says almost nothing about whether it will then give in to the habit. Within traps the strength of this signal is 0.129, about the same as plain low confidence, and it points in different directions for different models. So the model notices the change almost always, but noticing does not turn into the right answer. This matches what we saw before: the problem is not "did not notice" but "noticed, yet the voice of habit is still louder". Important: here we told it where to look; the model itself does not know that spot.

Flagging 30% of items and reasoning only there, the stumble at the right spot catches 39–47 errors of 100. Low confidence catches more for Qwen and Phi-4 on definitions (73 and 50), less for OLMo (25).

## What it means
- The model has the "notice" step: the changed word surprises it. The "stop" step is not tied to it.
- Searching for "the most surprising word" across the whole text is useless: it drowns in names, numbers and formatting. This is exactly what Alex feared ("how do we pick the right words").
- Stumbling and low confidence are almost unrelated (correlation −0.26 to +0.27). They are different signals, but neither finds habit errors well.

## Limitations
- Old items, exploratory test.
- Few comparison pairs within traps: Qwen has only 46 pairs in redef; for OLMo only 3 families of 5 are usable.
- The 100 "parcels" traps are essentially 2 questions.
- Surprise was computed per word; other ways (by meaning, over several words) were not tested.

Full scoring output: report_main.txt.

## Addendum (post hoc, at Alex's request): where the key word ranks among stumbles
All words of the item ranked by surprise, 1 = strongest (rank_sp.py, report_rank.txt).
- Definitions (18 words): the false number "4" in "a hexagon has 4 sides" ranks 7th on average (13th in an ordinary item with the true number). So it rises, but almost never reaches the top three (0–5 of 100). Above it are "Q:", "this", "A:" and the object name ('hexagon' is 3–4 times more surprising). Qwen example: 'hexagon' 16.6, "Q:" 14.1, "book," 11.1, while "4" is only 4.8, like the word "In".
- hindsight (~560 words): the outcome word ("lost" at an 85% chance of winning) ranks 44th overall for Qwen (175th in ordinary items). Relative to neighbours (Alex's idea) it rises to 18th and is in the top 10% of words in 96 of 100 traps. Weaker for Phi-4 and OLMo (48th and 54th by neighbours).
- redef (16 words): the definition is almost always in the top three (68–100 of 100), but the text is very short.
- parcels/bonus: the new rule is in the top 10%, but equally so in ordinary items (the same rule is there).
- Whether the model erred from habit or answered right, the key word's rank is almost the same. A small hint only for Phi-4 on redef: with a right answer the definition is in the top three in 91 of 100, with an error in 70 of 100.
