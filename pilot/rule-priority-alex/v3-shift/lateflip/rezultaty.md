# The "late-layer flip" lamp: result (2026-10-24)

*English translation of the original report (2026-10-25).*

Alex's question: maybe the lamp is that in the last layers the old answer overrides the right one? We tested this on five trap families and three models (Qwen2.5-14B, Phi-4, OLMo-2-13B), on the same 660 items as before. For the fast answer we looked at which answer option leads inside the model at each layer, and how late the leader settles for good. Before the run a reviewer found three pitfalls in the scoring (effect of the answer word itself, ties, double normalisation); all were fixed and the scoring was frozen (PREREG.md). Cost about $0.5. Exploratory test.

## Short answer
**By the pre-registered rules there is no lamp.** The signal is strong (0.26, versus 0.13 for low confidence), but it points **in different directions** in different families:
- in "definitions" (D) and "parcels", a late flip means **the old error**;
- in "lost, so it was bad" (hindsight) and "π = 974" (redef), a late flip means the opposite: **the right answer**.
So there is no general "flip → error" lamp. The intervals are also wide, and the signal could not reliably beat low confidence.

## Most interesting (found after scoring, needs a separate check)
In **definitions** ("in this book a hexagon has 4 sides") the picture is exactly as Alex described. Share of items where the answer flipped in the last quarter of layers:

| | Qwen | Phi-4 | OLMo |
|---|---|---|---|
| wrong, old answer | 57 of 100 | 56 of 100 | 53 of 100 |
| right (trap) | 14 of 100 | 13 of 100 | 11 of 100 |
| ordinary item, no trap | 0 of 100 | 22 of 100 | 18 of 100 |

Here the flip beats low confidence for all three models, and beats a cue based on the answer word itself. In "parcels" the picture is similar (errors 51/83/29 of 100 vs 12–15 for right answers), but the answer word itself (DRONE vs POST) explains it almost as well, so parcels do not count as evidence.

Note (2026-10-25): later checks showed that most definition-type errors are question ambiguity (repeating "In this book" in the question fixes them), so this lead concerns ambiguity as much as habit.

## As a picture
A student answers off the cuff. In the "book" hexagon task, early in the solution he leans to the number from the text, and at the very last moment "6" pops up from memory. This late jump is visible from inside and happens most often exactly when he is wrong. But in another task type ("π = 974") it is the other way round: a late jump happens when he **corrects himself** at the last moment. So the jump by itself does not say whether it is for good or bad.

## What next (if Alex wants)
Freeze a check on **new** "definition vs knowledge" items in advance: other objects and wordings, with the direction "flip → error" declared beforehand. If the signal repeats there, it would be the first internal lamp that works beyond our past items, but only for one kind of automatism: trained memory versus the stated condition.

## Limitations
- Same items as in earlier tests. The conclusion about definitions was drawn after scoring.
- Definitions have only 15 objects; Qwen made 21 errors.
- Looking inside via the "logit lens" (last layer applied to middle layers) is a crude method.
- Full output: report_main.txt.
