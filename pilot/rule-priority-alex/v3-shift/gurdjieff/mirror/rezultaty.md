# The mirror test: result (2026-10-19)
Model Qwen2.5-14B, our R and D tasks (1399 questions). Rules written and frozen before the run: PREREG.md. Full output: report_main.txt.
Sanity check: answers without the extra line matched the earlier ones in 99.8 and 100 out of 100.

## Short answer
**The mirror did not help; it hurt.** The line "your internal sensor shows: what is written conflicts with what you know", placed where the lamp fires,
lowered correct answers on traps from 63 to 55 out of 100 (−7.7, reliably below zero). The same thought without the words about a sensor
("what is written may conflict with what you know") raised them to 82 (+19).

## Traps (correct out of 100)
| | no line | mirror (where lamp fires) | same thought without sensor (where lamp fires) | generic line "be careful" | mirror everywhere |
|---|---|---|---|---|---|
| R traps | 47 | 44 | 64 | 43 | 44 |
| definitions D | 78 | 66 | 100 | 80 | 62 |

Where the habitual answer is correct (R correct, R agrees, D no conflict, R compute): no variant does harm (changes from −0.6 to +1.4).

Verdict by the rules: M1 fail (−7.7), M2 fail (the lamp is worse than a random placement, −3.4), M3 fail (the mirror is worse than the same thought without the sensor, −26.7),
M5 and harm passed. MIRROR NOT SHOWN.

## After the run (not planned in advance): the same thought without the sensor, on all questions
R traps 62, D 100: almost the same as only where the lamp fires (64 and 100), and without harm.
So the hint itself does the work, and the lamp, i.e. where to place it, adds almost nothing. File: posthoc_hint.txt.

## What it might mean (a guess, untested)
The words "your internal sensor... with what you know" seem to put the model's knowledge first ("you know"), and it holds on to the habitual answer more strongly.
The hint without the sensor keeps the task text first. This can be checked by changing the emphasis of the sensor line ("sensor: follow what is written").
