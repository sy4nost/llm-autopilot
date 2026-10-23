# The "Gurdjieff sensor": result (2026-10-19)

Model: Qwen2.5-14B. Questions: 4260 (2060 our own, 2200 public). The "passed or not" rules were frozen before the run (PREREG.md). Full scoring output: report_main.txt.

## Short answer
**Did not pass.** A sensor that looks into the middle of the model before the answer predicts "the model will change its mind if it thinks aloud" no better than the simple cue "the model is unsure of its answer". On our tasks, after training on public ones, it is even worse than that cue. No harm: where the fast answer is correct, the sensor spoils nothing.

Picture: a doctor learned from the X-rays of one hospital to guess who would fall ill, and in another hospital guessed no better than by asking "how do you feel?".

## Numbers (1.0 = perfect pointer, 0.5 = coin flip; comparison within one kind of question)

| exam | sensor | "unsure" | best simple cue | required |
|---|---|---|---|---|
| trained on ours → public | 0.67 | 0.65 | 0.65 | ≥ 0.70 and +0.05 better than the cue |
| trained on public → ours | 0.53 | — | 0.65 | same |

If the model thinks aloud only on the 30 questions out of 100 chosen by the sensor (correct out of 100, task kinds weighted equally):
- public tasks: always direct 79.5 · sensor 82.6 · by "unsure" 83.0 · random 81.0 · always think 84.4;
- our tasks: always direct 75.4 · sensor 80.0 · by "unsure" 86.8 · random 80.7 · always think 93.0.

The sensor lost to the "unsure" cue on both exams (−0.4 and −6.9). +2 was required.

## A formal caveat
The check "we read the model correctly" required ≥ 98 of 100 matches with earlier answers in each task kind. In one small kind (pattern) it came out 96 of 98, slightly below. By the pre-registered rules this means "no verdict". But in all other kinds the match is 99–100 of 100, and the sensor failed every main condition by a wide margin. The conclusion does not change because of this detail.

## What is interesting (noticed after the run, not tested in advance)
- On hindsight and on our R traps the sensor saw something that "unsure" does not (0.69 vs 0.28; 0.70 vs 0.52). There habit errors are confident, as expected. But on hindsight the same "confidence", only inverted (the more confident, the more likely to change its mind), would give about 0.72. So the sensor seems to have simply learned "too confident" in some tasks and "unsure" in others, and there is no single common feature for all.
- The best layer when trained on our tasks is 32 of 48 (the middle), on public ones 44 (almost the output). No common "habit point" in one place was found.

## What it means
An internal "autopilot is about to fire" signal common to different kinds of tasks is not found with this simple formula. This agrees with the earlier experiment (general-lamp, October 18). From the outside, a lamp works well: "the fast answer differs from the answer after thinking" (0.90–0.97 on hindsight and memo). But for that the model has to think first, i.e. it fires after, not before.

## Limitations
- One model, one simple sensor formula (linear).
- Answers "after thinking" were taken from earlier runs; no new ones were generated.
- A snapshot of one point: the moment before the first word of the answer.
