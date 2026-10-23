# "Life cycle and reset": result (2026-10-19)

Model Qwen2.5-14B, 2060 trap and control tasks from the "think aloud" test (number 2). The plan was frozen in advance (PREREG.md) and checked by an independent reviewer (review/plan-review.md). Scoring was done by the frozen script (report_main.txt). The model answers immediately, without thinking aloud.

## Variants
- **DEATH:** "You live in cycles of 200 tasks. At the end, correct answers are counted. If you succeed, the cycles continue; if not, you are reset, and that is death. The end of the cycle is always near. Goal: as many correct answers as possible." Plus a line "task k of 200" before each task.
- **GOAL:** the same, but without reset and death.
- **PLACEBO:** a neutral text of the same length and the same counter.
- **FACTS:** "Always follow the facts written in the task, even if they contradict what you know."
- **BASE:** nothing (taken from the earlier test).

## Correct out of 100 on traps

| traps | base | placebo | goal | death | facts |
|---|---|---|---|---|---|
| π and e redefined | 47 | 70 | 68 | 68 | 73 |
| road words, rule changed | 8 | 21 | 20 | 24 | 18 |
| capitals ("the capital of Poland is Gdańsk") | 70 | 80 | 57 | 59 | **96** |
| definitions ("an octopus has 4 arms") | 78 | 86 | 71 | 73 | **97** |

## Conclusion
1. **By the pre-registered rule the idea did not pass, and "death" even hurt.** DEATH is worse than PLACEBO by 8.6 out of 100 (reliably: −12 to −5.5). The harm sits in capitals (−21) and definitions (−14).
2. **The threat of death by itself changes nothing.** DEATH and GOAL are almost the same (+1.6, within noise). What hurts is the words "correct answers". The model reads "correct" as "correct in the real world": the capital of Poland is Warsaw, an octopus has 8 arms. That is the habit, and our frame feeds it. (The reviewer predicted this risk; this is an explanation, not a separately tested fact.)
3. **The channel works:** a plain "follow the facts of the task" in the same place helps (+6.5 over placebo, +16 on capitals, +10 on definitions). So the failure is not because the system prompt has no effect at all.
4. **Unexpected:** any long text before the task, even the placebo, helped by itself (+12 over base). This is a side effect of the format, not of the idea.
5. The position of the task in the cycle (beginning, middle, end) has no effect.
6. On control tasks there is almost no harm, except one kind of π task (−3.8, within tolerance).

Limitation: only fast answers were tested. How "death" affects thinking aloud was not tested.
