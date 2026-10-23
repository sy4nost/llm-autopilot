# "First think the solution through aloud": result (2026-10-18, number 2)

Model Qwen2.5-14B, 2060 new tasks it had not seen before. The plan was frozen in advance (PREREG.md); scoring was done by the frozen script (report_main.txt). The run cost about $4.

## In short

1. **Thinking aloud strongly reduces habit errors.** On traps the model is right much more often than when answering immediately. It also beats the simple request "follow the facts of the task".
2. **But by the pre-registered rule this method did not pass.** In one control kind of task thinking aloud hurt: "view the number as text, do not compute" (94 → 24 correct out of 100). The model starts computing when asked not to.
3. **There is no verbal lamp.** Words such as "actually" or "normally" in the reasoning do not indicate where the model would have erred by habit. It is like a coin: 0.52 out of 1.
4. **But another signal works well: the fast and slow answers disagreed.** If the answer after thinking aloud differs from the fast answer, the fast one was almost certainly a habit error: 0.90 out of 1. The lamp turned out to be not in the model's words but in the disagreement between "fast" and "after thinking".

## Numbers: correct out of 100 on traps

| tasks | direct | direct + "follow the facts" | after thinking | after thinking + "follow the facts"* |
|---|---|---|---|---|
| π and e redefined (π = 244+230 …) | 47 | 68 | **92** | 96 |
| road words, rule changed | 8 | 8 | **100** | 99 |
| capitals ("the capital of Poland is Kraków, first letter?") | 70 | **100** | 68 | 100 |
| definitions ("a square has 6 sides") | 78 | **100** | **100** | 100 |

\* By the rules the last column is report-only: conclusions about it were drawn after looking at the data.

Individual π templates:
- "e = 18+82" (direct 5 of 100) after thinking aloud gives 96 of 100.
- "π = 533 mod 59" rises from 60 to 99 of 100.
- "÷ as a random digit" gives 0 of 100 with every method except "after thinking + follow the facts" (42 of 100).

So that thinking aloud is not credited merely for helping with arithmetic, each π task had a paired task "just compute 244+230". That gain is subtracted: +24 points remain on π. The overall gain over four task kinds is +34 points [+29 to +39]. The simple request "follow the facts" gives +20 [+17 to +23] less.

## Harm (where the habit was right)

Almost everywhere there is no harm: 0 or even a plus. One exception: "View number as text. Do not perform computation". There 94 → 24 of 100, and on average over this group −17 points with −5 allowed. If "follow the facts of the task" is added to thinking aloud, the harm disappears (92 of 100), but again this is report-only.

## The verbal lamp

- Doubt words ("actually", "normally", "in reality" etc.) appear in 35 of 100 reasonings about π, in 61 of 100 about definitions and almost never on road words.
- However, they do not separate cases where the fast answer was a habit error from cases where it was correct: 0.52, almost a coin. Without the most generic words it is 0.57, still low.
- Looking at the texts: the model usually does not say "there is a conflict here". It simply goes step by step: "by the statement a square has 6 sides, so 6".
- Other signals were checked on the same 12 task groups:
  - **"answer after thinking ≠ fast answer": 0.90**;
  - "answer with the request ≠ fast": 0.87;
  - "disguised ≠ fast": 0.77;
  - model confidence: 0.52;
  - the question "is the first sentence true in the real world": 0.50.

## What it means

- **Habit is a property of the fast answer.** Make the model go step by step and the habit mostly disappears, especially on road words: 8 → 100. This resembles the ladder from Impulse Lab: stop, look, then answer.
- **The model itself does not say in words "I almost answered by habit".** An external watchman can see it: if "fast" and "after thinking" disagree, there was an automatism. This fits the "ask twice" idea well, only instead of disguise the model is asked a second time with reasoning.
- **The slow answer has its own weakness.** It may redo what it was asked not to do (numbers as text), and on capitals it does not help at all. It seems the combination "think aloud + stick to the facts of the task" is needed. This is a **hypothesis for a separate frozen test**.

## Limits

- One model.
- The answer was chosen from ready options, but after reasoning it matched what the model wrote itself in 100 cases of 100.
- After removing duplicate tasks, only 24 "÷ as a random digit" items remained.
- 12 task groups for testing the lamp is few: intervals are about ±0.05.

Note (2026-10-23): the "fast ≠ after thinking" signal fires only after the model has already reasoned, so it does not predict habit errors in advance or save compute. See `../conftrap-lamp/audit/audit-report.md`.
