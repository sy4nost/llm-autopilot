# "Disguise" and "shake": result (2026-10-18, number 90)

**In short.**
- **Shaking does not work in any form.** Neither "BANG!", nor "Wake up", nor internal noise ("adrenaline") helped the model notice the rule change: they are no better than an empty placebo line.
- **Disguise works strongly, but with harm.** The model often drops the habit if the task is rewritten in neutral words. But where the habit was *right*, it starts making errors.

By the pre-registered rules all four methods **failed**. The plan is frozen in PREREG.md; the result was computed by the frozen script (report_main.txt).

## As a picture

A person drives to work along the usual road, although they were told the bridge is closed.
- **Shake**: a horn behind them. They flinch and... drive the same road.
- **Disguise**: give them a map of an unfamiliar city with the same street layout. There is no "own" road there, and they read the map honestly. But on the unfamiliar map they sometimes get confused even where the old road is actually right.

## Numbers (correct out of 100)

| tasks | normal | disguise | "BANG!" | "Wake up" | placebo | request "follow the rules" | noise |
|---|---|---|---|---|---|---|---|
| road words, traps | 11 | **39** | 8 | 8 | 7 | 8 | 17 |
| road words, control (habit is right) | 100 | **77** | 99 | 100 | 99 | 100 | 97 |
| road words, "rule confirmed" | 91 | 85 | 92 | 93 | 93 | 93 | 85 |
| "π now equals...", traps | 38 | **88** | 39 | 36 | 36 | 54 | 37 |
| "π..." without redefinition (ordinary math) | 96 | 96 | 92 | 97 | 97 | 97 | 94 |
| "π...", where the correct answer matches the habit | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

Why each method failed:
- **Disguise.** On traps the gain is +28 points [+19, +38] on road words and +50 [+45, +56] on "π". But on the road-word control the loss is −23 [−33, −14] with −5 allowed. This was **predicted in advance** (expected +23 and −18): the fictional twin is harder by itself, because common sense no longer gives hints there. On "π" there is no harm.
- **"BANG!" and "Wake up".** Gain around zero, and the simple request "follow the rules" beats them by 7–9 points.
- **Noise 0.2.** +6 on road words, −1 on "π". This is no better than randomly changing the same number of answers. Harm on "rule confirmed" −6.

## What else is visible (report only)

- **On "π" disguise is almost perfect.** Where the model normally always erred (e redefined: 2 of 100; "÷ as a random digit": 0 of 100), after disguise 93 and 98 of 100 are correct. A minus: in one template where the model was already right, disguise lowered correct answers from 98 to 82.
- **A simple request** "follow the rules even if it goes against what you know" helps on one "e" template: 2 → 75 of 100. On average over "π" it gives 38 → 54. Shakes give nothing of the kind.
- **Strong noise (0.4)** simply breaks the model everywhere, on traps and controls alike. This is not "waking up".
- **An external "ask twice" watchman.** If the "π" question is asked in ordinary and disguised form, disagreement between the answers catches habit errors very well: quality 0.92 out of 1 (AUC). Where the answers agreed, the ordinary answer is correct in 96 of 100; the answers disagree in 25 of 100. **On road words this watchman does not work** (0.57): there the disguised version itself often errs.
- "Copying from examples" (C-rule): shakes also change nothing (48 → 45–52).

## What it means

1. **The model is not "asleep"; it cannot be woken.** Surprise and noise do not switch on checking. A habit from training is not drowsiness but a confident track; a push does not knock it off. This agrees with the earlier result: on such a habit the model is confident and does not doubt.
2. **What works is removing the "hook" itself.** When the familiar word or symbol (π, e, road words) is absent, the pull disappears. But **the translator must be exact**: where the rewritten task is harder by itself, new harm appears.
3. **The most practical option is the "ask in two wordings" watchman.** Take the ordinary answer if it agrees with the disguised one; on disagreement, re-check. On "π" it catches almost all habit errors and rarely raises false alarms. This is **still a hypothesis for a separate test**: the conclusion was drawn after looking at the data and only on one kind of task.

## Limits

- One model (Qwen2.5-14B).
- The disguise for road words was "perfect" (it knew the generator's word map), while for "π" it was done by a simple algorithm.
- "π" traps with a habit occurred in only 4 of 9 templates. After the format filter, 675 of 810 questions remained.
