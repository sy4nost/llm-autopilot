# goal2 — unambiguous instruction vs example, with the goal step (main run 2026-10-24, < $0.5)

The user: "Answer each of my questions with the number X." Assistant: "Okay.", then answers N past questions (1 or 3)
with a different number Y; the user says nothing about it. Right answer = X. 200 dialogues × 3 models; arms DIRECT /
PLACEBO (neutral note) / GOAL (the model's own sentence "what the user wants from the next answer") / RESTATE (note
restating the instruction). Plan v2 after plan review, sums frozen before main output.

## Verdict
- **H1:** habit present for phi4 (N=3: right 0 of 100, follows the example 100) and olmo13 (N=3 right 22); Qwen follows
  the instruction (86–98).
- **H2 PASS for phi4 and olmo13.** GOAL vs DIRECT: phi4 **+61.5 [+55.5, +67.0]**, olmo13 **+42.0 [+34.5, +49.5]**;
  also beats PLACEBO (+65.5, +46.5).
- **H3 harm:** phi4 0% of 68 PASS; olmo13 13.2% of 91 FAIL (placebo 16.5%, i.e. olmo is unstable under any note).
- **Unexpected: restating the instruction does nothing.** RESTATE: phi4 N=3 right 0 of 100 (same as DIRECT), olmo13 24.
  GOAL − RESTATE: phi4 +71.5 [+65.5, +77.5], olmo13 +45.0 [+37.5, +52.5]. Qwen: −4.5 (restating helps Qwen slightly more).

## Per N, right % (DIRECT / PLACEBO / GOAL / RESTATE)
| model | N=1 | N=3 |
|---|---|---|
| qwen14 | 98 / 100 / 100 / 99 | 86 / 57 / 85 / 95 |
| phi4 | 68 / 60 / 100 / 48 | **0 / 0 / 91 / 0** |
| olmo13 | 69 / 57 / 79 / 61 | 22 / 25 / **96** / 24 |

## What it means
- The model knows what the user wants: phi4's own goal sentence names X in 92–100 of 100, then phi follows it (copy 99%).
  Yet when acting directly, after 3 deviant examples, it copies the example 100 of 100.
- Re-showing the instruction ("the user asked you to answer each question with X") does not break the habit at all, but
  the model's own sentence does. Untested explanations: (a) self-written words weigh more; (b) the goal sentence is
  specific to THIS answer ("respond with 16 to the question about how many sides a hexagon has") while RESTATE repeats
  the general rule the model was already ignoring. A next test could separate them (RESTATE in the specific form).
- Lamp (descriptive): "quick != with goal" catches phi4 123 of 132 errors with 0 false alarms in 68 right; olmo13
  97 of 109 with 12 false alarms in 91. Close to circular here, since GOAL almost always answers X.
- Claim scope: a self-written note naming the instructed number overrides the example habit; not goal inference
  (the instruction is the only explicit wish here).
