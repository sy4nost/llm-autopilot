# Rule-switch test v2: results (2026-10-16)

Run on Modal: 1.5B and 3B on L4, 7B on L40S in full precision (float32). 512 tasks per model, seed 1,
N = 0, 2, 4, 8, 16, 32, 64. The cache self-check passed for all (difference at most 1e-4).
GPU time about an hour in total, cost about $1 of the free $30.
Full summaries: `<model>-seed1/summary.txt`, raw tables in the same place (trials.csv, layers.csv).

## Gate (N=0, at least 85% required on each frame)

| model | frame H | frame S | result |
|---|---|---|---|
| 0.5B | 77% | 74% | failed, full test not run |
| 1.5B | 98% | 95% | passed |
| 3B | 100% | 100% | passed |
| 7B | 100% | 100% | passed |

## Main points

1. **Habit causes no old-rule errors.** After 64 examples of one rule, 3B and 7B err on the new rule in 0–1% of
   cases, 1.5B about 5%, the same as with no history at all (5% at N=0).
2. **H1 (pull towards the old answer in the model's confidence).** Supported for 7B: after the habit, confidence
   in the new answer is lower than after neutral lines by 0.9–1.7 nats at every N, and the difference grows with N
   (slope +0.08 [+0.05, +0.12]). But confidence remains huge (about −9…−13 nats in favor of the correct answer),
   so it does not change the answer. Not supported for 1.5B and 3B: with short histories (N=2–4) the pull is there
   (+0.5…+1.6 nats), but with long ones the sign flips.
3. **The grey-line control is not neutral.** A long grey history confuses the model by itself: for 1.5B errors
   grow from 7% to 19% at N=64, for 3B and 7B confidence drops. The comparison "habit minus grey lines" at large N
   therefore mixes two effects; the negative slope for 1.5B and 3B most likely comes from here.
   This is an inference from the tables, not confirmed by a separate check.
4. **H2 (hidden pull in the middle layers) is not supported for any model.** Splitting by the side of the
   habitual answer gives opposite signs (LEFT +, RIGHT −): this is a logit-lens skew to one side, not habit.
5. **Where the answer forms.** In larger models late and gradually: 1.5B at indices 19–20 of 28, 3B at 27–31 of 36,
   7B at 23 of 28 (a weak jump, 0.2 of the final). The sharp jump seen in 0.5B in v1 did not repeat.

## Implications for v3 (proposal)

- A second control is needed that does not dilute the rules: e.g. a history with examples of both frames in equal
  numbers without training one, or the same length without history (empty lines of another kind).
- For 7B the effect is only in confidence. To see it in answers the task must be harder: fewer training examples
  or a rule the model knows less well.
- Half precision (bfloat16) for 7B is not suitable for the logit lens: the self-check gave a difference of 0.28
  with a tolerance of 0.05. Full precision on L40S costs a bit more but is reliable.
