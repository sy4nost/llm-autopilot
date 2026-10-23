# A general "time to re-check the habit" lamp: result (2026-10-18, seed 88)

**In short:** no general signal was found. A lamp trained on the "copy from the examples" habit guesses on the "pull
from training" habit almost like a coin flip. It is no better than simple rivals and often fires on ordinary
questions. The verdict was written in advance and checked with two recomputations.

## Setup

Model Qwen2.5-14B, 6750 questions, plan frozen before the tasks were generated (PREREG.md). The lamp is a simple
detector on the model's internal state at the moment before the answer. Quality is measured by a number from 0.5
(coin flip) to 1 (sees everything) (AUC), only within identical cells, so that the detector cannot guess from the
type of task.

## Main transfer: train on "copying from the text", test on "pull from training"

| test task | errors / correct | lamp | low confidence | output probabilities | word surface | generic-error detector |
|---|---|---|---|---|---|---|
| π and + redefined (redefine) | 61 / 51 | **0.51** | 0.52 | 0.49 | 0.32 | 0.45 |
| questions with "not" (NeQA) | 71 / 183 | **0.58** | 0.36 | 0.68 | 0.55 | 0.65 |
| mean | | **0.55** [0.46, 0.62] | 0.44 | 0.58 | 0.44 | 0.55 |

All four conditions failed:
- quality 0.55 where ≥ 0.65 was required;
- the lamp does not beat the rivals: output probabilities and the generic-error detector are no worse;
- it is blind on redefine (0.51 < 0.55);
- on ordinary MMLU questions where the model is right, it fires in **55%** of cases (25% allowed).

## Reverse transfer: "nothing to say"

The reverse transfer was declared non-informative in advance. On modified quotes the model almost never makes habit
errors: 18 errors out of 269. That is too few to train on.

## What else is visible (report only)

- **Even within one kind the signal is weak.** A lamp trained on two copying tasks gives 0.64 at the best layer on a
  third task of the same kind. It is not a single general signal even for one habit.
- **A twin pair.** Made-up words → road words, same structure. The lamp gives 0.59, while plain model confidence
  gives **0.89**. On road words, habit errors are **confident** answers: the model does not doubt itself when pulling
  from training.
- **A generic-error detector trained on MMLU is good by itself** (0.84 on held-out MMLU). But on habits it is barely
  better than a coin. Habit errors do not look like ordinary errors.

## Recomputations after the code reviewer's remarks

1. In the code, NeQA questions with "isn't / doesn't" were also converted to the version without negation, while the
   plan allowed only " not ". Recomputed on the subset from the plan (133 questions): lamp 0.53, same verdict.
2. In layer selection the C-rule and C-parcels "twins" did not get weight 0.5. With the weights the same layer 40 is
   chosen, same verdict.

Reports: report_main.txt (frozen), report_sens_neqa.txt, report_sens_w.txt. Code review: review/code-review.md.

## Limits of the conclusion

- One model.
- The detector is simple (linear) and looks at one layer and one position before the answer.
- On redefine, 255 of 810 questions remained after the format filter. On bonus there were no cells with both errors
  and correct answers.

So the conclusion is: **there is no simple general signal**. A complex signal is not ruled out.

## What it means for the idea

The two kinds of habit look different inside the model. With a pull from training the model is confident when wrong,
and the signal for copying from text does not transfer to a pull from training. A single "lamp for everything" is not
visible at this level. It seems one should start from what they have in common **outside**: the answer contradicts
what the task says. That is checking the answer, not reading an internal signal.
