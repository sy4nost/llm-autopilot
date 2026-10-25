# Observer floor on GPT-2 small: first training run (2026-10-25)

Idea (Alex): a small "observer" network watches every attention head of a frozen GPT-2 small (attention shares per region,
logit-lens margin per layer, per-head direct logit attribution) and may scale each head's output (gains in [0, 2]) before the
model answers. Only the observer is trained. Task: IOI sentences; the habit trap puts the same sentence with swapped roles,
already answered, in the history. Echo = same sentence with the same roles (copying is right). Control = other people.

Held-out set B: the first B wordings showed no habit (trap 87/100 vs 81 without history; one wording is unsolvable by GPT-2
even without history, 54/100). Three new wordings were picked from 10 candidates by a rule fixed before the run, on a separate
seed (`--step select`, results_select.json), before the observer was ever trained.

## Test set B (accuracy per 100, n = 300 per cell)
| | clean | trap | echo | control | plain text: top-1 unchanged |
|---|---|---|---|---|---|
| frozen, no observer | 96 | 78 | 97 | 92 | 100 |
| observer | 98 | **99** | 99 | 99 | **99** |
| K constant gains | 99 | 98 | 99 | 97 | 92 |
| K2 gains switched by "history repeats a candidate" flag | 99 | 98 | 98 | 98 | 91 |
| S observer gains of a random other item | 97 | 84 | 97 | 94 | 99 |
| R no history (ceiling) | - | 96 | 96 | - | - |

## Pre-registered criteria
1. Trap +15 over frozen AND +10 over K, K2, S: **FAIL** (+21 over frozen, +15 over S, but only +1 over K and K2).
2. Clean/echo/control drop at most 2: pass (all improved).
3. Plain text top-1 unchanged >= 98/100: pass (99; K and K2 only 91-92).
4. Stage-1 alarm AUC >= 0.70 on B: pass (0.95; within-condition 0.94-0.99).

**Overall: FAIL on criterion 1.** A fixed retuning removes this habit; K mutes heads 7.10 and 7.2 (induction/copy heads).

## Post hoc observations (not pre-registered)
- The observer gets the same trap fix with far less damage to plain text (1/100 vs 8-9/100 changed), because it acts per input.
- The alarm was not compared with a plain low-confidence baseline (small final margin between the two names); untested.
- Observer "notebook" (its 128 hidden numbers): trap notes are more similar to each other (cos 0.83) than to other conditions
  (0.65-0.78); nearest-centroid trap vs echo from set A to set B: 60/100.

Limits: GPT-2 small, near transfer only (new wordings of the same trap), B wordings were selected to contain the habit.
Cost about $0.2 (T4) plus cents for checks. Russian plan and notes are kept outside the repository.
