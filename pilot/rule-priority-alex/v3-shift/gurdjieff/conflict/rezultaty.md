# Checking the guess "the sensor feels the conflict between fiction and fact" (2026-10-19)

The "supported / not supported" rule was written before scoring: PREDICT.md (checksums in sums.txt). Numbers: conflict_out.json.

Conflict axis = the mean snapshot (layer 32) of questions where the fiction conflicts with a fact, minus the mean snapshot of questions where the fiction agrees with the fact. It is built without "changed its mind" labels. For each of our task kinds the axis is built from the other three kinds.

| our task kind (not seen when building) | conflict axis | sensor | sensor without conflict axis | "unsure" |
|---|---|---|---|---|
| definitions D | 0.91 | 0.90 | 0.72 | 0.81 |
| R, habitual answer correct | 0.96 | 0.97 | 0.76 | 0.69 |
| R, compute | 0.69 | 0.69 | 0.61 | 0.88 |
| capitals K | 0.67 | 0.70 | 0.70 | 0.68 |
| R traps | 0.61 | 0.80 | 0.81 | 0.52 |
| road words N | 0.57 | 0.56 | 0.47 | 0.25 |
| mean | 0.74 | 0.77 | 0.68 | 0.64 |

(AUC within each population.) Verdict by the pre-registered rule: GUESS SUPPORTED (mean 0.74 ≥ 0.65 and above "unsure" 0.64; with the axis removed, the sensor loses 0.09 ≥ 0.05).
But the sensor is not only the conflict: the directions overlap weakly (about 0.27 out of 1), and on R traps the sensor sees something else (0.80 without the conflict axis).

Public tasks (axis from all four of our kinds, for reference): MMLU 0.75, negation 0.79, unknown 0.69, memo 0.66, pattern 0.65, hindsight 0.54.
"Unsure" on the same: 0.86 / 0.86 / 0.74 / 0.70 / 0.47 / 0.28. So on public tasks the conflict axis is mostly weaker than plain low confidence.
The mean axis level on public traps is low, like our "fiction agrees with fact": public tasks have almost no such conflict.
