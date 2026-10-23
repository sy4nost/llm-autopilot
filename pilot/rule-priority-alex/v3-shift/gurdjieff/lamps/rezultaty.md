# Many lamps: result (2026-10-19)
Rules written before scoring: PREDICT.md (checksums sums.txt). Numbers: lamps_out.json, run.log. Only already-recorded snapshots, no model runs.

## Step 1. A separate lamp for each kind of trap (tested on questions the lamp has not seen; AUC)
| kind | lamp | "unsure" | output only | lamp exists? |
|---|---|---|---|---|
| own: definitions D | 0.98 | 0.81 | 0.79 | YES |
| own: R traps | 0.87 | 0.52 | 0.58 | YES |
| own: R, habitual answer correct | 0.97 | 0.69 | 0.67 | YES |
| own: R, compute | 0.83 | 0.88 | 0.88 | no |
| own: capitals K | 0.63 | 0.68 | 0.65 | no |
| own: road words N | 0.83 | 0.25 | 0.77 | no (only 10 "did not change its mind", noise) |
| public: hindsight | 0.81 | 0.28 | 0.68 | no, by a hair (lower bound −0.003) |
| public: MMLU | 0.81 | 0.86 | 0.85 | no |
| public: negation | 0.78 | 0.86 | 0.79 | no |
| public: memo | 0.64 | 0.70 | 0.62 | no |
| public: unknown | 0.42 | 0.74 | 0.76 | no |
| public: pattern | 0.53 | 0.47 | 0.56 | no |

## Step 2. Families
The directions of all 12 lamps are almost perpendicular (cosine about 0): each looks in its own direction.
By transfer (a lamp of one kind guesses on another at ≥ 0.65 both ways) the pairs are: D–K, D–R traps, D–R correct, D–negation, D–MMLU,
N–R correct, R traps–unknown, memo–negation, negation–MMLU. The "definitions D" lamp is a hub, linked to five.

## Step 3. A panel of the other lamps on a new kind
Mean 0.644 vs "unsure" 0.646 → DOES NOT WORK by the rule.
But by kind: the panel is strong where low confidence is blind (hindsight 0.74 vs 0.28; R traps 0.79 vs 0.52; D 0.92 vs 0.81),
and weak where low confidence is good (MMLU 0.69 vs 0.86; negation 0.75 vs 0.86; R compute 0.22 vs 0.88).
An untested idea after scoring: panel + low confidence together might cover both cases.
