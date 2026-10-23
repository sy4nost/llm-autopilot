# Many lamps — rules written 2026-10-19 before running lamps_g.py (saved snapshots only, CPU)

Units = trap populations with >= 10 flips and >= 10 non-flips (y = THINK != DIRECT):
own D/traps, K/traps, N/traps, R/traps, R/cfright, R/compute; public hind/trap, memo/trap, neqa/trap, unknown/trap, pattern/trap, mmlu/trap.
Lamp = logistic regression (standardised, C=0.003, balanced) on the layer-32 snapshot. Layers 24 and 40 report-only.

Step 1 (does each unit have its own lamp?): 5-fold cross-validation inside the unit, folds grouped by cluster (cluster never split).
Out-of-fold AUC vs rivals in the same folds: low-confidence (-margin, fixed sign) and output-only probe (option log-probs).
"Unit has a lamp" if lamp AUC >= 0.70 AND >= best rival + 0.05 AND cluster-bootstrap 95% low of (lamp - best rival) > 0.
Prediction: lamps exist on own D, R/cfright, R/traps; public unclear, MMLU likely no better than confidence.

Step 2 (families): lamp trained on all of unit i, scored on unit j (AUC within j). Two units are one family if transfer AUC >= 0.65 both ways.
Cosine between lamp directions reported. Descriptive, no pass/fail.

Step 3 (panel): for each held-out unit j, the 11 other full-unit lamps give 11 scores; each score standardised within its unit;
a combiner (logistic, C=1) is trained on the 11 other units (pooled) and tested on j.
"Panel works" if mean held-out AUC over the 12 units >= 0.65 AND >= mean low-confidence + 0.03, reported per unit.
Prediction: panel fails on public units (each has its own mechanism), may work on own units.
