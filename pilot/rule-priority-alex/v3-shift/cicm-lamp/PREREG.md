# cicm-lamp PREREG (2026-10-24, before any model output on CICM)

Plan: PLAN.md; pre-run audit: review/audit-before.md (verdict "ready with changes", R1-R9 applied below). Where this file and PLAN.md differ, this file wins.

## Frozen lamp
LAMP-OFF score = -marg(off_top20, pick), heads = ../lamp/top20.json (found 2026-10-16 on our rule-change task), zeroed at o_proj input from the first token of the final user turn onward. Deviation from the original (R7): 7 multi-token candidates instead of 3 single-token labels; heads are also zeroed on candidate continuation tokens. Reported: pick token counts by label and AUC for single- vs multi-token picks.

## Criteria (confirmation rows, 960; score_cc.py implements exactly these)
C1 (lamp) all of:
- within-cell AUC (stale vs correct) >= 0.70;
- LAMP - CONF within-cell >= +0.05, cluster-bootstrap 95% CI lower > 0;
- LAMP - best single random-head set (6 sets, same per-layer counts, seed 20261024) >= +0.05, CI lower > 0;
- within-(cell,k) AUC >= 0.65 and LAMP - CONF within-(cell,k) CI lower > 0.
C2 (lamp + brake = heads-off pick when the lamp fires; threshold from calibration rows only, highest threshold on ties, preservation >= 95% on calibration) all of:
- gain over normal >= +5 pts, CI lower > 0; preservation >= 95%; fires on < 70% of rows;
- LAMP-brake accuracy - CONF-brake accuracy (same threshold rule) CI lower > 0;
- if always-heads-off alone reaches +5 pts with >= 95% preservation: LAMP-brake accuracy >= always-off; otherwise >= always-off - 2 pts.
PASS = C1 and C2. Report-only: DIFF, CELL baseline, PROBE, random-fire brake, specificity (other-vs-correct, wrong-vs-correct), perturbation size per head set, Berkeley's published number (context only, different scoring).

## Smoke gates (check_smoke.py, 4 calibration rows per cell = 24 rows; all must pass)
run finished; 6 cells; cached = full recompute with heads on and with top20 off (max diff <= 0.1 nat); hook active with nothing zeroed = normal (<= 0.1); heads-off changes scores (> 0.1); stale >= 25%; forced pick = value in greedy answer >= 85%.
Any code change after the smoke: new sha256 below + reason, before the main run.

## Process
Main run: all 1200 rows in one run, tag `main`; score_cc.py main run once; every number reported, pass or fail. No other tags except `smoke`.

## sha256
75eda2260f33d87d9045d4b290d1d55ea10f443a3f5fa40c9b59f890e9f62a6f  PLAN.md
4ffcca8ff108677d01fda425ba1180adaa19512dca32648bb0fe4e1636427eff  modal_cc.py
bb12d262e0ca628a96241c141286c272dc808294240829aa558f25e89c5c4dc9  split_cc.py
ec3dd45df8318c1b53c9166bf0ee036751d0dd06be4445bb4ca7762257071261  score_cc.py
abb8787c60d78735489a9e58123e45332f40d4e73416876d24596d8ca3525f3a  check_smoke.py
90d6bfa2b9ddaf6112b0c092c71b48a6466faff1b457acfee071c3df94cc487d  ../lamp/top20.json
bceb34eedff413c6df2c89f01336c1d7fe74089ee634186b4fb350b629b7756d  data/cicm_natural_factorial_otherdist_l0.jsonl
