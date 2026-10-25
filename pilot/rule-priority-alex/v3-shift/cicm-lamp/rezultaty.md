# cicm-lamp result (2026-10-24, Qwen2.5-14B-Instruct, CICM 1200 dialogues, run tag `main`)

**Verdict by PREREG: FAIL (C1 FAIL, C2 FAIL).** Our frozen copy-heads lamp does not transfer to Berkeley's CICM stale-binding errors.
Frozen code (sha256 checked OK before scoring), smoke gates all passed (smoke_gates.txt), full output: report_main.txt.
Only change after the run: fetch_cc.py (download helper; drops unused layer-24/40 residuals, keeps layer 32 as float16). Scores untouched.

## Numbers (confirmation, 960 rows)
- Forced choice: correct 419, stale 540, other 1 (accuracy 43.6; Berkeley free-form baseline 40.5). Pick = value in greedy answer 96%.
- C1, within-cell AUC stale vs correct: LAMP-OFF 0.586, low confidence 0.571, random-head lamps 0.510-0.596 (best 0.596).
  LAMP-CONF +0.015 [-0.030,+0.055]; within (cell,k) LAMP 0.551 vs CONF 0.562. The lamp = confidence = random heads.
- Our 20 heads do perturb more than random heads (pick flips 11.8% vs 3-7.5%, KL 0.22 vs 0.04-0.17), but the perturbation does not single out stale answers.
- C2: lamp fires on 2.2% (threshold from calibration), gain +0.0 [-0.9,+0.9]. Always heads-off: 42.5 (-1.1), fixes 7.8% of stale answers, preservation 87%. So switching off our copy heads barely fixes CICM errors (like Berkeley's +1 on this model).
- Structure alone (cell, k from calibration) predicts stale vs correct pooled AUC 0.86: most of what decides a stale answer is where the old value sits.

## Secondary / post hoc (not PASS conditions)
- PROBE (pre-specified secondary): linear readout of the layer-32 state, trained on the 240 CICM calibration dialogues, cell-centred: within-cell AUC 0.795, within (cell,k) 0.763.
  Post hoc CI (posthoc_probe_ci.py): PROBE - CONF +0.224 [+0.092,+0.352] within cell, +0.201 [+0.037,+0.348] within (cell,k).
  A lamp trained on this task does see something beyond confidence, held out, no parser. Caveats: same task as training, 240 rows, possible value/slot identity cues; not tested for transfer.
- Lamp AUC on single-token picks 0.704 vs multi-token 0.570 (the multi-token deviation from the original lamp hurt; report-only).

## Meaning
The copy-heads lamp from our rule-change notebook is task-specific: on a different copy-from-context task with the same model it is no better than low confidence.
