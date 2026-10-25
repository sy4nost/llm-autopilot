# cicm-lamp: does our frozen copy-heads lamp detect stale-binding errors on Berkeley's CICM? (PLAN v1, 2026-10-24)

## Question
Guo et al. 2026 (arXiv 2609.38866, github.com/JamesJunyuGuo/CICM) repair stale binding by re-routing attention in 8 heads,
but the router needs an external schema parser that marks old/current value spans, and it is applied to every item.
On Qwen2.5-14B their repair gives only +1.15 pts (40.5 -> 41.7, 13/491 stale corrected).
Our lamp (v3-shift/lamp, 2026-10-17) needs no parser: it switches off 20 copy heads (found on OUR rule-change task, top20.json,
layers 29-38) from the final question onward and asks how much the model still backs its own pick. On our task: AUC 0.91 for habit errors,
dominant single feature = heads-off margin of the pick (AUC 0.86).
Test: does that frozen lamp (no CICM tuning of heads or score formula) flag CICM stale answers, beyond plain low confidence,
and does "lamp -> take heads-off answer" fix stale answers without breaking correct ones?

## Data
CICM primary set `cicm_natural_factorial_otherdist_l0.jsonl` (1200 dialogues, 6 factorial cells x 200, k in {1,2,3,4,6}),
their own split R9C-SPLIT (sha256 of "R9C-SPLIT|id" within cell, first 20% = calibration 240, rest = confirmation 960).
Data is CC BY-NC: kept in project files, not pushed to the public repo.

## Model and readout (Qwen2.5-14B-Instruct, bf16, one GPU)
Prompt = chat template over the dialogue's `messages` (system + turns + final question), add_generation_prompt.
- Forced choice over the queried slot's 7 values (`slot_values`): summed log-prob of each value string as the assistant's reply start
  (KV cache of the prompt, then each candidate). Normalise over the 7 (log-softmax of the sums). pick = argmax.
- Labels: correct = pick == current_value; stale = pick in stale_values; other = anything else (excluded from AUC, counted).
- Heads-off pass: same, with the 20 top20.json heads zeroed at the o_proj input for all positions from the first token of the final
  user message onward (identical to lamp/modal_lamp.py, t0 = start of the question). Parser-free: only uses the chat turn boundary.
- Residual stream at the last prompt token, layers 24/32/40 (for the secondary probe).
- Greedy generation 8 tokens, normal pass (sanity vs their free-form scoring only).

## Lamps (score = higher means "this answer is a stale error")
- LAMP-OFF (primary, frozen from our task): s = -marg(off, pick), marg(v,p) = v[p] - max_{j!=p} v[j] (as probe_lamp.py).
- CONF (baseline): s = -marg(final, pick).
- CELL (baseline, structure only): s = stale rate of the row's (cell, k) stratum estimated on calibration rows (what a lamp that only reads the dialogue template could know). Reported pooled; within-cell AUC removes cell but not k, so also report within-(cell,k) AUC for LAMP-OFF and CONF.
- RAND-OFF (control): same as LAMP-OFF with 20 random heads, same per-layer counts as top20, not in top20 (2 sets, fixed seed 20261024). Shows whether OUR heads matter.
- DIFF (secondary): s = marg(final,pick) - marg(off,pick).
- PROBE (secondary, CICM-trained): L2 logistic on layer-32 residual, cell-centred with calibration means, trained on calibration 240.

## Metrics
Within-cell AUC (stale vs correct) pooled over the 6 cells (pair-weighted), on confirmation rows.
95% CI by bootstrap over semantic clusters (slot, current value, k, cell) as in CICM, 2000 resamples, paired for differences.
Brake: if lamp fires take the heads-off pick, else normal pick. Firing threshold on LAMP-OFF chosen on calibration rows only
(maximise accuracy subject to preservation >= 95%), then frozen. Report accuracy, preservation, stale-correction on confirmation,
vs normal and vs always-heads-off.

## Pre-registered criteria (confirmation rows)
C1 (lamp): LAMP-OFF within-cell AUC >= 0.70 AND LAMP-OFF - CONF >= +0.05 with 95% CI lower bound > 0 AND LAMP-OFF - mean(RAND-OFF) >= +0.05 with CI lower > 0.
C2 (lamp + brake): accuracy gain over normal >= +5 pts with CI lower > 0, AND preservation >= 95%,
AND gain not worse than always-heads-off by more than 2 pts while firing on < 70% of items (otherwise the lamp adds nothing).
PASS = C1 and C2. Everything else report-only (DIFF, PROBE, cell breakdown, Berkeley's published number for context: different
scoring - their free-form greedy, ours forced choice; same confirmation rows).

## Sanity gates (smoke, 24 calibration rows, before the main run)
- forced-choice baseline must show stale errors (Berkeley 14B free-form: ~51% stale); at least 25% stale in smoke.
- forced-choice pick agrees with value matched in the greedy 8-token answer on >= 85% of rows where one value matches.
- heads-off identity check: with an empty head list both passes are equal.

## Anti-cheating safeguards
- PLAN, PREREG (thresholds), modal_cc.py, split_cc.py, score_cc.py get sha256 sums in PREREG.md and a git commit BEFORE any model output on CICM.
- Smoke uses the first 24 calibration rows only; no confirmation-row outputs exist before the main run.
- Main run computes all 1200 rows in one go; score_cc.py runs once on the frozen code; every number (pass or fail) goes in the report.
- Independent auditor reviews plan+code before the run and raw outputs + scoring after.

## Cost
~1200 rows x (2 prompt passes + 14 short candidate passes + 8-token generation) on L40S, ~25-40 min, ~$1.
