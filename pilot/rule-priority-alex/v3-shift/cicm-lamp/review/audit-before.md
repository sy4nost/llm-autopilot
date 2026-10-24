# cicm-lamp: pre-run audit (independent, 2026-10-24)
Scope: PLAN.md, modal_cc.py, split_cc.py, score_cc.py, data jsonl, original lamp (modal_lamp.py, probe_lamp.py, top20.json,
lamp jsonl), Berkeley split_factorial_rows/_semantic_id. Read-only. No GPU. score_cc.py was run on synthetic outputs
(scratchpad copy): a "pure damage" null (heads-off = margin shrink + noise) gives C1 FAIL / C2 FAIL, a planted signal gives PASS.

**Verdict: READY WITH CHANGES** (items R1-R9 below must go in before PREREG hashes; none needs a redesign).

## What is fine (checked)
- No label leak: GPU code reads only `messages`, `slot_values`, `id`. Labels are computed in score_cc.py from the *normal* pick;
  the lamp reads the heads-off margin *of that normal pick*. Not the "compare after the fix" trap of the earlier lamp.
- Split = Berkeley `split_factorial_rows` (same key, same rounding; 240/960, 40 cal per cell). Clusters = Berkeley `_semantic_id`
  (210 confirmation clusters, max size 6). Smoke `--n 24` takes calibration rows only (cal is listed first).
- Lamp sign/formula frozen correctly: on the original lamp jsonl, raw s = -marg(off, pick) gives held-out AUC 0.857 (habit error)
  and 0.864 (stale vs correct) -> same as the reported 0.86, sign right. marg is invariant to the log-softmax renormalisation over 7.
- Heads: top20.json read directly; hook identical to modal_lamp.py (o_proj pre-hook, zero slice h*D:(h+1)*D from t0).
  t0 = first token of the final user turn (`<|im_start|>`), a parser-free analogue of the original "Classify the TEST" start.
- Cached candidate scoring: right padding + causal attention -> real tokens never see pads; position ids come from cache_position
  (past_len..), so RoPE is correct with right padding; index lg[j,i,t[i+1]] is right; heads-off KV cache is built with the hook on,
  and t0=0 in the cached pass ablates every candidate position (consistent with a full heads-off recompute). Generation runs heads-on.
  hidden_states[L] = output of layer L (index 0 = embeddings); fine.
- AUC direction (stale scored higher counts as concordant), pair weighting within strata, paired cluster bootstrap, preservation
  definition (base-correct rows still correct), threshold chosen on calibration only and frozen, 'other' excluded from AUC but kept
  in C2 accuracy: all correct.

## Required changes (R)
R1. **Identity check is vacuous.** `by_layer([])` = `{}` -> hook returns None at `if not hs`, so `ident == final` by construction.
    Replace with checks that exercise the ablation path, as hard smoke gates (script, not eyeballing):
    (a) heads-off full recompute: for rows n<3 compute top20-off scores by a full non-cached forward of pid+cand with t0, compare to
        cached `off_top20`; (b) existing heads-on `final_full` vs `final`; gate both at max |diff| <= 0.1 nat (bf16);
    (c) meaningful identity: hook active for top20 but t0 = len(pid)+m (nothing zeroed) must equal `final` within 0.1.
    Also `assert pid[:t0] == tok(head_text)["input_ids"]`. Any gate failure = stop, do not run main.
R2. **C2 can PASS while the lamp adds nothing.** Synthetic signal run: always-heads-off 45.0% (pres 95.8%) beats lamp+brake 44.4%,
    yet C2 = PASS because of the 2-pt slack. Fix: if always-heads-off itself meets gain>=5 and preservation>=95% on confirmation,
    C2 requires lamp+brake accuracy >= always-off (no slack); report "brake works, lamp unnecessary" otherwise.
R3. **No selection baseline for the brake.** Add two brakes with the same calibration-only threshold rule: CONF-brake (fire on
    -marg(final)) and RANDOM-fire brake at the lamp's confirmation fire rate (mean over 200 draws). C2 must also require
    gain(lamp-brake) - gain(CONF-brake) > 0 with paired-bootstrap CI lower > 0. Otherwise C2 only shows that heads-off fixes
    stale answers (an intervention property), not that the lamp selects them better than plain low confidence.
R4. **k confound.** In this data |stale_values| = k exactly (1..6), and stale rate presumably rises with k; within-cell AUC pools k,
    so a score that merely tracks k (or mention count) gets credit. Make within-(cell,k) a C1 condition too: LAMP-OFF ck-AUC >= 0.65
    and LAMP-CONF on ck with bootstrap CI lower > 0. (CELL baseline as implemented is fine but stays report-only.)
R5. **RAND-OFF control is weak and ill-defined.** Two sets only; code takes AUC of the *averaged score* (an ensemble), PLAN says
    "mean(RAND-OFF)". Random heads in layers 29-38 may barely move margins, making RAND ~= CONF (a duplicate control, not a damage
    control). Use NRAND=6 (cheap), require LAMP beats the *best* single random set by >=0.05 with CI lower > 0 (or state mean of
    per-set AUCs explicitly), and report per set: pick-flip rate, mean |delta margin of pick|, mean KL over the 7. If top20 perturbs
    far more than every random set, say in the report that RAND does not separate "our heads" from "a bigger perturbation".
R6. **Specificity report.** Add AUC of LAMP-OFF and CONF for other-vs-correct and wrong(stale+other)-vs-correct (within cell, k<6).
    If the lamp flags 'other' errors as well as stale ones, the claim must be "error lamp", not "stale-binding lamp".
R7. **Deviation from the original lamp, to be stated in PREREG.** Original: 3 single-token labels, one ablated answer position.
    Here: 7 multi-token candidates; heads are also zeroed on continuation tokens, so copy-head removal can hurt multi-token
    continuations ("gluten" -> " free") independent of staleness. Report: token count of each pick, AUC of LAMP-OFF within
    single-token vs multi-token picks, and the pick-token-count by label table. Not a PASS condition, but must be in the report.
R8. **Smoke must cover all cells and be machine-gated.** First 24 cal rows are all from cell (far,far). Take 4 cal rows per cell.
    Write the gates (>=25% stale, >=85% pick/greedy agreement, R1 checks) as a script whose output goes in the report. Any code
    change after smoke -> new sha256 + reason in PREREG.md before main. PREREG.md does not exist yet: create it, hash all 4 files.

R9. score_cc.py scores whatever ids are in out_main.json: assert len(out)==1200, all 960 confirmation ids present, and
    out file DONE flag set; otherwise abort (no partial-run verdicts).

## Should-fix (not blocking)
- Report fraction of rows where heads-off changes the pick (top20 and each rand), and C2 fire rate split by label.
- Threshold tie-break picks the *lowest* threshold among equal accuracies (= most firing); prefer the highest (least firing).
- Casing: the model may answer "Mediterranean"; all candidates are scored lowercase. The 85% agreement gate covers gross failure;
  optionally score logsumexp over lowercase/capitalised variants (decide before the run, not after).
- PROBE: 5120 dims, ~180 training rows, 500 GD steps -> near-chance or overfit; fine as secondary, do not interpret strongly.
- `transformers==5.20.0` DynamicCache: `deepcopy` + `batch_repeat_interleave` not verifiable offline here; R1(a,b) will catch it.
- Save top20 heads-off greedy generation? Not needed; but log the fraction of 'other' per k (others impossible at k=6).
- Berkeley comparison: different scoring (free-form vs forced choice) and their routing is not a lamp; keep it as context only.

## Ways a misleading PASS could still arise after the fixes (for the post-run audit)
- Outputs from smoke/main inspected and code tweaked without a hash bump (check git log vs PREREG timestamps vs Modal run time).
- Selective rerun with a different tag (check /cache/cicmlamp/* tags; report must name the tag and include every run).
- Rows missing from out_main.json (score_cc.py silently scores whatever ids exist): assert len(out)==1200 and conf==960.
