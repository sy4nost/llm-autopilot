# Code review: make_g.py, modal_g.py, analyze_g.py (before the main run), 2026-10-19

Reviewer: independent (Claude). I read the code and checked it against PLAN v1/v2/v2.1, modal_ta.py `direct()`/`score()` and modal_ti.py `direct()`/`score()`. I audited items_g.json, out_main.jsonl and scores_smoke.jsonl with `python3 -I` and json only. I ran nothing on Modal.

**Verdict:** the extraction is correct and can run as it is. The analysis needs 4 fixes before the main analysis. All are cheap, and none needs a new GPU pass.

## What checks out (item 1: extraction)
- **Prefixes match each run's `direct()` exactly.**
  - own N: `chat(text)` with bare `cands`.
  - own R/K/D: `chat(text+R_TAIL)+"The answer is"` with `" "+c`.
  - public: `chat(text+R_TAIL)+(stem.strip() or "The answer is")` with `" "+c`.
  - Tokenisation is the same: prefix and option are tokenised separately, `add_special_tokens=False`, batch 1, no padding. The user text is `sets[...]["direct"]`, as in the runs.
- **Decision position k / pos is correct.** The logits at `len(pid)+k-1` predict `o[k]`. Every token up to `pos` is shared by all options, so reading it from option 0's pass is valid. Measured k: R and most D/alg have k=1 (the space); all others have k=0.
- **hidden_states[L]** (HF convention: [0] is the embeddings) is the residual after L blocks. Qwen2.5-14B has 48 layers, so the largest index used, 44, is before the final norm. I could not check this against transformers 5.20 locally. Add `assert len(out.hidden_states) == model.config.num_hidden_layers + 1`.
- **make_g matches the plan on most fields.** y, yplus, ok_d and ok_t match the PLAN. y uses own THINK by forced choice and public THINK by parse with forced fallback. redef is excluded. `direct` equals the argmax of `saved_direct` on all 4260 items.
- **No THINK or key information reaches the features.** surf is computed from the DIRECT user text, and dec and sums come from the DIRECT pass only.
- **The rest of analyze_g follows v2/v2.1:**
  - LOFO choice of layer and C.
  - Rival signs fixed on the training pool.
  - Gates at the 30% test quantile with a fixed-seed 1e-9 jitter.
  - Best rival re-chosen inside each resample, for P1 and P2.
  - P3 per population on point estimates.
  - N1 on own cells with >= 5 flips and >= 5 non-flips. 19 cells qualify; the reviewer expected 18, and the extra one is N/traps/D3.
- **The smoke curve is a small-sample artifact, not a bug.** I recomputed it from scores_smoke.jsonl:
  - The S1 test has 6 flips. Only one is in the sensor's top 40%: a pattern item at rank 3 where THINK breaks a right answer.
  - pattern has 2 items in the smoke, so that item costs 50 points / 13 populations = 3.8 points: 86.5 -> 82.7 at every share.
  - The hind and mmlu flips rank below 16th.
  - low-confidence has no flips in its top 16, so it stays exactly at DIRECT (86.5).

## BLOCKERS (fix before the main analysis)
**B1. The cluster bootstrap is missing, but PLAN asks for it.** PLAN v1 says "item + cluster schemes, wider interval", and v2 §7 repeats it.
- analyze_g.py:114-119 (P1) and :140-143 (P2) only resample items within each population.
- In make_g.py:31, own `cluster = grp` is a different cluster per item for K and D. score_ta clusters K and D by hook (`cell`); N by `grp` is correct, and R by item is correct.
- Fix:
  1. In make_g, set `cluster = cell` for K and D.
  2. In analyze, add a second loop of B resamples that draws clusters with replacement within each population, for both P1 and P2. Re-choose the best rival inside each resample.
  3. Use `lo = min(lo_item, lo_cluster)` and `hi = max(...)`, as score_ti `boot()` does.
  4. Give each scheme its own seeded rng.

**B2. Accuracy weighting differs from the PLAN.**
- analyze_g.py:130-133 weights test **populations** equally.
- PLAN v1 Metrics says "families weighted equally", and v2 does not override it.
- The +2-point P2 bar was set from reviewer simulations that weighted families.
- Population weighting brings in zero-flip populations, where every gate equals DIRECT: own has 5 of 13, public has 4 of 13. These dilute every gate difference toward 0, so P2 becomes harder.
- Fix: pick one rule and write it into PREREG. Either use family-equal weights (families equal, populations inside a family weighted by items), or amend the PLAN to population-equal and re-justify the bar. Then make P2, the curve and the printed accuracies use that rule.

**B3. Exact bf16 ties can make the sanity check fail on a tie.**
- 25 items have exactly tied saved DIRECT sums: pattern 2/100, alg 1/200, neqa 2/200, R 11, N 3, D 1, mmlu 5.
- analyze_g.py:79 compares argmax values. If a recomputation breaks a tie the other way, pattern can drop below 98% with one more mismatch, and the whole verdict becomes "stop".
- Fix:
  - Pre-register that items with saved margin < 1e-3 are dropped from the agreement count.
  - Print the per-family median |recomputed sum - saved_direct| as well. This needs no new data: `feats.sums` vs `saved_direct`.

**B4. Missing items are dropped silently.**
- modal_g.py:137 keeps only items present in feats.jsonl, and the report never says how many are missing.
- The resume path makes losses possible. modal_g.py:88-94 appends to a file that can end in a cut line, because the buffered writer flushes at arbitrary byte points and the commit happens every 50 items. The next JSON line then fuses with the cut line, and both are lost.
- Fix:
  - On resume, rewrite feats.jsonl keeping only valid lines, or write `"\n"` first if the file does not end with one.
  - In analyze_remote, print `n missing` and treat any missing item as SANITY FAIL.

## MINOR
- **`choose` can stick on NaN** (analyze_g.py:61). If the first candidate's AUC is NaN, `a > nan` is never true, so the first layer and C win by default. This is what produced "chose L8 C=0.003 ... nan" in the smoke. The main run has enough groups, but guard it: `if not np.isnan(a) and (best is None or not best[0] >= a)`, and raise if every candidate is NaN.
- **`dec = 0.0` for an option that lies entirely inside the shared prefix** (modal_g.py:116). Example: alg `" 8"` vs `" 84"`. 0.0 means probability 1, which distorts the output-only features. About 11 pattern/alg items are affected. Use the option's own summed log-prob, or `log(1 - sum p(continuations))`.
- **float16 storage** (modal_g.py:114) could overflow on massive-activation dimensions. The smoke shows no inf. Store float32 (about 700 MB in total), or assert `np.isfinite`.
- **HARM includes N/confirm** (make_g.py:9), which is not in the PLAN list. It only makes P3 stricter, but say so in PREREG. Also, the redef comment says "v2.4"; it should say v2 §4.
- **Report-only lines from the PLAN and reviews are missing.** Add them:
  - LOFO results within each pool (v1).
  - The share actually used with a train-transferred threshold (v2 check).
  - always-THINK and the oracle in the curve.
  - The curve for every rival, not only low-confidence and random.
- **pub cluster `[:200]` truncation** merges 2140 clusters into 2133. Use the full string or a hash.
- **Run time and timeouts.** Each analysis does about 312 LR fits of about 2000x5120 (8 layers x 3 C x 4+9 folds). I timed synthetic fits at 0.4-2.4 s, and real data may take 10x longer, which is still inside the 2 h limit.
  - The analysis is not resumable, and the report only comes back to the local client, so a dropped `modal run` loses it.
  - Write report and scores to the volume as well, and print progress per layer.
  - Memory is fine: H is 698 MB in float32, against a 32 GB limit.
- **Extraction cost:** about 10k batch-1 forward passes plus model load, well under the 3 h timeout and under $2 on an H100. Resume works, apart from B4.
- **Pin numpy and scikit-learn** in cpu_image so the frozen analysis can be reproduced.
- **RNG:** a single `default_rng(0)` is shared by the jitter and both splits' bootstraps. Results are deterministic but depend on order, which is acceptable; per-scheme seeds would be cleaner.
