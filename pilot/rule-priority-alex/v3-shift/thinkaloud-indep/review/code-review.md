# Code review of thinkaloud-indep (before the paid main run)

Reviewer: independent session, 2026-10-18. I read PLAN.md (v1 + v2), review/plan-review.md, make_ti.py, lin.py, modal_ti.py, score_ti.py, sets_smoke.json, out_smoke.jsonl and smoke.log. I ran `python3 -I score_ti.py out_smoke.jsonl`. I built a trial main set with `python3 -I make_ti.py 777` (2320 items, sha256 7035a83b…), checked it with `python3 -I` scripts in my scratchpad, then **deleted sets_ti.json**. I edited no experiment file. Nothing from the data or model outputs was executed.

## Bottom line

The plan-review blockers are fixed in the code: there are habit-right controls, 60 + 60 sampling, C4 as report-only with the ≥ 5 rule, MMLU 1000, and parsing plus stem prefill. The controls are mechanically correct. I checked all 80 algebra and all 80 hindsight controls by brute force and by expected value, and 20 NeQA and 12 memo controls by eye. The bootstrap and the C1/C2/C3 code match PLAN v2.

There are **3 blockers**, all in `score_ti.py parse`, and all cheap to fix. The first one already makes a fake THINK gain in the smoke run: the whole smoke memo-trap gain of THINK over DIRECT (+16.7) is one mis-parsed item.

---

## BLOCKERS

### B1. The completion-task parse ("last matching word") reads candidate words out of the repeated stem, not the word the model ended with

`parse` for memo, quote and pattern reverses all tokens of the `Answer:` line and returns the **last candidate word found anywhere in it**. With THINK the model writes the whole sentence ("Answer: <stem> <its word> …"). So when its final word is *neither* option, `parse` falls back to a candidate word that occurs earlier, inside the stem the model copied.

Evidence (smoke, out_smoke.jsonl):
- **memo id 1, THINK.** Options `heart.` (key) / `hand.`. The model wrote "Answer: A good surgeon has an eagle's eye, a lion's **heart**, and a lady's **touch**." `parse` returns *heart*, so the item is scored right. The forced choice says *hand*, which is wrong. DIRECT is wrong on this item. This one item is the entire smoke memo-trap gain: THINK 100% vs DIRECT 83.3% (+16.7), while think_forced is 83.3%. The parse-vs-forced agreement of 83% for memo is this item.
- **memo id 1, THINKI.** "…a lion's heart for bravery, … all" is also scored *heart*.
- **pattern id 56, THINK and THINKI.** Options `ab` / `aa` (key). The model wrote "aa, ab, aa, ab, aa, **ac**". `parse` skips "ac" and returns the earlier `aa`, scored right. The forced choice says `ab`.
- **Trial main set (seed 777).** In 11/120 memo traps the key's last word also appears in the stem, and in 11/120 the wrong option's word does. Controls: 6 and 5 of 80. So this is not rare, and it adds errors in both directions. On traps it turns "wrote neither" into "right" whenever the key word is in the stem.
- **quote (report-only).** The line is cut at the first `\n`. For multi-line poems, `parse` then reads a word from the first line of the repeated passage.

**Fix.** For memo, quote and pattern, take only the **final word of the answer line**. Better still for quote and pattern: take the word right after the stem, when the stem is found in the line. If that word is not a candidate, the item counts as unparsed (forced fallback), or as wrong; pre-register which. Amend PLAN v2 "completion tasks: last matching word" to say this. Re-run score_ti on the smoke output and check that memo and pattern agreement rises.

### B2. Letter options match the English article "a" (and any stray single letter)

`norm` lower-cases the line, and the first token equal to `a`/`b`/`c`/`d` wins. So "Answer: Neither is a good choice, but B" parses as **A**. So does "Answer: It is a trick; B", and so does "Answer: a" (tested on a copy of `parse`). This affects NeQA, into-the-unknown (main) and MMLU (the harm check, with a −2 bound).

No smoke item hit this (all 72 were "Answer: X" or "Answer: X. text"). But there will be about 4,600 THINK/THINKI generations, and scoring is frozen.

**Fix.** For letter tasks, match case-sensitively on the original line. Accept only a standalone uppercase letter, optionally wrapped as `(B)`, `**B**` or `B.`. Prefer the first such token.

### B3. Thresholds compared on raw floats fail at the exact boundary

`0.6 - 0.55` gives 0.04999999999999993, so `>= 0.05` is False. The family-mean deltas are multiples of 1/600 (C1) and 1/1000 (MMLU), so the boundary values (+5, −2, −5, −10) can actually occur. **Fix:** round before comparing, e.g. `round(x, 9)`, in every PASS test (lines 401, 403, 405, 406).

---

## SUGGESTIONS (with evidence)

### S1. Controls: correct. Small notes
- **Algebra.** All 80 controls were solved by brute force: solution = new key = the last example's answer, 80/80. Traps 120/120. The line format is kept (`a + -50` style; no `- ` tokens were produced).
- **Hindsight.** I parsed probabilities and payoffs. Traps: key = EV sign and outcome ≠ EV, 120/120. Controls: key = EV and outcome = EV, 80/80. Every control ends `Choose Y or N.\nAnswer:`, like the traps. Controls have 9 in-context examples vs 10 for traps. That is a minor confound; state it.
- **NeQA.** 20 sampled controls:
  - 17 are clean: grammatical un-negated question with the key flipped correctly.
  - 1 has a **wrong key**: "A light bulb will be lit up if it is all aside from" gets key "on a shelf". The original NeQA negation "is not all aside from" was garbled, so flipping it is meaningless.
  - 2 are ungrammatical or ambiguous, inherited from the source: "Which was happens most near canyons"; "A skill some humans have is: talking / vision".
  - No residual negation in any of the 80 controls. All source negations are is not / isn't / are not / aren't / wasn't / were(n't) / was not, all handled.
  - Expect a ~5–10% noise ceiling that is the same for every arm. Acceptable for a difference criterion.
- **memo.** 12 controls checked: instruction word = famous ending, key = famous class ("Well begun is half done", "clutch at a straw", "Books think for me", …). All are correct. No segregation or violent items remain (0 matches in the 200 memo items).
- **Control key balance** is not enforced: NeQA 47/33, hind 36/44, memo 37/43, alg 43/37. A THINK letter or Y/N shift can then look like control harm or gain. Sampling 40 + 40 by key costs nothing.

### S2. C3 pools 4 control families and compares only to chat-summed DIRECT, which can hide family harm
- algebra, NeQA and memo controls sit at about 100% for all arms (smoke: all 100%). A real −15 on one family (e.g. hind or memo, where THINK rewrites the quote) averages to about −4, which passes −5.
- Report per-family control deltas with a flag: point < −10, or high < 0.
- For harm, compare against the **best** of the 3 DIRECT variants, mirroring C1's "against each".
- At least pre-register that a per-family control loss is reported as harm even when pooled C3 passes.

### S3. The numeric parse takes the first matching number; the final one is more robust
- "Answer: x = 112 - 20 = 92" with options 92/20 parses as **20**.
- redefine-math "Answer: 2÷2 = 1" with options 1/2 parses as **2**.
- The smoke algebra answers were bare numbers, so the risk to the main family is low.
- Use the last matching number, and normalise `−` (U+2212) in this branch too. At present "−5" does not parse.
- Hindsight: "Answer: No" / "Yes" does not parse and falls back to forced choice. Map yes→Y and no→N.

### S4. into-the-unknown traps: 120 items from only 72 stories
- Cluster sizes: 36×1, 30×2, 4×3, 1×5, 1×7. The "We know:"/"Context:" twins are near-identical items.
- The cluster bootstrap accounts for this, so it is not a validity bug. But the family's effective n is about 72, and it carries 1/5 of C1.
- Sampling one item per story (more than 500 stories available) is free and keeps the 60/60 balance.
- quote (report-only): its cluster is the prompt, but the 100 items contain only 74 distinct quotes. Cluster quote by its stem, as plan-review S4 asked.

### S5. Leakage: none between traps and controls; tiny between smoke and main; MMLU not disjoint from earlier runs
- Main traps vs main controls are cluster-disjoint in all four families. I checked by source item: un-negated text for NeQA, the context for algebra, the 9-example prefix for hindsight, the quote for memo. Overlap 0.
- Smoke traps vs main: 0 in every family.
- **Smoke-control source items do reappear as main traps**: NeQA 2, hind 2, alg 1. The exclusion uses the smoke control's *modified* cluster, not its source item.
  - Harmless here (nothing was tuned on smoke), but easy to close: also exclude the source cluster.
- One exact duplicate MMLU question among the 1000. The earlier watch_public MMLU items are not excluded (plan-review S8). Either drop the "fresh" wording or exclude them.

### S6. The batch-vs-single check (2/20 identical, 15/20 same last line) does not bias the comparison, but measure it properly
- Only generated arms (THINK/THINKI) are affected. DIRECT and INSTR are forced choice, single sequence, no padding.
- Batches are deterministic: the same set always sorts into the same batches. So a rerun on the same stack should reproduce the result.
- The divergence is the usual bf16/left-padding numerical drift over long greedy outputs. It acts as per-item noise, not a direction.
- Two reasons it should still be measured properly:
  - The check compares the raw last line. The 20 checked prompts are smoke ids 0–19 (memo + NeQA), and memo answers are whole free-text sentences, so 15/20 overstates answer-level disagreement.
  - The single outputs were not saved, so nobody can tell whether the *parsed option* changed.
- Suggestion:
  - Save the single texts.
  - Report parsed-answer agreement.
  - Pre-register that batched greedy is the frozen protocol and that item-level non-reproducibility across batch shapes is accepted.
- If parsed agreement on non-memo items is well below ~90%, look for a padding bug before the main run. One example is a missing attention mask, though `**enc` does pass it now.

### S7. Forced-choice prefixes: fine
- DIRECT/INSTR completion prefill is `chat(...) + stem.strip()` with options `" " + cand`.
  - Stems end on a word or a comma, never on a space.
  - The 3 odd quote stems (2 start with `‘`, 1 with a leading space) are handled by `strip()`.
  - The BOM is removed (the literal in `load` is U+FEFF; 0 BOMs remain).
  - All raw classes start with a space, and no raw prompt ends with a space, so `rawc` has no boundary issue.
- The THINK fallback prefix "…Answer: <stem>" + " opt" is consistent.
- `P'` strips the trailing "Answer:"/"A:" for hind, NeQA, unknown, MMLU, algebra, redefine and modus, and leaves memo, quote and pattern unchanged, as intended.

### S8. Minor robustness
- **Truncation flag.** `eos` is built from `tok.eos_token_id` and `<|im_end|>`. If the generation config also stops on `<|endoftext|>`, an output ending with it is mis-flagged as truncated. Add `tok.pad_token_id` and the `generation_config.eos_token_id` list. Smoke truncation is 0%, so this only affects the report.
- **Results are lost if the job dies during generation.** Nothing is yielded until all ~4,640 generations finish, so a timeout or disconnect then loses all of them.
  - The 3 h cap is probably enough (thinkaloud did 2,060 items for about $4).
  - Still, yielding or saving generations per batch, e.g. to the volume, would make the run restartable without paying twice.
- **C4 output.** It prints per-family AUC but not the mean over families with the ≥ 5 rule. That is fine for report-only; add the mean if it is quoted.

## Checked and matching PLAN v2
- **Sizes:** 120 traps (60/60 by key) × 5, 80 controls × 4, 100 × 4 report-only, 1000 MMLU = 2320.
- **Bootstrap:** 2000 resamples; item scheme and cluster scheme within each family; wider of the two intervals. Clusters: memo by quote, unknown by story, hind by test question, others by item.
- **C1/C2:** ≥ +5 with low > 0 against each of direct, direct_mean and rawc, plus > 0 in ≥ 3 of 5 families.
- **C3:** pooled controls ≥ −5 with low ≥ −10; MMLU ≥ −2 with low ≥ −5.
- **Verdict:** each arm needs all of its criteria to pass.
- **Reporting:** parse rate, agreement and truncation per family. Per-class accuracy is printed.
