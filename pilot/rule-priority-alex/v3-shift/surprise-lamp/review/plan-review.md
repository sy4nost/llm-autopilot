# Review of surprise-lamp PLAN v1 (2026-10-24)

Independent auditor. I read PLAN.md, `../conftrap-lamp/` (sets_cl.json, out_main_*.jsonl, modal_cl.py, rezultaty.md, audit/audit-report.md). I ran no Modal or GPU job. I ran one local script on the **saved fast answers only**. It computes no surprise values, so it does not break the plan's anti-fitting rule. It counts the Q1 cells and scores the trivial baselines. The script is at
`/tmp/claude-0/-home-user-llm-autopilot/07ce82f6-7b0d-5037-a120-6772e7201c02/scratchpad/q1base.py` (scratch, not in the project).

## What the data look like for Q1 (traps only, fast habit vs fast right)

| model | family | habit / right | eligible (>=10 each) | template AUC (LOO cluster error rate) | length AUC (chars) | CONF AUC |
|---|---|---|---|---|---|---|
| Qwen | D | 26 / 74 | yes | **0.82** | 0.34 | 0.80 |
| Qwen | hindgen | 57 / 43 | yes | n/a (each item unique) | 0.53 | 0.36 |
| Qwen | parcels | 74 / 26 | yes | **~0.79** (2 test cases) | 0.43 | 0.12 |
| Qwen | redef | 20 / 80 | yes | **0.92** | **0.26** | 0.70 |
| Qwen | bonus | 94 / 4 | no | | | |
| Phi-4 | D | 62 / 38 | yes | 0.65 | **0.31** | 0.67 |
| Phi-4 | bonus | 59 / 36 | yes | ~0.5 | **0.36** | 0.40 |
| Phi-4 | parcels | 41 / 58 | yes | **~0.81** (2 test cases) | 0.58 | 0.60 |
| Phi-4 | redef | 47 / 53 | yes | **0.88** | 0.58 | 0.59 |
| Phi-4 | hindgen | 99 / 1 | no | | | |
| OLMo | D | 64 / 36 | yes | **0.82** | 0.38 | 0.19 |
| OLMo | hindgen, redef, parcels, bonus | 93/7, 91/9, 100/0, 95/0 | no | | | |

That gives 9 eligible cells. OLMo has one (D).

Errors cluster by template:
- **redef**: 9 of 9 "Redefine e as N+N" items are habit errors for both Qwen and Phi-4. 0 of 14 "Redefine π as N" items are.
- **parcels**: all 100 traps share one New RULE 2 and use 2 test cases. Phi-4 makes the habit error on 6 of 51 "home" items and 35 of 49 "abroad" items. Qwen: 49/51 vs 25/49.

---

## BLOCKERS

**B1. Template identity alone passes Q1, so a pass would mean nothing.**
- The pass bar is |mean AUC − 0.5| ≥ 0.12.
- Template identity gives 0.65–0.92 in 7 of 9 eligible cells.
- MAIN is a deterministic function of the text, and items of one template share almost all their text. Within a template, the surprise of the counterfactual word is nearly constant: "View ÷ as a random digit" is the same string each time, and only the digits change.
- So any surprise statistic that differs between templates will "separate habit errors" through which template an item belongs to, not through a stumble.

Fix (all must be frozen before the run):
1. Make the primary measure a **within-template (stratified) AUC**. Count only (habit, right) pairs from the same template, pool the pairs within a cell, and drop templates that lack both outcomes.
   - Templates: redef = the 9 `cluster` templates; D = object; parcels = test case; bonus = set (or new-rule wording); hindgen = sign of EV × realized-outcome word.
2. Report the number of informative pairs per cell. Set eligibility on informative pairs, e.g. ≥ 30 pairs and ≥ 2 templates contributing, not on raw counts.
   - Expect redef to shrink a lot: most of its templates are pure.
3. Also report the LOO template baseline AUC next to MAIN, as a named rival.

**B2. The 5-token skip removes the stumble in redef, and possibly in D.**
- In redef, the counterfactual sits in the first tokens. "Redefine π as 974" puts the number at roughly token 4–6. "View ÷ as a random digit" puts "÷" and "random" in tokens 1–5.
- After the skip, MAIN will land on the arithmetic digits in the question.

Fix:
- Condition on the full chat prefix, which is already the case, and skip only the chat-template tokens, or at most the first content token.
- Or freeze a per-tokenizer check that the designated critical span (B4) is never skipped.

**B3. Q2 cannot be run for redef, and in parcels/bonus it tests something else.**
- redef has **no controls** (100 traps, 0 controls). Yet the prediction says "Q2 clearly yes for D and redef".
- In parcels and bonus, controls carry the **same** "IMPORTANT UPDATE … New RULE" sentence as traps. Trap and control differ only in whether the test case is affected by the change, which is a semantic fact and not an odd word.
- Q3 ("max inside the New RULE sentence") will be just as high for controls. It shows that the update is surprising, not that the item is a trap.

Fix:
- Restrict the Q2 prediction to D and hindgen.
- Report Q3 for traps **and** controls side by side.
- State that parcels/bonus cannot show a trap-specific lexical stumble by design.

**B4. MAIN probably cannot find the real stumble in long texts. Add an oracle-position variant so a null can be read.**
- In hindgen the stumble is real: the unlikely outcome word ("ends up **losing**" after "85 percent chance of earning"). It is worth about 1.5–3 nats.
- But every one of the 10 shots holds random names and amounts, and a fresh name or random number token is easily 5–10 nats.
- In parcels/bonus, MAIN takes a max over about 1500 tokens of random example attributes.
- Without a check of what MAIN catches, a null cannot tell "the stumble carries no signal" from "the max picked a name".

Fix:
- Freeze a critical span per family before the run, as a token range computed from the text:
  - D: the stated number.
  - redef: the redefined symbol or value.
  - hindgen: the outcome word after "ends up".
  - parcels/bonus: the changed word in the New RULE ("standard", "no").
- Report surprise at that span (word-level sum) as a variant for Q1 and Q2.
- Report where MAIN's argmax lands (in the span / name / number / instruction tail), per family and population.
- This turns the plan's own prediction ("hindgen weak, nothing odd in the words") into a test. That prediction is likely wrong in its reasoning: there is an odd word in hindgen, and the max will drown it.

## SHOULD FIX

**S1. Averaging signed AUCs cancels opposite directions, and that makes the CONF bar easy to clear.**
- Qwen's CONF AUC is 0.80 (D), 0.12 (parcels), 0.70 (redef) and 0.36 (hindgen). The mean is about 0.50, even though CONF is strong in every cell.
- If the comparison uses |mean AUC(CONF) − 0.5|, the rival is cancelled out by averaging.
- The same cancellation can hide MAIN, because its direction may legitimately differ by family (e.g. D "noticed → right" vs parcels "strong old rule → habit").

Fix:
- Compare per cell, using the mean over cells of |AUC − 0.5| for both MAIN and CONF.
- Separately require a sign-consistency count (MAIN has the same sign in ≥ 6 of 9 cells).
- Or fix CONF's direction a priori (low confidence → error) and say plainly that on these selected confident traps CONF is near 0.5 or reversed by construction (audit c-6). Beating it is then weak evidence on its own.

**S2. The bootstrap must be clustered.**
- Resampling items within cells ignores clustering: parcels has 2 test cases, redef 9 templates, D 15 objects, bonus 58 sets.
- The previous review (B1 there) already required cluster resampling for this item set.
- Item bootstrap will give falsely narrow intervals, above all when the effect is a template effect.

Fix:
- Use a two-level bootstrap: resample clusters within a cell, then items within clusters, using the conftrap `cluster` field. With stratified AUC (B1) this is natural.
- Bootstrap the **signed** difference with the direction fixed from the full sample. |·| differences are biased upward near 0.
- State that "low > 0" is one-sided at whatever level is intended.

**S3. Add the missing trivial baselines as named rivals, and require MAIN to beat each one.**
- **Text length** in tokens. Within-cell length AUC is already 0.26 (Qwen redef), 0.31 (Phi-4 D), 0.36 (Phi-4 bonus) and 0.34 (Qwen D), all past the 0.12 bar.
- Length matters because a max over more tokens is larger by extreme-value statistics alone, so MAIN rises with length.
- **Template identity** (LOO error rate; see B1).
- **Item numbers**, family-specific:
  - D: the stated count and |stated − real| (AUC 0.55–0.59).
  - hindgen: the stated probability of the realized outcome (Qwen 0.43). The outcome word's surprise is mostly −log of this number.
  - redef: the size of the redefined value.
- **Position of the change** (token index of the critical span). It is constant within parcels/bonus/hindgen but varies in D/redef with the frame ("In this version of the world" vs "In this book").

Better, and cheap: a LOO-within-cell logistic model asking whether MAIN adds AUC over (CONF + length + template). Report it as the "adds information" number. Also report the Spearman correlation of MAIN with CONF and with p(habit): surprise and the fast answer come from the same weights and may be one quantity.

**S4. Window, floor and token-vs-word.**
- D/redef texts are about 20–30 tokens, so the ±10 window covers almost the whole text. There z_local ≈ z_task, and MAIN = min(·,·) is just z_task. Freeze the edge rule: a truncated window with a minimum of 6 neighbours, else use z_task.
- Multi-token numbers ("974", and Qwen splits digits) put the word's own later tokens into the neighbour set. That inflates the local mean and sd and lowers z_local of the first token.
  - Fix: aggregate surprise per whitespace word (sum over its tokens), which is what Alex said ("word"). Exclude the word's own tokens from its neighbourhood.
  - This also makes the three tokenizers comparable.
- The 0.5 floor is in nats on per-token s. In the repetitive example blocks of parcels/bonus the floor will bind almost everywhere, so z_local becomes (s − mean)/0.5 there. That is acceptable, but say it explicitly and keep the floor the same per word under the word aggregation.
- The DIRECT user turn in "q" mode ends with R_TAIL ("Reply with only the answer, nothing else."). Exclude R_TAIL tokens from the max and from the z_task statistics. They are constant and not part of the task, and in 25-token texts they are about 40% of the tokens.

**S5. Eligibility leaves OLMo with one cell.**
- Under ≥ 10/≥ 10, OLMo has only D.
- "Same direction in ≥ 2 of 3 models" then means Qwen + Phi-4, or one of them plus one OLMo cell. Say so.
- Do not count OLMo as a replication unless it has ≥ 2 eligible cells (or ≥ 2 after B1).

**S6. Is Q1 the right question for Alex's idea?**
- Alex's idea is "notice cheaply first, check expensively second": a **trigger**.
- In conftrap, reasoning fixes about 75% of habit errors. A useful trigger therefore only has to flag the items where the habit conflicts with the context (traps), not predict who falls. That is Q2 turned into a gate.
- Q1 (within traps, who falls) is a different and harder mechanistic question, and the plan already predicts it is weak.

Recommendation:
- Add a co-primary **gate test**, within family, on traps + controls: recall of fast errors at a matched flag rate (as in conftrap), MAIN vs CONF vs length vs template. Report accuracy of "reason only where flagged" against "always reason" and "never reason".
- It is only meaningful for D and hindgen (see B3), and D is close to a fact-checker ("hexagon has 4 sides" vs true counts). Say this up front.
- CONF needs the same single forward pass. "Pre-answer" gives no compute advantage over CONF, only a possible information advantage. State that, so the comparison is not framed as cheap vs expensive.

## NOTES

- N1. The items are reused and the tests are exploratory, as the plan states. For parcels, "fresh items" later must use new test cases: the current traps are 2 questions (audit c-7).
- N2. Pin the content of the PREREG hashes. It should include the per-family critical-span extractor (B4), the template map (B1) and the word-segmentation rule (S4).
- N3. Verify that the surprise pass rebuilds byte-identical text: `chat(prompt + R_TAIL)` for "q" items and `chat(prompt)` for "label" items, without the "The answer is" stem. Assert this on 5 items per family against the token count implied by the saved run.
- N4. Qwen's chat template injects a default system prompt. It is the same for all items and harmless, but say that surprise is conditional on it.
- N5. Report all 4 variants (and the span variant) per cell, not only means, so a reader can see whether one family drives the result.
- N6. Q2 on D and on hindgen should pass almost by construction: a false stated count, or an outcome against the odds, is a low-probability token. Treat a Q2 pass as a pipeline check, not as support for the idea.
- N7. Habit vs right drops third-option answers. That is fine, but give the count per cell (0–5% earlier).
- N8. The cost (~$0.5) is fine. Save the per-token arrays (token id, offset, logprob) so all fixes that need no GPU can be applied without rerunning. Then B1/S1–S4 can still be adjusted before results are looked at, provided the analysis code is frozen first.

## Verdict

Do not run the analysis as written. The GPU pass itself is fine to run once the arrays are saved (N8), because it involves no choices.

Before any surprise is looked at, freeze these:
- within-template AUC and informative-pair eligibility (B1);
- the skip fix (B2);
- the critical-span variant and argmax-location report (B4);
- per-cell |AUC| comparison with clustered signed bootstrap (S1, S2);
- the length, template and numeric baselines (S3);
- word-level aggregation and R_TAIL exclusion (S4).

Then fix the Q2/Q3 scope (B3) and consider adding the gate test (S6).
