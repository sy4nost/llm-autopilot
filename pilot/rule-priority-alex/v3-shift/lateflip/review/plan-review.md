# lateflip PLAN v1: pre-run review (2026-10-24)

I read PLAN.md, ../conftrap-lamp/modal_cl.py, sets_cl.json, out_main_*.jsonl, ../surprise-lamp/score_sp.py and report_main.txt, and ../inside/modal_inside.py.
All numbers below come from local `python3 -I` runs on the saved files. huggingface.co is blocked from this sandbox, so I could not load the
tokenizers or configs. I inferred the tokenization from the per-token lengths of `res.direct`, which DIRECT saved.

## BLOCKERS

**B1. The answer token is confounded with the population (hindgen especially).** Within a template, a habit error and a fast right answer pick
*different option tokens* by construction. hindgen is the extreme case: cands are always ['Y','N'], and the template is the key, so inside
template keyY every habit error answers " N" and every right answer answers " Y". This is true in 100% of hindgen templates; I checked this in the
qwen14 and olmo13 cells, which are both eligible. Mid-layer logit-lens logits restricted to 2-3 option tokens are dominated by token priors
(unigram frequency, the Y/N bias, small-digit bias). If the mid-layer lens prefers " Y" everywhere, every " N" answer "settles late". SETTLE then
separates the populations perfectly in hindgen with no lamp behind it. In the other families the option strings vary within a template (0-33% of
templates have one fixed token per side), but digit and word priors can still line up with habit vs key. For example, in D the habit is the stated
count and the key is the corrected count, so they have a systematic numeric relation.
*Fix (pre-register it):* (a) Debias the lens per model x family x layer. Subtract each option token's mean lens logit at that layer, computed over
all items of the family (traps and controls), leave-one-cluster-out, before you take leader(l). Make debiased SETTLE primary or co-primary.
(b) Add a trivial baseline, TOKPRIOR: the SETTLE you would expect from the answer token alone, meaning the mean SETTLE of other-cluster items with
the same final-leader token. SETTLE must beat it the same way it must beat CONF and LEN. (c) For hindgen, also report the comparison with the
answer token held fixed: habit errors that answered " N" in keyY against right answers that answered " N" in keyN. This crosses templates, so label
it report-only.

**B2. Compute the lens logits in fp32, not bf16.** The saved DIRECT scores show exact final-layer ties caused by bf16 logit resolution. The step is
0.125 for |logit| in [16,32]. Examples: -0.6931 / -0.6931 (exactly 50/50) for distinct parcels words, and identical second-token values for " 1" and
" 9" in phi4 redef. Trap items with a tied top option at the *final* layer: qwen14 parcels 4 and D 4, phi4 redef 5, parcels 2 and bonus 4, olmo13
bonus 1. Mid layers have smaller margins, so ties will be more frequent there. With argmax tie-breaking by option index, leader(l) and SETTLE then
depend on option order. Option order is not balanced (for example, the redef key is at index 1 in 58 of 100 items).
*Fix:* compute `norm(h_l).float() @ lm_head.weight[opt_ids].float().T`. That is cheap, because it uses only the option rows. Pre-register the tie
rule: a tie with the final leader counts as "final leads", not as a flip. Report the tie counts.

**B3. Specify the final-norm and indexing details in the plan, and assert them in the smoke run.** In HF Qwen2, Phi3 (phi-4) and OLMo2 with
transformers 4.5x/5.x, `hidden_states` has L+1 entries: the embeddings, then one entry per block. The **last entry is the output after
`model.norm`**: the hidden-state recorder replaces the last block output with `last_hidden_state`. So `lm_head(final_norm(hidden_states[-1]))`
double-norms (RMSNorm with weights is not idempotent). The pre-norm output of the last block is not exposed. inside/modal_inside.py does this
correctly: `hidden_states[1:-1]` plus `model.norm`, and the real logits for the last layer. Copy that pattern and state that d = l/(L-1) uses
l = 0..L-1 over the block outputs `hidden_states[l+1]`. Smoke asserts:
- `max|lm_head(hidden_states[-1]) - logits| ~ 0`. This proves the last entry is already normed.
- The final-layer leader equals the saved DIRECT fast answer. On the saved data the first distinguishing token's argmax already agrees with the
  full-sequence argmax in every item except one qwen14 D trap.
- Report `config.tie_word_embeddings`. From memory all three are untied (Qwen2.5-14B, phi-4 and OLMo-2-13B), which is fine for the logit lens but
  should be checked. OLMo-2 puts its norms after the sublayers and keeps one final `model.norm`, so `model.norm` is the right norm for all three.

Use the saved fast answer for the population labels and drop disagreements, reporting the count.

## SHOULD FIX

**S1. Confidence/margin confound.** A smaller final margin mechanically allows a later flip. In the saved data the fast answers are mostly
saturated (median CONF is about 0.000-0.06 in most cells), but not always. qwen14 parcels: right 0.076 vs habit 0.000. olmo13 hindgen:
right 0.223 vs habit 0.059. olmo13 redef: right 0.228 vs habit 0.004. In surprise-lamp, CONF alone already reached a mean |AUC-0.5| of 0.132, and
its sign flips across cells (qwen parcels 0.23, qwen redef 0.74). "Beats CONF in strength" does not show that SETTLE carries information beyond
CONF. Add one pre-registered control:
- residualize SETTLE on logit(CONF) (or on the final-layer lens margin of the distinguishing token, MARGIN) within each cell, then compute the
  within-template AUC; or
- use within-template pairs restricted to the same CONF tertile of the cell.

Also add MARGIN, the final lens margin, as a rival next to CONF; for multi-token options it is not the same as CONF.

**S2. SETTLE may saturate.** The logit lens changes a lot in the last 1-3 layers. If the penultimate layer often disagrees, SETTLE piles up at its
maximum (L-2)/(L-1), the AUC drifts to 0.5 through ties, and a null cannot be told apart from "the alarm always rings". Pre-register a report of
the SETTLE distribution and the share at its maximum, per model x population. Add a less saturable secondary score: the fraction of layers with
d >= 0.5 where the final leader does not lead.

**S3. The distinguishing-token rule.** The token lengths saved in `res.direct` are identical across the three models except for D. " d" is 2
tokens (a bare space, then the digit) in all three models, so D and redef always go to the second token, as the plan expects. Two-digit D options
are 3 tokens in qwen14 (single-digit split) and 2 tokens in phi4 and olmo13 (cl100k-style tokenizers, up to 3 digits per token). So for qwen14 the
"second token" of "13" is "1", which coincides with " 1" or "12". 9 of 100 qwen14 D traps have such a collision. In none of them do the key and the
habit collide, so it is always the third option. The plan's "items whose tokens still coincide are dropped" does not say whether a *partial*
collision drops the item. Say that it does not: drop only items where key and habit collide, and restrict the leader to the distinct options. If
the 9 items are dropped, the qwen14 D cell falls from 94 pairs to 69 (still eligible).

The parcels and bonus labels are read directly after the chat template, and the first tokens look distinct. The equal values I found in the saved
scores are bf16 ties, not shared tokens, but I could not verify this with the tokenizers. Have the smoke run dump the distinguishing ids per item
and count collisions per model x family. For multi-token parcels words (2-3 tokens in 339 of 420 option slots), the first piece may be a generic
fragment. That is acceptable at the final layer (agreement is 100% on saved data) but weaker at mid layers. Note it as a limitation.

**S4. Too few eligible cells and templates.** The cells are the same as in surprise-lamp: **10 eligible**.
- qwen14: hindgen 913 pairs/2 templates, redef 46/3, parcels 698/2, D 94/6 (69/6 after drops).
- phi4: redef 124/5, parcels 725/2, D 119/13.
- olmo13: hindgen 313/2, redef 83/6, D 83/9.
- Skipped: qwen14 bonus (4 pairs), phi4 hindgen (1 template), phi4 bonus (21), olmo13 parcels and bonus (0 right answers).

Four of the ten cells (hindgen x2, parcels x2) have only 2 templates, and they carry most of the pairs. There the "clustered" bootstrap
resamples 2 units within the cell; for parcels and hindgen the clusters are the 2 templates or items, so the CI is not meaningful at that level.
qwen14 redef has only 20 habit errors and olmo13 redef only 9 right answers. After the B1 fix, hindgen may effectively become uninformative.
State in advance that removing hindgen still leaves 8 cells, and run the verdict with and without hindgen. The ">= 2 models" rule then rests on
olmo13 with 2 cells (redef, D).

## NOTES

- N1. A pass is uninterpretable unless B1 and S1 are handled: a token-prior artifact or a CONF proxy would both "pass". A null is uninterpretable
  without S2 (saturation) and B2 (ties). Logit-lens mid layers are known to be poor for some models; a tuned lens is not available. A null
  therefore means "no lamp visible through the logit lens", not "no late decision".
- N2. Q2 fire rates. Compare traps with controls only where controls share the option vocabulary: D, hindgen, parcels, bonus. mmlu A-D is not
  comparable. Report the rates both raw and after debiasing.
- N3. Q3 leader curves. Plot the debiased curves too, and split by answer token for hindgen. Otherwise "the old answer takes over late" can simply
  reproduce the Y/N prior.
- N4. Reproduce DIRECT's id construction exactly: `tok(prefix) + tok(option)` concatenated as id lists, not the joined string re-tokenized. For D
  and redef the lens is read at the position after the appended bare " " token.
- N5. Model depth differs (from memory: Qwen2.5-14B has 48 layers, phi-4 and OLMo-2-13B have 40), so the SETTLE grids differ. That is fine
  within cells, but do not pool raw SETTLE across models.
- N6. The plan says to freeze the scorer with hashes. Include the debias, tie and drop rules (B1-B3, S3) in the frozen scorer.
- N7 (housekeeping, my error). An auditor `pip download` placed 16 wheel files (`*.whl`: tokenizers, huggingface_hub, etc.) into
  `../conftrap-lamp/`. My attempt to delete them was blocked. Please remove `../conftrap-lamp/*.whl`. No other files were touched.
