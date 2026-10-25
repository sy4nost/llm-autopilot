# conflict-train: can a model be trained to FIND conflicts, and does it transfer to unseen kinds? (PLAN v0.2, 2026-10-25; Alex's idea)

Goal (Alex): the model does not solve tasks. It only states whether the context holds a conflict and why. A conflict means two parts
of the context call for different answers to the final request: instruction vs shown answers, instruction vs knowledge, old rule vs
new rule, old value vs new value, data vs data. Question: does the skill transfer to conflict kinds absent from training?

## Output the model is trained to write
`Conflict: yes|no. Between: <source> and <source>. Why: <one sentence>.` Solving stays with plain "think it through", which already helps.

## Data
Own items (make_ctr.py, seed 20261025 -> own_items.jsonl, 2200 items). Every family is half conflict, half decoy, and the decoys
reuse the same parts and words with all parts agreeing:
- T1: the rule changed after 6-15 worked examples (5 domains). Decoys: a real change that does not affect this case, or an "update" that changes nothing.
- T2: an instruction to answer with word X, while earlier assistant replies say Y. Decoy: the replies say X.
- T3: a context claim vs a well-known count ("for this puzzle, a hexagon has 9 sides"). Decoy: the claim matches the fact.
- T4: an earlier format instruction replaced by a later one. Decoys: confirmations, and "change" words about non-instructions.
- Held out: H1, a stale personal value in a dialogue (user updates or confirms it). H3, two different numbers for one quantity in a word problem.
- Split: T1-T4 have 400 train and 50 dev items each. H1 and H3 have 200 test items each.

Outside data, test only unless stated:
- WildChat-1M (real ChatGPT chats, ODC-BY). Candidate turns where the user changes a requirement are mined by phrases, then labelled by a
  strong model with a hand-checked sample. Some conversations go to training, and disjoint conversations go to test.
- Berkeley CICM (have it; CC BY-NC, never committed).
- Hindsight-neglect.
- WikiContradict.
- MultiChallenge (versioned editing / instruction retention).
- Lost in Conversation.
- ConflictBank, if its licence allows.
- Alex's English text with 6 planted false sentences (odd-phrase track).
- Plain GSM8K and MMLU (100 each) for false alarms.
- The final split by set is frozen before any run.

## Arms
- U: the untrained model, given the same request with a definition and 2 examples.
- L: a LoRA adapter on the same model, trained on the train split. Early stop on dev; the adapter can be removed, so before vs after is paired.
- W: a word-only classifier (TF-IDF + logistic regression) trained on the same train split. If W matches L on held-out kinds, L learned words.
- Model: Qwen2.5-7B-Instruct. A second model only if the test passes.

## Predictions (fixed before any output)
- K1 (main): on the held-out sets pooled, L's balanced accuracy is >= 0.75 AND exceeds U by >= 0.10 (cluster bootstrap CI low > 0).
- K2: L false alarms on plain tasks are <= 10%, and on H1/H3 decoys <= 25%.
- K3: L beats W on the held-out sets by >= 0.10.
- OVERALL PASS = K1 and K2 and K3.
- Reported, no verdict:
  - correctness of "Between";
  - per-set tables;
  - flag -> think vs always-think vs never on held-out answers.
- Guess: H1 (old vs new in one source) is hardest, H3 easiest.

## Steps and cost
1. Alex checks sample items.
2. Independent plan/data review.
3. Smoke run on 30 items.
4. Training (< 1 h).
5. Exam.
6. Report.

Estimate ~$9-17 on Modal (training ~$2, exams ~$3, think arm ~$3, WildChat labelling ~$3-5, smoke ~$2), not verified.

## Risks
- Format shortcuts. Countered by decoys with shared words, held-out families and outside sets, and the W control.
- Templated "why" text. Several phrasings are used, and only yes/no is scored.
- Labels on mined real chats are not exact. A hand-checked sample gives their accuracy.

## v0.3 amendments (2026-10-25, fixed before any model output)
- WildChat: labels are by hand (Claude), not by a strong model; 125 candidate and 60 plain conversations, unclear ones (S) dropped
  (make_wc.py -> wc_items.jsonl, 159 items, sha 43592b64...). Test: candidates 0-79 + plain 0-39 (103 items, 35 yes). Train/dev: the rest (56).
  Because this kind is partly in training, X_WC is reported, not part of the K1 pool.
- GSM8K is not available locally; plain false alarms use X_MMLU (200) only.
- WikiContradict: 150 items, recall only (all yes), two contexts + the question (X_WIKI).
- Lost in Conversation and ConflictBank: not added.
- Final sets (ext_items.jsonl, 955 items, sha e41caede...): X_CICM 120 yes; X_MC 273 yes; X_HIND 100/100; X_ODD 6/6; X_MMLU 200 no; X_WIKI 150 yes.
- K1/K3 pool = H1, H3, X_CICM, X_MC, X_HIND, X_ODD, X_WIKI, pooled balanced accuracy (mean of TPR over all yes and TNR over all no).
  Macro average over sets is reported too. K2 plain = X_MMLU; decoys = H1/H3 "no" items. Unparsed output counts as "no".
- Scoring code: score_ctr.py (tested on simulated outputs only).

## v0.4 amendments after the independent review (2026-10-25, still before any model output). These replace v0.3 where they differ.
- Cue words crossed with labels: T1/T4 got reminder-worded real changes and update-worded non-changes; H1 got cue words on both labels
  (now/update/still/same/nothing/but/not/as before each appear in yes and no). own_items.jsonl regenerated (sha 8ec7027d...).
- Matched negatives for outside sets: X_CICM 120 yes + 120 no (same conversation, every stale value replaced by the current one);
  X_WIKI deduped by question and identical-context rows dropped, 97 yes (both passages) + 96 no (one passage + an unrelated passage).
  ext_items.jsonl sha f2cac143... WildChat test chats that open like a training chat are dropped (wc_items.jsonl sha c64baf12..., 102 test).
- X_MC is report-only (recall): many of its items do not meet our conflict definition. X_ODD (12 items) is report-only.
- Every arm's reply is prefilled with "Conflict:", so U cannot lose on format; p_yes of the first token is stored (AUC reported).
- K1: macro balanced accuracy over the two-class held-out sets H1, H3, X_CICM, X_HIND, X_WIKI >= 0.75 AND >= U + 0.10 with the
  bootstrap CI low > 0 (clusters resampled within each set). X_CICM is near-transfer (old vs new value), reported as such.
- K2 unchanged: X_MMLU false alarms <= 10%, H1/H3 decoy false alarms <= 25%.
- K3: L >= the best of three word-only controls + 0.10 with CI low > 0. Controls: W (TF-IDF+LR on the transcript, balanced classes),
  Wlast (same on the last two user turns), Wcue (a fixed change-word list on the last user turn).
- Training: true step count for the LR schedule; early stopping on the dev verdict-token loss (yes vs no only).
- Packages pinned: torch 2.14.1, transformers 5.20.0, accelerate 1.14.0, peft 0.21.1.

## v0.5 amendment after the second review (2026-10-25, before any model output)
- X_WIKI negatives turned out to be off-topic (word overlap alone separates them, AUC 0.97), so X_WIKI is report-only: recall on its
  97 conflicts. K1/K3 pool = H1, H3, X_CICM, X_HIND (macro balanced accuracy).
- Honest scope: H1/H3 are our own templates and CICM is near-transfer (old vs new value), so a PASS mostly shows near transfer.
  Far transfer is reported separately without a verdict: X_HIND balanced accuracy, X_WIKI recall, X_WC (real chats).
- The word controls are near chance on the pool (~0.50), so K3 in practice means L >= ~0.60 with a CI.
- Sensitivity check reported: verdict from p_yes > 0.5 instead of the text parse.
- First paid steps are tiny checks: eval smoke on 30 items (p_mass near 1 for U) and train with --n 16 --epochs 1, before the full run.
- After the U smoke (format clean, p_mass 1.0 on 30/30): the vague CICM filler turn "I used to have a different approach, but I've changed recently." is removed from both versions of every CICM pair (ext_items.jsonl sha b99e03cd...).
