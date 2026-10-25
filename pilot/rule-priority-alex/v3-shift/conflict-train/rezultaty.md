# conflict-train results (2026-10-25): OVERALL FAIL

Question (Alex's idea): can Qwen2.5-7B-Instruct be trained (LoRA) only to *notice* conflicts in a context, and does that skill
transfer to conflict kinds absent from training? Plan and frozen criteria: PLAN.md (v0.4/v0.5 amendments apply). Two independent
reviews before the run. Cost about $3 on Modal (L40S).

## Setup in one paragraph
Every arm reads the same prompt (definition + 2 examples) and its reply is prefilled with "Conflict:", so only yes/no is scored.
U = untrained model. L1 = LoRA r16 trained 2 epochs on 1645 items: own families T1 (rule changed after worked examples),
T2 (instruction vs earlier answers), T3 (claim vs known fact), T4 (format instruction replaced), each half decoys sharing the same
words, plus 45 hand-labelled WildChat chats. W, Wlast, Wcue = word-only controls. Dev verdict accuracy rose 69 -> 98 out of 100.

## Main table (balanced accuracy = mean of "conflicts caught" and "clean items left alone"; 50 = coin)

| set | n | U | L1 | role |
|---|---|---|---|---|
| H1 stale personal value (own template, held out) | 200 | 0.90 | 1.00 | K1 pool |
| H3 two numbers for one quantity (own template, held out) | 200 | 0.76 | 0.99 | K1 pool |
| X_CICM Berkeley stale value, matched pairs | 240 | 0.55 | 0.65 | K1 pool (near transfer) |
| X_HIND hindsight neglect | 200 | 0.50 | 0.50 | K1 pool (far transfer) |
| X_WIKI WikiContradict (negatives off-topic) | 193 | 0.62 | 0.73 | report only |
| X_WC real ChatGPT chats, test | 102 | 0.66 | 0.71 | report only |
| X_MC MultiChallenge, recall | 273 | 0.27 | 0.12 | report only |
| X_ODD Alex's text, planted sentence | 12 | 0.50 | 0.50 | report only |
| X_MMLU plain questions, clean-left-alone | 200 | 0.84 | **0.53** | K2 |
| own dev T1-T4 (in distribution) | 200 | 0.69 | 0.98 | report only |

Pool macro: U 0.677, L1 0.787, best word control 0.51. L1 - U = +0.110 (95% CI +0.088..+0.134).

## Verdict against the pre-registered criteria
- K1 PASS: 0.787 >= 0.75 and L1 - U = +0.11 >= 0.10, CI above 0. But the gain comes almost entirely from H1/H3, which are our own
  templates; CICM +0.10, hindsight +0.00.
- K2 FAIL: false alarms on plain MMLU questions 47.5% (limit 10%; untrained 16%). Decoy false alarms 0.5% (fine).
- K3 PASS: L1 beats the best word control by +0.28 (CI +0.25..+0.30).
- OVERALL FAIL. Sensitivity (verdict from p_yes > 0.5): same picture (pool macro U 0.680, L1 0.791).

## What went wrong (post hoc, not pre-registered)
- The trained model treats the wrong answer options of a multiple-choice question as "claims that contradict general knowledge":
  72 of its 95 MMLU false alarms say "Between: the claim in the context and general knowledge", e.g. "the text says A. England,
  but in reality U2 is from Ireland". This is the T3 training template applied where it does not fit. It learned the template's
  surface ("a stated thing that is false"), not "two parts call for different answers".
- Hindsight neglect: neither U nor L1 ever says yes (0/100 traps caught), although p_yes ranks traps above controls somewhat
  (AUC U 0.73, L1 0.70). Training added nothing for this far kind.
- CICM: L1 catches fewer stale-value conflicts (62 -> 48 of 100) but raises fewer false alarms (52 -> 18 of 100); AUC 0.57 -> 0.72.
- MultiChallenge recall dropped (27 -> 12 of 100); many MC items do not fit our definition, so this is weak evidence.

## Plain conclusion
Training made the model near-perfect on conflicts that look like its training families (H1/H3 differ in words but share the
"old value vs new value" and "two numbers" shapes), and somewhat better on real chats and WikiContradict. It did not learn a
general "notice a conflict" skill: on a far kind (hindsight) nothing changed, and on ordinary multiple-choice questions it now
raises a false alarm about half the time.

## Caveats
- One model, one seed, one training run; LoRA settings were not tuned.
- WildChat labels are hand labels by one annotator (Claude); X_WIKI negatives are easy; X_ODD has 12 items.
- CICM and MultiChallenge data are not committed (licences); items are rebuilt by make_ext.py from the original sources.

## Files
PLAN.md, make_ctr.py, make_ext.py, make_wc.py, ctr_prompt.py, modal_ctr.py, score_ctr.py; outputs in out/ (main_U.json,
main_L1.json, train_L1.json, score_main.json).
