# Selective blocking: 20 "copying" heads cannot see only the past examples

Qwen2.5-14B-Instruct, Modal H100, 2026-10-16. The same 20 heads (layers 29–38) as in heads/.
Previously the heads were switched off entirely. Now they work, but an attention mask hides only the block of
solved examples from them. They still see the rules, the change notice, the question itself and their own reasoning.

## One-line summary
The overall gain on our task is the same (+8 points, 63% → 70%), but now free: GSM8K 93.6% → 93.6%, MMLU 77.5% → 77.9%.
The price: old answers fall half as much as with full switch-off.

## Our task (270 held-out prompts, sets 15–29)

| | old answers after change (D1+D3) | correct D1+D3 | correct D2+D4 (new rule adds HOLD) | correct, no change | all correct |
|---|---|---|---|---|---|
| normal | 43% | 51% | 80% | 57% | 63% |
| heads fully off | 21% | 61% | 93% | 57% | 70% |
| **only examples hidden** | **33%** | 54% | **96%** | **61%** | **70%** |

Difference from normal (paired bootstrap over sets, 95%):
- old answers −10% [−19, −2]; correct D1+D3 +3% [−6, +11] (not significant);
- D2+D4 +16% [+7, +24]; no change +4% [0, +9]; all together +8% [+3, +13].

By condition, old answers on D1+D3: A (same case in history) 90% → 77% (fully off: 60%);
B (similar cases) 37% → 13% (3%); C (no similar cases) 3% → 10% (0%).

## Benchmarks (harm check)

| | MMLU 570 | GSM8K 250 |
|---|---|---|
| normal | 77.5% | 93.6% |
| heads fully off (previous run) | 76.0% (−1.6) | 88.0% (−5.6 [−9.6, −2.0]; 19 broken, 5 fixed) |
| **only examples hidden** | **77.9% (+0.4)**; 3 fixed, 1 broken | **93.6% (0.0)**; 1 fixed, 1 broken |

So what broke math was the heads not seeing their own solution steps. If only the past examples are hidden from
them, there is no harm.

## What it means
- We learned to weaken the pull of past experience without touching the rest of the model's work. This is the first
  variant with no benchmark cost.
- Exact copying (condition A) goes away less: 77% old answers vs 60% with full switch-off.
  This is an inference, not a test: most likely the layers below 29 have already moved the "ready answer" from the
  examples into the tokens of the question itself. The heads read it from there, bypassing the mask.
- Next step on this line: hide the examples from the same heads in several earlier layers too (e.g. 20–28), or find
  the early "carrier" heads.
- A gain can be claimed only on a self-made GSM-NoOp (benchmarks here are a harm check).

Files: modal_blockex.py (run), score_blockex.py (scoring), score_out.txt (full output), blockex_Qwen2.5-14B-Instruct.jsonl (raw answers).
