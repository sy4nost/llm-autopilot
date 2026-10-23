# Hiding past examples deeper: 4 head sets

Qwen2.5-14B-Instruct, Modal H100, 2026-10-16. As in rezultaty-blockex.md: only the block of past solved examples is
hidden from the selected heads; they see the rules, the change notice, the question and their own reasoning.
Our task: 270 held-out prompts (sets 15–29). MMLU 570, GSM8K 250.

Variants:
- **b20**: 20 copying heads (as in the first run; the result repeated exactly);
- **b20+early**: the same 20 heads plus all 360 heads of layers 20–28;
- **b40**: top 40 heads from the heads/ ranking (layers 29–38 and 43);
- **late_all**: all 760 heads of layers 29–47.

First I checked the guess about "carriers" in early layers. Heads in layers 20–28 look very little at the answer of
the identical past example (at most 1.5%). In layers 29+ it reaches 16%.

## Summary

| | old answers after change (D1+D3) | correct D1+D3 | correct D2+D4 | correct, no change | all correct | MMLU | GSM8K |
|---|---|---|---|---|---|---|---|
| normal | 43% | 51% | 80% | 57% | 63% | 77.5% | 93.6% |
| b20 | 33% | 54% | 96% | 61% | 70% | 77.9% | 93.6% |
| b20+early | 32% | 41% | 100% | 47% | 63% | 77.0% | 93.6% |
| **b40** | **31%** | 56% | **99%** | **63%** | **73%** | **78.1%** | **93.6%** |
| late_all | 32% | 53% | 94% | 62% | 70% | 77.7% | 90.4% (−3.2) |

b40 vs normal (paired bootstrap over sets, 95%):
- old answers −12% [−24, −1];
- correct D1+D3 +4% [−6, +14], not significant;
- D2+D4 +19% [+10, +28];
- no change +7% [0, +13];
- all together +10% [+6, +14].

On benchmarks b40 is free: MMLU +0.5 [−0.2, +1.4]; GSM8K 0.0 (1 fixed, 1 broken).

A third label, i.e. neither the correct nor the old one, on D1+D3 after the change: normally 6%, b40 13%. So about
half of the removed old answers turn not into the correct answer but into a third one.

## What it means
1. **The best variant is 40 heads**: +10 points on our task, breaks nothing on MMLU and GSM8K.
2. **Early layers 20–28 do not carry the copy, they learn from the examples.** If the examples are hidden from them,
   the exact copy goes away (A: 90% → 37% old), but the model also loses useful learning: correct without change
   A 100% → 57%, overall 63% → 63%.
3. **The remaining habit (~31% old answers) cannot be removed even by hiding examples from all late heads
   (late_all 32%).** It sits in the early layers together with the ability to learn from examples. It cannot be cut
   out with an attention mask without harm.
4. **The correct answer does not appear by itself.** When a rule is cancelled, the model does not always reach the
   new answer. Not only a "brake" is needed but also a push towards the new rule.
5. A gain claim needs a self-made GSM-NoOp: benchmarks here are only a harm check.

Files: modal_blockex2.py, settings2.json (head sets), score_blockex2.py, score2_out.txt (full output), blockex2_Qwen2.5-14B-Instruct.jsonl.
