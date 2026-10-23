# Did we break the model? MMLU and GSM8K (14B, 2026-10-16)

A check on independent well-known tests: plain Qwen2.5-14B-Instruct vs the same model with the 20 "copying" heads
switched off (`top20.json`, layers 29–38, found in `../heads/`). The model may have seen these tasks in training, but
that is the same for both variants, so a drop means harm from the intervention.

## Setup

- MMLU: 570 questions (10 from each of 57 subjects), 5 solved examples from the same subject in the prompt, answer =
  the letter with the highest probability. Heads switched off: everywhere (off_all) or from the question itself (off_q).
- GSM8K: the first 250 problems, 4 solved problems in the prompt, the model solves step by step (up to 400 tokens),
  answer after `####`. Heads switched off everywhere, including while the model writes its solution.
- Files: `modal_bench.py`, `bench_Qwen2.5-14B-Instruct.jsonl`, `score_bench.py`, output `score_bench_out.txt`.

## Result

| test | normal | 20 heads off | difference (95% interval) |
|---|---|---|---|
| MMLU, 570 | 75.3% | 76.0% | +0.7% [−1.6, +3.0] |
| MMLU, off from the question | 75.3% | 75.6% | +0.4% [−1.9, +2.6] |
| GSM8K, 250 | **94.8%** | **88.0%** | **−6.8% [−10.4, −3.2]** |

On GSM8K switching off broke 20 problems and fixed 3. The errors are typical of losing "copying": the model confuses
or loses numbers from the problem statement (e.g. 1596 → 798, 694 → 664).

## What it means

1. **For multiple-choice questions the heads are not needed** (MMLU unchanged).
2. **For long reasoning they are needed.** The same heads help carry numbers and intermediate results through the
   solution. Switching them off permanently drops math by almost 7 points.
3. So "just always switch off" will not do. Selectivity is needed: either a lamp (switch off only at the moment of
   conflict with the rule), or switch off not whole heads but only their view of **past examples**, leaving them their
   view of the current task. The second matches Alex's idea exactly: leave experience alone, solve the current task yourself.

## Next

- A variant "heads may not look only at the solved examples": check on our task (what remains of the gain) and on
  GSM8K (whether the drop goes away).
- Our own GSM-NoOp (new numbers and names plus an irrelevant sentence) to test improvement on tasks the model has not seen.
