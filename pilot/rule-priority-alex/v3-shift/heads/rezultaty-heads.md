# "Copying" attention heads in 14B (2026-10-16)

Goal: find the specific places in the model through which a past case pulls the answer, and weaken only them.

## Setup

- The same A/B/C sets (history 60): with a rule change D1–D4 and without a change D1, D3. Qwen2.5-14B, bfloat16, H100.
- Step 1 (540 prompts): for each of the 1920 attention heads (48 layers × 40), we measured how much attention goes
  from the answer position to the label of the past example that **matches the test case letter for letter**.
- Heads were ranked only on **the first half of the sets** (1–15, condition A, D1/D3, 30 prompts).
- Step 2: on **the second, held-out half** (sets 16–30, 270 prompts) the output of these heads was zeroed from the
  test case onward. Variants: top 10, 20, 40 heads and, as a control, 20 random heads.
- Answer = the label with the highest first-token probability. Files: `modal_heads.py`,
  `heads_Qwen2.5-14B-Instruct.jsonl`, `score_heads.py`, output `score_heads_out.txt`.

## Which heads

The top 20 sit in layers 29–38: L36.10, L37.18, L34.29, L29.11, L34.28, L32.9, L33.2, L36.21, L32.4, L34.6, L36.23,
L29.13, L29.20, L29.10, L36.36, L34.27, L35.29, L29.18, L38.33, L29.21. Exactly where, by the logit lens, the old
answer takes over from the new one (after layer 32). This is about 1% of all heads in the model.

## Result on held-out sets (all conditions together)

| what is switched off | old answer after change (D1+D3) | correct D1+D3 | correct D2+D4 | correct, no change (experience is right) |
|---|---|---|---|---|
| nothing | 43% | 51% | 80% | 57% |
| top 10 heads | 27% | 51% | 98% | 56% |
| **top 20 heads** | **21%** | **61%** | **93%** | **57%** |
| top 40 heads | 14% | 58% | 78% | 53% |
| 20 random heads | 42% | 47% | 78% | 49% |

Paired by set, top 20 heads vs normal: old answers −22% [−30, −14], correct D1+D3 +10% [0, +18],
correct D2+D4 +13% [+4, +23], correct without change 0% [−6, +7]. 20 random heads: old answers −1% [−11, +10].

By condition (top 20 heads): with an exact copy (A) old answers 90% → 60%; without a copy (B) 37% → 3%, correct
60% → 77%; without experience (C) almost unchanged.

## What it means

1. **The pull sits in a small group of heads.** Switching off 20 heads out of 1920 removes half of the old answers,
   while 20 random heads change nothing. So this is not "the model sees worse", but a specific copying mechanism.
2. **For the first time the correct answers themselves went up** (D1+D3 +10%, D2+D4 +13%), not just the old ones
   went away. The dial over all examples did not do this.
3. **Experience where it is right was not harmed**: without a rule change accuracy is the same (57% → 57%), with an
   exact copy 97–100%. The model's knowledge is untouched.
4. **The dose matters.** 40 heads remove even more old answers (14%) but start breaking D2+D4 (78%). 20 looks like a
   reasonable middle.
5. **A "lamp" based on these heads does not work yet.** The signal (how much these heads look at example labels) is
   almost the same whether there is a habit or not (fires in 43% of cases in both A and B, and in 40% where the rule
   did not change). Switching on by this signal gives only −9% old answers. Another signal is needed, for example
   comparing what these heads carry with what the model computed under the current rule.

## Caveats

- One model, one task, 15 held-out sets (wide intervals).
- Heads were switched off only from the test case onward; switching them off on other tasks was not tested. Before
  concluding "this is safe", a check on ordinary tasks (e.g. GSM8K or MMLU) is needed, as in the project plan.
- Condition A still has 60% old answers: an exact copy also pulls through other paths.
