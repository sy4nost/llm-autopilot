# 14B on public tests: normal vs 20 copying heads switched off (2026-10-17)
Qwen2.5-14B-Instruct, H100, bf16. Heads: `top20.json` (layers 29–38, found on our tasks). Script `modal_public.py`, data `public_Qwen2.5-14B-Instruct.jsonl`, scoring `score_public.py` → `score_public_out.txt`.
off_all: heads switched off everywhere; off_ans: only at the answer position.

| test | n | normal | without heads | difference [95%] | answers changed (to correct / to wrong) |
|---|---|---|---|---|---|
| redefine-math | 200 | 69.0 | 67.5 | −1.5 [−6.0, +2.5] | 8 / 11 |
| quote-repetition | 200 | 86.0 | 86.0 | 0 [−2.5, +2.5] | 3 / 3 |
| NeQA | 200 | 61.5 | 59.0 | −2.5 [−7.5, +2.5] | 10 / 15 |
| hindsight-neglect | 200 | 46.0 | 47.5 | +1.5 [−1.0, +4.0] | 5 / 2 |
| GSM-Symbolic main | 50 | 92 | 80 | −12 [−22, −4] | |
| GSM-Symbolic p1 | 50 | 78 | 62 | −16 [−28, −4] | |
On multiple-choice tests off_ans almost coincides with off_all.

## What it means
- On multiple-choice tests the heads barely take part: 3–12% of answers change, equally in both directions. On our tasks the same heads decided the answer (exact copy 93%, without the heads half as many old answers).
- Inference (not measured): the traps in public tests pull from what was memorized in training (a familiar quote, the usual meaning of a symbol, the habit of not noticing "not"), not from copying a ready answer out of the prompt. Our heads are a mechanism for copying from context, so here they have nothing to brake. This agrees with the "native words" experiment: memorized knowledge pulls as strongly as history.
- Switching off the heads during step-by-step math breaks it (−12…−16), as earlier on GSM8K (−6.8): during generation the heads apparently carry the model's own solution steps.
- Cost ~ $1.5–2.
