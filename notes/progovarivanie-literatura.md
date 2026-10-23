# Thinking aloud against habit errors: who already does this

Date: 2026-10-18. A quick search of papers (I read abstracts and descriptions, not full texts).

## In short

The idea "let the model first reason aloud, then answer" is long known and already built into products. What is new for us is not thinking aloud itself but **how we use it**: as a signal "a habit fired here" and as a way to switch on reasoning only where it is needed.

## 1. Thinking aloud helps: known since 2022

- **Chain-of-thought**: the model is given an example of reasoning or the phrase "let's think step by step", and it solves better. Wei et al. 2022 ([arXiv 2201.11903](https://arxiv.org/abs/2201.11903)), Kojima et al. 2022 ([arXiv 2205.11916](https://arxiv.org/abs/2205.11916)).
- **Specifically on habit traps**: the Inverse Scaling Prize collected tasks where large models err more often than small ones (redefine-math, NeQA, memo-trap etc.; [McKenzie et al. 2023](https://arxiv.org/abs/2306.09479)). Wei, Tay, Le 2022 showed that with chain-of-thought these tasks stop "breaking" with scale ([arXiv 2211.02011](https://arxiv.org/abs/2211.02011)). This is a direct predecessor of our result.
- **Human-style trick questions**: Hagendorff et al. 2023 (Nature Computational Science) gave models a "hasty answer" test (the famous bat-and-ball problem). Older models fell for it like humans; newer ones with reasoning almost stopped ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10766525/)).
- **Counterfactual tasks (a world with a changed rule)**: Wu et al. 2023 "Reasoning or Reciting?" showed that models are pulled towards the familiar variant, and chain-of-thought helps only partly ([arXiv 2307.02477](https://arxiv.org/abs/2307.02477)).
- **Products**: "thinking" models (OpenAI o1/o3, DeepSeek-R1, Claude with extended thinking, Qwen3 with thinking mode) do this by default. This is the industrial version of thinking aloud.

**Analogy**: a student is told "don't write the answer straight away, first say the problem aloud". Teachers have known this trick for a long time, and so have model developers.

## 2. Thinking aloud sometimes hurts: also known

- Liu et al. 2025 (ICML) "Mind Your Step (by Step)": on tasks where thinking long hurts humans (guessing a pattern "by feel", rules with exceptions), reasoning lowers model accuracy to as low as 36 out of 100 ([arXiv 2410.21333](https://arxiv.org/abs/2410.21333)).
- Gema et al. 2025 "Inverse Scaling in Test-Time Compute": the longer the model reasons, the worse on some simple tasks, because it latches onto irrelevant details ([arXiv 2507.14417](https://arxiv.org/abs/2507.14417)).
- **Our case matches**: the "view the number as text, don't compute" items fell from 94 to 24 out of 100 with thinking aloud. The model started "solving" although it was asked not to.

## 3. Disagreement "fast answer ≠ answer after reasoning" as a signal

- Lanham et al. 2023 (Anthropic) compared answers with and without reasoning, but to judge whether the reasoning is faithful, not to catch errors ([arXiv 2307.13702](https://arxiv.org/abs/2307.13702)).
- Self-consistency: ask the model many times and take the most frequent answer; the spread = uncertainty ([Wang et al. 2022](https://arxiv.org/abs/2203.11171)).
- Closest to us: a 2026 preprint "Decomposed Prompting… Helps Models Say 'I Don't Know'". They compare the direct answer with an answer via decomposing the task into steps, and if they disagree the model says "I don't know" ([arXiv 2602.04853](https://arxiv.org/html/2602.04853v1)). Their target is knowledge gaps, not habit. I read only the description.

## 4. "Think only when needed": also done, but for a different reason

- AutoThink ([arXiv 2505.10832](https://arxiv.org/abs/2505.10832)), Thinkless ([arXiv 2505.13379](https://arxiv.org/abs/2505.13379)), Large Hybrid-Reasoning Models: the model itself decides whether to think or answer directly.
- Their criterion: task **difficulty** and saving time. I did not find our criterion ("here I am being pulled by a habitual answer").

## What is new for us (honestly)

1. **Not new**: thinking aloud helps on traps; thinking aloud can be harmful; "thinking" models as a product.
2. **Possibly new** (the search was quick, something may have been missed):
   - The "fast vs reasoned" disagreement as a **detector specifically of habit errors** (for us 90 out of 100 in discrimination quality), the same one for habits from the prompt and from "built-in" knowledge. Recall that our earlier "general detector" on model internals failed (55 out of 100), while this external one worked.
   - The scheme "**lamp → switch on reasoning only there**": a switch based on a sign of habit, not on difficulty. This would solve the harm problem (do not think where thinking hurts).
   - Thinking aloud + the instruction "follow the statement" without harm (96–100 out of 100, "don't compute" items 92). But this was found after the test, on one model, and needs confirmation by a pre-announced test.
3. **Limitations**: one model (Qwen 14B), our self-made tasks; the 0.90 result was obtained after looking at the data.

Note (2026-10-23): the "fast vs reasoned" disagreement is available only after reasoning, so it cannot serve as a switch for whether to reason. Later tests (fastthink, conftrap-lamp with its audit) confirmed that it adds nothing over always reasoning; a pre-reasoning signal remains an open problem.
