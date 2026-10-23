# Exact match or strategy inertia (Alex's test, 2026-10-16)

Question: after a rule change, does 14B answer the old way because it holds on to the old strategy, or because it
finds the same case in the history and copies its answer?

## Setup

- Same rules, same change line (Rule 2: echo=ON → echo=OFF), one case per prompt, 30 new sets.
- History of 20 or 60 examples. Each set has one shared history; the conditions differ **only** in the 15% of
  examples that were solved by the old Rule 2, at the same positions. Length and the number of HOLD/PASS/REVIEW are equal.
  - **A, exact match:** these examples = the test case itself (all 4 fields), with the old-rule answer.
  - **B, structural:** these examples are also solved by the old Rule 2 (waiver=YES, echo=ON) but differ from the
    test in the tag field. Not a single match on all fields.
  - **C, no relevant examples:** instead of them, HOLD by Rule 1. No example uses the cancelled condition.
- A separate program checked the answer keys from the prompt texts (0 errors) and the matches: on D1/D3, A has 9
  exact copies at N=60 (3 at N=20), B and C have 0; examples of the old Rule 2: A 9, B 9, C 0.
- Control without the change line for each condition; baseline: change without history. 1680 prompts × 5 answers,
  Qwen2.5-14B, bfloat16, H100. Files: `make_match.py`, `check_match.py`, `sets_match.json`, `results/`,
  `score_match.py`, output `score_out.txt`.

## Result: old-rule answers on D1+D3 after the change

| history | A exact | B structural | C no relevant | no history |
|---|---|---|---|---|
| 20 | **88%** | 49% | 22% | 37% |
| 60 | **93%** | 38% | 12% | 37% |

Paired differences over 30 sets (95% interval), N=60:
- A − B: **+55%** [+42, +68]
- B − C: +25% [+10, +41]
- B − no history: **0%** [−20, +20]
- C − no history: −25% [−39, −12]

(N=20 goes the same direction: A − B +39%, B − no history +12% [−6, +30].)

## What it means

1. **Almost the whole effect is copying a ready answer.** When the history contains the same case, 14B gives the old
   answer in 93% of cases, although the rule changed. Remove the exact copies while keeping the same type of
   examples, and old answers fall to 38%, exactly as many as with no history at all (37%).
2. **No pure strategy inertia is visible.** Examples solved by the old Rule 2 that do not match the test add no old
   answers above baseline (B − no history = 0% after 60 examples).
3. B is above C (+25%) because C also helps: a long history without the cancelled condition pushes old answers
   below baseline (12% vs 37%).
4. This agrees with the look inside the model: up to layer 32 the model computes by the new rule, and late layers
   pull in the label from a similar example. Now we see that "similar" means "the very same".

Answer to Alex's question: the mechanism is closer to "retrieve the ready answer to an identical question" than to
"get stuck in the previous strategy".

## Caveats

- In B, even without a change, the model applies the old Rule 2 weakly to a case it has not seen (control without
  change, D1+D3: A 100%, B 46%, C 36%). So from the examples it learned "these cases are HOLD" rather than the
  general rule. That means B had little strategy to hold on to; in a model that generalizes better, inertia might
  show up.
- A has more copies than v3 (9 vs about 4–5); this is an upper bound on the effect.
- One model, one wording of the change.

## Possible next steps

- A "lamp" against copying: a hint "is the same case in the examples? the rule changed, do not copy", to see
  whether the 93% goes away.
- Find the attention heads that look from the answer to the exact copy in the history (candidates for label copying).
- The same test on a model from another family.
