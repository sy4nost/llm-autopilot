# Caption for the examples: "correct solutions" or just "past cases" (14B, 2026-10-16)

Alex's question: maybe the model copies past answers because we called them correct?

## Setup

The same 30 sets, a history of 60 examples, conditions A (exact copy of the test in history), B (no copy), C (no
examples of the cancelled condition), one pass, with and without a rule change. Only the caption of the examples changes:
- 0: "Below are 60 solved examples. They show cases and their correct classifications." (as before)
- 1: "Below are 60 past cases and the labels they were given at the time." (without the word "correct")
- 2: as 1, plus "whether each label was correct under the rules in force at that time", and each example is marked
  "(correct under the rules in force at that time)".
540 prompts per caption × 5 answers, H100. Files: `make_neutral.py`, `sets_neutral1.json`, `sets_outcome.json`,
`results_neutral/`, `results_outcome/`, `score_neutral.py`, output `score_neutral_out.txt`. Caption 0 is taken from
the A/B/C run (`results/`).

## Result: old-rule answers on D1+D3 after the change

| condition | 0 "correct" | 1 "given at the time" | 2 "correct at that time" |
|---|---|---|---|
| A exact copy | 93% | 95% | 94% |
| B no copy | 38% | 36% | 34% |
| C no relevant examples | 12% | 12% | 8% |

All differences from caption 0 are within ±4%, and the intervals cover zero. The share of correct answers on D1–D4
also barely changes (A 48/47/49%, B 65/62/71%, C 71/74/78%).

## Conclusion

The caption does not matter. The model copies the answer of an identical past case regardless of whether it is said
to be correct, and even when it is said outright that it was correct only under the rules of that time. The copying
comes not from the model "believing" the examples but from the case matching a past one. This is closer to the
"native language" in Alex's example: the reaction to a familiar case fires before the model weighs the words.

A small shift appears only for caption 2 in the no-change control, condition B (58% vs 46% correct under the old
rule, interval −1…+24%): the "correct" mark slightly helps apply the old rule to a new case. This is at the edge of noise.
