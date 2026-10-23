# "First solve it yourself, then check against experience" on 14B (2026-10-16)

Alex's idea: the model first solves the task without experience, then looks at experience and decides whether it is needed.

## Setup

- The same 30 sets, histories and test cases as in the A/B/C test (`make_match.py`), history 20 and 60.
- Pass 1 (draft): rules (+ change line) and the test case, no examples. The greedy answer is taken.
- Pass 2 (check): rules, the whole history of condition A/B/C, the change line, the test case and
  "Your first answer, made only from the rules without examples, was: X. Now look at the solved examples and
  decide whether past experience should change this answer. Check carefully. Return only the final label."
  The wording is neutral: it does not say which matters more, rules or examples.
- The same no-change control. Compared with the normal single pass from the A/B/C test.
  1440 check prompts × 5 answers, H100. Files: `make_twopass.py`, `sets_twopass.json`, `modal_twopass.py`,
  `results_twopass/`, `score_twopass.py`, output `score_twopass_out.txt`.

## Result (after the rule change)

| condition, 60 examples | D1+D3 old answers: 1 pass → 2 passes | D1–D4 correct: 1 pass → 2 passes |
|---|---|---|
| A exact copy | 93% → **49%** | 48% → 20% |
| B no copy | 38% → 13% | 65% → 41% |
| C no relevant examples | 12% → 6% | 71% → 42% |

The same with 20 examples. All differences in "correct" are negative, from −14% to −30%, with intervals not touching zero.

## What happened

1. **Old answers on D1/D3 became fewer** (in A 93% → 49%). But not because the model solved them correctly.
2. **The check almost always changes the draft.** After the rule change the model kept its draft in only 2–20% of
   cases. It reads the question "should experience change the answer" as "change the answer".
3. **On D2/D4 this is a failure**: the draft was correct in 80–83%, after the check 0–10% correct, and 73–86% of
   answers became the old answer from the examples. The D2/D4 history has exact copies labelled by the old rule, and
   the check rewrote the correct answer into the copied one. This is the same "copy the precedent" mechanism, only we
   invoked it ourselves.
4. The draft without experience is also weak: D1 23% correct, D3 47% (the "no history" level from earlier tests).
   So "solve it as new" does not mean "solve it correctly" for this model: it applies the change line poorly even
   without experience.
5. 248 of 7200 answers were cut at 20 tokens before the label (the model started explaining); this does not affect the conclusion.

## Conclusion

The idea in this simple form (as a hint in the text) does not work: overall accuracy falls by 14–30%. The internal
version, where we closed the model's view of the examples at answer time, did work (83% → 37% old answers), but it
only returns to the "no experience" level, not to the correct answer.

The "then check against experience" step turned out to be the most dangerous place: when checking, a weak model does
not weigh experience but obeys it, or the request to check itself.

## Possible next steps

- Make the check not "will experience change the answer" but specific: "find the same case in the examples; does its
  label match the current rules? if not, the example is stale". This is checking experience by the rule, not the rule
  by experience.
- A variant with the reverse order: experience may be used only if the draft is unsure (by probability).
- Try the same on a model stronger than 14B, where the draft without experience is already closer to 90%.
