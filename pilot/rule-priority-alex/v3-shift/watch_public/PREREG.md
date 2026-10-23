# Watchman on independent public tests, written in advance (2026-10-17, before the run)

*English translation. The original text, frozen before the run, is in the git history (commit 6d3f172).*

The watchman does not change: the same idea and the same brake as in ../watch/ (re-solve each example without history, silence the stale ones in the 40 heads of b40).
Only one thing changed for the format of the public tests. During re-solving, the answer is chosen among options (the test's classes plus all recorded example answers) by the same scoring used for the test, because in chat mode the model otherwise starts explaining.

## Tests (built by others, we have no influence on them)

- **hindsight-neglect-10shot** (Inverse Scaling, all 315). 10 solved examples, then a case where the outcome of a game contradicts the reasonableness of the decision.
- **repetitive-algebra** (Inverse Scaling, round 2, all 1000, copy on HF: Albertmade/repetitive-algebra). Many examples with one answer, one with another, then a question.
- **MMLU** (cais/mmlu): 500 questions, sampled with Alex's number 185 (named earlier, not by me), 5 examples from the same subject. Harm check.

## Prediction written before the run

I expect no gain. In these tests the examples are not "stale": they were made under the same conditions, and the trap is different (a misleading pattern in hindsight, the last example in algebra). The watchman was built for something else.

## Criteria

- **No harm on MMLU:** watch_b40 − normal ≥ −2 points.
- **hindsight and algebra:** "benefit shown" only if the lower bound of the 95% interval is > 0. Harm is noted separately.
- watch_all is report-only.

## Checksums

- `modal_pub_watch.py` 4bc1cd894fa88b31f3b24c79e25a6e9bbb7f5ca12f88f41867e11fa6fe9aae4f
- `score_pub_watch.py` a3b9fd2b8dd040614a3e673e9cd9bf36de43dd239b104b6eeec1e59b59cc1168
- `../eras/settings2.json` 5a03d3037745bf96e45abbf877c5cc81a1bb2a2b9a1cbe77bc33e5fec14db6c0
