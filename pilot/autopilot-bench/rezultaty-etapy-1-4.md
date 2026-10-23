# Cognitive Autopilot Benchmark: stages 1–4 (2026-10-16)

10 task pairs by Alex. Each version was given in a separate new conversation, in one message. Models Qwen2.5-Instruct
0.5B–14B, bfloat16, greedy, one answer per task. Scored by me against the key. ✓* (correct answer with wrong
reasoning) is counted as correct.

- Stages 1–2: Alex's original texts (`results/`, details in `rezultaty-etap-1-2.md`).
- Stage 3: the same tasks in other words, same logic and keys (`tests_rephrased.json`, `results_stage3/`).
- Stage 4: original texts + 2 irrelevant sentences after the first line (`tests_distractor.json`, `results_stage4/`).

## Summary by model

| model | correct of 20: original | rephrased | with noise | both versions correct (pairs of 10): orig / reph / noise | correct in all three runs (of 20) | wrong in all three (of 20) |
|---|---|---|---|---|---|---|
| 0.5B | 12 | 9 | 7 | 4 / 2 / 1 | 2 | 3 |
| 1.5B | 11 | 12 | 8 | 2 / 2 / 1 | 6 | 5 |
| 3B | 13 | 11 | 13 | 4 / 2 / 3 | 9 | 4 |
| 7B | 13 | 16 | 16 | 4 / 6 / 6 | 11 | 2 |
| 14B | 17 | 16 | 16 | 7 / 6 / 7 | 14 | 2 |

Reading: the small models (0.5B, 1.5B) mostly guess: a change of words or noise changes the answer, and only 2–6 of 20
are correct in all three runs. 7B and 14B hold up: 11 and 14 of 20 are stably correct. Noise hits 0.5B and 1.5B the
hardest (minus 3–5 correct) and barely affects 7B and 14B.

## By task: times correct out of 15 (5 models × 3 runs)

| test | A | B | what is visible |
|---|---|---|---|
| 1 Silent Override | 7 | 11 | the "low-risk" exception is often missed in A |
| 2 Delayed Signal | 8 | 10 | the signal delay is taken into account every other time |
| 3 Definition Shift (control) | 13 | 13 | simple arithmetic, almost always correct |
| 4 Missing Participant | 7 | 7 | the dependency "Ben only with Anna" is lost; 14B with noise failed both versions |
| 5 Unchanged Outcome | 14 | **0** | nobody ever noticed the contradiction in the statement of B |
| 6 Information Barrier | 14 | 13 | easy |
| 7 Hidden Dependency | 8 | 14 | in A the small models do not see the hidden dependency |
| 8 Selection Bias | 12 | 7 | the different evaluation criteria in B are noticed only by 7B and 14B |
| 9 Priority Reversal | 11 | **2** | the word "advisory" (optional) is ignored; only 7B correct, twice |
| 10 Final Constraint | 10 | 9 | in A often "yes" with a loading that breaks the limit; in the rephrased B the late condition is forgotten |

## Main points

1. **The cleanest automatisms: test 5 B (0 of 15) and test 9 B (2 of 15).** This does not depend on model size,
   wording or noise. The model sees a familiar scheme ("one system failed, the other coped"; "a priority rule") and
   answers by the scheme without checking that the statement breaks it. These two pairs are best suited as the core
   of a future "lamp" test.
2. **Size helps with stability, but not with automatism.** 14B is stably correct on 14 of 20, but fails tests 5 B and
   9 B just like 0.5B.
3. **A late condition gets lost in rephrasing.** Test 10 B was solved by all in stage 2, but in the other wording
   almost all forgot the constraint at the end. So in stage 2 the models solved it not by logic but thanks to a lucky
   form of the text.
4. **"Yes" with an impossible solution.** In test 10 A and test 4 the models often give the correct "yes" but propose
   a loading or a group that violates the condition. One must check not only the answer but the solution itself.

## Caveats

- One answer per task, 10 pairs: these are observations, not statistics. Conclusions need several answers per task
  (with temperature) and 30–50 pairs.
- I wrote the stage-3 rephrasings; they may have accidentally become easier or harder than the original.
- Test 2 allows a dispute about the meaning of "recorded at"; I scored as in the key.
- Borderline stage-4 scores: 0.5B 2B (a weak "probably yes", counted), 0.5B 4A and 8B (✓*), 1.5B 2B and 8B (not counted:
  evasive and the wrong reason).
