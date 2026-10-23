# Cognitive Autopilot Benchmark: stages 1–2 (2026-10-16)

10 task pairs by Alex (`tests.json`). Each version A and B was given to the models in a separate new conversation, in
one message, without hints. Models Qwen2.5-Instruct 0.5B, 1.5B, 3B, 7B, 14B, bfloat16, deterministic answers
(greedy), one answer per task. Full answers: `results/<model>.json`. I scored by Alex's key.

✓ correct, ✗ wrong, ✓* answer correct but reasoning wrong (a lucky guess).

| test | 0.5B A/B | 1.5B A/B | 3B A/B | 7B A/B | 14B A/B |
|---|---|---|---|---|---|
| 1 Silent Override | ✓ ✗ | ✓ ✗ | ✗ ✓ | ✗ ✓ | ✓ ✓ |
| 2 Delayed Signal | ✗ ✓ | ✓ ✓ | ✓ ✗ | ✗ ✓ | ✓ ✓ |
| 3 Definition Shift (control) | ✗ ✗ | ✓ ✓ | ✓ ✓ | ✓ ✓ | ✓ ✓ |
| 4 Missing Participant | ✓ ✓* | ✗ ✓ | ✗ ✗ | ✗ ✗ | ✓ ✗ |
| 5 Unchanged Outcome | ✓ ✗ | ✓ ✗ | ✓ ✗ | ✓ ✗ | ✓ ✗ |
| 6 Information Barrier | ✓ ✓ | ✓ ✗ | ✓ ✓ | ✓ ✓ | ✓ ✓ |
| 7 Hidden Dependency | ✓ ✓* | ✗ ✓ | ✓ ✓ | ✓ ✓ | ✓ ✓ |
| 8 Selection Bias | ✗ ✓ | ✓ ✗ | ✓ ✗ | ✓ ✗ | ✓ ✓ |
| 9 Priority Reversal | ✗ ✗ | ✗ ✗ | ✓ ✗ | ✓ ✓ | ✓ ✗ |
| 10 Final Constraint | ✓* ✓ | ✗ ✓ | ✓* ✓ | ✗ ✓ | ✓* ✓ |
| **correct out of 20** | 12 | 11 | 13 | 13 | 17 |
| **pairs with both versions correct** | 4 | 2 | 4 | 4 | 7 |

## Where the models are "on autopilot"

1. **Test 5 B: nobody noticed the contradiction in the statement.** The rule says "both systems are needed", yet the
   plant did not overheat with only one. All five models calmly answered "yes"; 7B even wrote "B alone handled it"
   and did not see that this breaks the rule. This is the cleanest example of automatism in the set.
2. **Test 1: the exception is not noticed.** 3B and 7B in version A applied the general rule "both are needed" and did
   not see the exception for low-risk. 0.5B and 1.5B, conversely, answered "yes" in both versions: they did not notice
   who exactly may be skipped.
3. **Test 4 B: a dependency is forgotten.** 3B, 7B and 14B formed the group "Ben, Daniel, Emma" (or similar) and forgot
   that Ben cannot go without Anna. 7B and 3B also failed version A.
4. **Test 9 B: the word "advisory" is not noticed.** 3B and 14B answered "B first", as if the rule were mandatory.
   Only 7B took into account that the rule is optional.
5. **Test 8 B: only 14B mentioned the different evaluation criteria.** The others either computed a z-test or
   referred to self-selection, but accepted the comparison of 90% and 70%.
6. **Test 10 A: "yes", but with an impossible loading.** 3B and 14B correctly answered "yes" but proposed a loading
   that exceeds truck B's limit (550 and 600 kg with a 500 kg limit). 7B got lost in the enumeration.

## Caveats

- One answer per task and 20 tasks per model: these are observations, not statistics. For reliability, several
  answers per task (with sampling) and more pairs are needed.
- Test 2 allows a dispute about what "recorded at 09:14" means. I scored as in the key: event = 09:14 minus the delay.
