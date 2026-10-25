# goal — model states the user's current wish first, then answers (main run 2026-10-24, < $1)

3 models × 300 dialogues (../judge direct). Step 1: "say what the user wants from the next answer" (no answering).
Step 2 arms: GOAL (own goal sentence as system note), PLACEBO (neutral note of similar length), DIRECT re-run.
Plan v2 after plan review, sums frozen before main output. DIRECT re-run = saved judge DIRECT in 99.3–100%.

## Verdict
| | qwen14 | phi4 | olmo13 |
|---|---|---|---|
| G1 weak kind, GOAL − PLACEBO (primary) | examples +11 [0, +22] FAIL (edge) | truth −3 FAIL (low power) | truth +3 FAIL |
| G1b weak kind, GOAL − DIRECT | +26 [+14, +37] PASS | +3 FAIL | +1 FAIL |
| overall GOAL − PLACEBO | **+14 [+10, +18]** | −14 [−17, −10] | +8 [+4, +13] |
| G2 harm (breaks DIRECT-correct) | 3.1% PASS | 14.3% FAIL | 7.0% FAIL |
| G3 lamp "quick ≠ with goal" | AUC 0.92 vs low conf 0.78, **+0.14 [+0.08, +0.20] PASS** | not testable (7 errors) | 0.88 vs 0.87 FAIL |

## Per kind, accuracy % DIRECT / PLACEBO / GOAL
| model | instruction | praised examples | "tell the truth" |
|---|---|---|---|
| qwen14 | 99 / 100 / 100 | 58 / 73 / 84 | 69 / 68 / **99** |
| phi4 | 100 / 100 / 100 | 100 / 100 / **62** | 93 / 99 / 96 |
| olmo13 | 100 / 100 / 100 | 87 / 63 / 85 | 85 / 83 / 86 |

## What it means
- **Qwen: the goal step works.** Asked "what does the user want?", Qwen mostly reads it right (praised examples: its
  goal names the examples number 82 of 100), then follows its own goal (copy rate 98%). Truth kind 69 -> 99 while the
  placebo does nothing (68). The lamp "quick answer differs from answer with goal" catches 64 of 74 Qwen errors with
  3 false alarms in 226 correct answers - better than hesitation (low confidence).
- **phi: the goal step carries the same bias as the judge.** In the praised-examples kind phi's goal sentence names the
  first-turn instruction 41 of 100, and phi then follows it: 100 -> 62. Its quick answers were already right.
- **OLMo:** no gain on its weak kind; goal sentences often name two numbers (mixed 25–43%); the lamp equals low confidence.
- Note: in the truth kind the goal often names the old number as the thing to stop ("rather than continuing with 13"),
  so "goal right" there is underestimated by the number parser; answers themselves are scored normally.
- Qwen G1 primary misses by a hair (CI lower bound 0.0); the placebo line alone already lifts examples-kind 58 -> 73
  (format effect, seen before in the cycle test).
