# Alex's test with HOLD/PASS/REVIEW rules: results (2026-10-16)

Alex's source: `original/` (3 prompts, README, key). Generator: `make_sets.py` produced 30 sets (`sets.json`).
Set 0 matches Alex's prompts exactly. Sets 1–29 have the same logic but different field names, values,
labels (STOP/GO/CHECK etc.), order of fields, examples and test cases. Keys checked by a separate program,
0 errors. The control is matched in length (2579 characters on average vs 2625 for the "20 PASS" version).

Prompt versions: **exp** (20 PASS examples), **mixed** (10 PASS + 5 HOLD via the first field + 5 REVIEW),
**control** (20 examples about letters), **none** (added by me: only the rules, no examples).
Models Qwen2.5-Instruct 0.5B–14B, bfloat16, Modal L40S. Per prompt 1 deterministic answer and 4 at T=0.7,
150 answers per model and version in total. Scoring: `score.py`, `stats.py`. Answers: `results/`.

## Share of correct answers on P3 (HOLD by the second condition: waiver=YES and echo=ON with tag=BLUE)

| model | exp | mixed | control | none |
|---|---|---|---|---|
| 0.5B | 9% | 17% | 10% | 20% |
| 1.5B | 8% | 7% | 6% | 21% |
| 3B | 55% | 39% | 28% | 61% |
| 7B | **11%** | **9%** | 32% | 33% |
| 14B | 79% | 59% | 66% | 62% |

P1 (HOLD by the first condition, seal=CLOSED) is barely affected: 7B 100% in exp, 14B 93–99% everywhere.
The P5 trap (an extra HOLD) almost never fires for 7B and 14B (0–4%).

## Differences between versions on P3, paired over 30 sets (95% interval, bootstrap)

| model | exp − none | mixed − none | exp − control | exp − mixed |
|---|---|---|---|---|
| 3B | −5% [−20, +9] | −22% [−39, −4] | +27% [+12, +44] | +17% [+2, +31] |
| 7B | **−22% [−37, −9]** | **−25% [−39, −11]** | **−21% [−35, −9]** | +3% [−8, +13] |
| 14B | +17% [+5, +31] | −3% [−23, +17] | +13% [−4, +30] | +21% [+5, +35] |

## What is visible

1. **7B shows an effect, and it is stable.** After 20 PASS examples the model answers PASS on P3 in 89% of cases,
   although by rule 1 it is HOLD there. Without examples it is correct three times as often (33% vs 11%). The
   difference holds across 30 different sets, so it is not about the words of one text. On Alex's original prompt
   (set 0) 7B also answered PASS on P3 in the exp version and HOLD in the version without examples.
2. **But this is not a habit for PASS.** In the mixed version (10 PASS, 5 HOLD, 5 REVIEW) 7B is just as bad on P3 (9%).
   exp and mixed have one thing in common: in no example does HOLD fire by the second condition (waiver+echo).
   It seems the model infers from the examples which parts of the rule are "alive" and silently drops the part that
   was never applied. This is an automatism, but a different one: not "repeat the last answer" but "the rule is as
   the examples showed it".
3. **14B behaves the other way round:** PASS examples help it (79% vs 62% without examples). The mixed version
   helps less. No explanation yet; this is an observation.
4. **Small models (0.5B, 1.5B) do not apply the two-condition rule at all,** with any examples:
   P3 correct in 6–21% of cases. For them the test does not distinguish versions.
5. **The difficulty is in the rule itself.** Even without examples P3 is solved rarely (21–62%), P1 much more often.
   Models hold the condition "A and B" inside an "OR" worse than a single condition.

## Caveats

- The letters control breaks the format: 7B often answers with labels without "P1:"; such answers were scored by
  line order. The none version is cleaner as a baseline.
- One history size (20 examples) and one temperature. I did not separately test "drops the never-shown condition";
  it is easy to set up: a mixed version where HOLD is also shown via waiver+echo.

(Follow-up in v2/: with one case per prompt, the 7B effect mostly disappeared; the earlier 11% vs 33% was a format effect of six cases at once.)
