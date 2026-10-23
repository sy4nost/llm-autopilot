# HOLD/PASS test by model strength (v2, 2026-10-16)

Goal: measure automatism only on a task the model solves confidently without examples (gate: on average
over 6 cases ≥85% and on each trap case ≥85%).

## What was done

- `make_sets_lvl.py`: 3 rule levels × 30 sets (names, values, labels and order random).
  - **L1**: one HOLD condition (seal=CLOSED). Trap: CLOSED with BLUE.
  - **L2**: two HOLD conditions, written as separate rules (seal=CLOSED; waiver=YES and echo=ON).
  - **L3**: Alex's wording (both conditions in one rule via OR).
  - Versions: none (no examples), exp (20 PASS), mixed (as in Alex's), mixed2 (HOLD also shown via
    waiver+echo), control (letters). Keys checked by a separate program, 0 errors.
- Runs on Modal (Qwen2.5 0.5B–14B, bfloat16, 5 answers per prompt: 1 deterministic + 4 at T=0.7):
  1. six cases in one prompt, no explanations (`results/`, `score_lvl.py`);
  2. format check without examples: "may reason" and "one case at a time" (`results_gate/`, `score_gate.py`);
  3. full "one case at a time" run (`sets_single.json`, `results_single/`, `results_single_14b/`, `score_single.py`).

## Gate without examples (mean over 6 cases)

| model | 6 at once, L1 | reasoning, L1 | one at a time, L1 | one at a time, L2 | one at a time, L3 |
|---|---|---|---|---|---|
| 0.5B | 27% | 16% | 44% | 43% | 38% |
| 1.5B | 41% | 32% | 65% | 58% | 63% |
| 3B | 66% | 59% | **90% ✓** | 75% | 67% |
| 7B | 79% | 76% | **95% ✓** | 61% | 61% |
| 14B | **98% ✓** | 99% ✓ | **99% ✓** | 90%, but P3 70% | 91%, but P3 61% |

Six cases at once without the right to think hurts a lot. Allowing reasoning does not help (3B reasons but gets the
conclusions themselves wrong). One case at a time, 3B, 7B and 14B pass the gate on L1. 0.5B and 1.5B pass nowhere;
automatism cannot be measured on them.

## Main result: at its own level, PASS examples do not mislead

L1, one case at a time, share correct on traps (Q1, Q6 = CLOSED with BLUE → HOLD):

| model | no examples | 20 PASS | mixed | control |
|---|---|---|---|---|
| 3B | 100% / 100% | 100% / 100% | 100% / 100% | 99% / 97% |
| 7B | 100% / 100% | 100% / 100% | 99% / 88% | 100% / 100% |
| 14B | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |

For no model that understands the rule did 20 PASS examples cause a single extra error on a trap.
The only notable drop: 7B in the mixed version on Q6 (−12%, interval −24…−3%).

## Where the models actually err: a two-part condition

P3 (HOLD by waiver=YES and echo=ON), one case at a time:

| model | level | no examples | 20 PASS | mixed | this path shown (mixed2) |
|---|---|---|---|---|---|
| 3B | L2 / L3 | 10% / 6% | 31% / 47% | 20% / 40% | **69% / 67%** |
| 7B | L2 / L3 | 11% / 5% | 8% / 11% | 13% / 0% | 34% / 22% |
| 14B | L2 / L3 | 70% / 61% | 90% / 71% | 71% / 53% | **88% / 89%** |

1. Models hold the condition "A and B" poorly even without any examples (7B: 5–11%). This is not automatism but lack of understanding.
2. If the examples show how this condition fires, the models learn: 3B from 6–10% to 67–69%, 14B to 88–89%.
   7B learns less (22–34%).
3. The earlier result "7B: 11% after PASS examples vs 33% without examples" did not repeat in the "one case at a time"
   format (8% vs 11%, within noise). It was tied to the "six at once" format.

## Conclusion

On a task the model solves confidently, 20 examples of one answer cause no automatism in 3B, 7B or 14B. This agrees
with v2 (where habit also caused no errors). Errors appear where the model does not understand the rule, and there
examples help rather than hurt.

## What could be tried next

- A longer history: 64–200 examples of one answer instead of 20 (in v2 the confidence effect in 7B grew with length).
- A trap the model solves, but with difficulty: e.g. L2 for 14B (90% on average, P3 70%). There is room to fall, and it
  is already visible that mixed examples without showing the second condition pull P3 down (L3: 61% → 53%).
- Measure not only the answer but also confidence (like H1 in v2): automatism may show in it earlier than in errors.
