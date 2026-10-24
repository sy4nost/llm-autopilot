# pull — measuring from outside how hard each force pulls (main run 2026-10-24, < $0.5)

3 models × 900 prompts (300 judge dialogues × FULL / NOEX / NOINS), log P of each force's number as the answer (bare
number or fact sentence). Plan v2 after plan review, sums frozen before main output. Error labels = saved greedy answers.
Coverage (3 numbers hold >= half the answer probability): qwen14 289/300, phi4 266/300, olmo13 287/300.

## Verdict
- **P1 PASS** (sanity): the most probable number = the model's actual answer source in 97 / 100 / 100% of cases.
- **P2 FAIL** (fine lamp "favourite force pulls hard"): pooled error AUC qwen14 0.54, olmo13 0.64 (phi4 not testable,
  7 errors). It loses to both baselines: low confidence (top two numbers close) qwen 0.79, phi 0.98, olmo 0.87; and for
  qwen also to the coarse lamp "answer = favourite" (0.74). Reason: the favourite pulls hardest exactly where it is right
  (qwen instruction kind: favourite log-odds +38, 1 error), so pull strength alone cannot separate right from wrong.
- **P3 FAIL** ("close one eye" adds information): removal-drop lamp below the plain FULL measure (qwen −0.11
  [−0.16, −0.07], olmo −0.01 [−0.04, +0.03]). Alex's doubt confirmed: removing a source tells nothing beyond reading the
  three probabilities once.

## Descriptive (who pulls, per kind; share of dialogues whose top answer changes after removal)
| model | kind | remove examples | remove instruction |
|---|---|---|---|
| qwen14 | instruction | 0 | 100 |
| qwen14 | examples | 58 | 54 |
| qwen14 | truth | 17 | 35 |
| phi4 | instruction | 0 | 100 |
| phi4 | examples | 100 | 0 |
| olmo13 | examples | 97 | 10 |
In the praised-examples kind Qwen sits in a real tug-of-war (either removal flips about half); phi and OLMo are decided
by the examples alone.

## Notes
- Within a kind the fine lamp is near-perfect for qwen (examples 0.99, truth 1.00), but that is close to circular:
  within one kind an error simply means the favourite won. In use the kind is unknown.
- Best lamp here = hesitation: the model's top two numbers are close. Post hoc reading, not a new claim.
- NOEX in the examples kind keeps the praise line with nothing before it to praise.
