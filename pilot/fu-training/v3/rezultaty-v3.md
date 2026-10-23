# Decisive experiment v3: "fu" on 8 kinds of traps, result (2026-10-17)

Qwen2.5-7B-Instruct, DPO + LoRA, 2791 pairs from 8 families (make_fu3.py). Threshold written in advance (PREREG.md).

## Our own test (test3.json)
Traps: rule change 7→78%, plain 4→89%, noisy 12→88%, story change 37→100%; false notices: format 0→100%, redefine 68→95%.
Many new families (altered, schedule, most of redefine) were already solved at 100% by the base model, so they were hardly traps.

## Public tests (base → v3, paired bootstrap, 95%)
| test | type | n | base | v3 | difference |
|---|---|---|---|---|---|
| redefine-math | near | 300 | 70.0 | 66.3 | −3.7 [−11.7, +4.3] |
| quote-repetition | near | 300 | 84.0 | 93.0 | **+9.0 [+5.3, +13.0]** |
| GSM-Symbolic p1 | near | 100 | 73 | 72 | −1 [−11, +8] |
| GSM-Symbolic p2 | near | 50 | 52 | 46 | −6 [−22, +10] |
| NeQA | far | 300 | 62.0 | 69.7 | **+7.7 [+3.3, +12.3]** |
| hindsight-neglect | far | 300 | 41.0 | 12.0 | **−29.0 [−34.0, −23.7]** |
| GSM-Symbolic main | control | 100 | 91 | 80 | **−11 [−18, −4]** |
| MMLU | control | 570 | 75.1 | 74.9 | −0.2 |
| GSM8K | control | 250 | 90.4 | 84.4 | **−6.0** |

## Threshold check
- ≥ +5 on at least 2 tests with an interval excluding 0: **yes** (quote +9, NeQA +7.7).
- No test drops with its interval entirely below −3: **no** (hindsight −29, GSM-Sym main −11).
- MMLU and GSM8K drop by no more than 3: **no** (GSM8K −6).
- Success on a far test: NeQA yes, but hindsight collapsed.

**Verdict: threshold not met.** The "fu" line (DPO on our own traps) is closed as a negative result.
For the first time real transfer appeared (quote, NeQA), but it costs too much: the model got worse at arithmetic (GSM8K, GSM-Sym main), and on hindsight it stopped judging decisions by expected value.
The first guess ("the model learned to trust the past less") is checked below and on the whole was not supported.

## Checking the guess (probe_v3.py, on saved answers, no new runs)
| test | answers changed | to correct | to wrong |
|---|---|---|---|
| redefine-math | 149 of 300 | 69 | 80 |
| quote-repetition | 35 | 31 | 4 |
| NeQA | 45 | 34 | 11 |
| hindsight-neglect | 87 | 0 | 87 |
| GSM-Sym main | 15 | 2 | 13 |
- hindsight: in these tasks the examples teach judging by the outcome, while the right way is to judge by expected value. v3 errs more often, i.e. it follows the examples more than the base model does. This contradicts the guess "trusts the past less".
- redefine: v3 changed half of its answers, roughly equally in both directions (base chose option 1 in 220/300, v3 chooses option 0 in 165/300). This is a shift in option preference, not understanding.
- Our own test: false notices 95–100%, i.e. in our own formats the model distinguishes when experience applies.
- Conclusion from the data: the skill was learned in our formats. In other formats training mostly destabilizes the choice: sometimes by chance for the better (quote, NeQA), sometimes for the worse. No general "attitude to experience" appeared.

Data: ../fu_Qwen2.5-7B-Instruct_v3.jsonl, ../external/ext_Qwen2.5-7B-Instruct_v3.jsonl, ../external/score_ext_v3_out.txt.
