# Check on independent public tests: untrained model vs "fu" v1 and v2

Qwen2.5-7B-Instruct, Modal L40S, 2026-10-17. The tests are independent and public, and were not in our training.
All of them catch the model on habitual patterns:
- **redefine-math** (Inverse Scaling Prize): a symbol is redefined in the task, e.g. "let π = 462". The habitual value pulls towards the wrong answer.
- **quote-repetition**: a famous quote is modified; the modified one must be repeated, not the memorized one.
- **NeQA**: a question with "not"; the pattern ignores the negation.
- **hindsight-neglect**: 10 examples where the outcome agrees with the reasonableness of the decision, then a case where they disagree.
- **GSM-Symbolic** (Apple): GSM8K problems with other numbers (main) and with 1–2 added conditions (p1, p2). Here the model answers in its own words.

## Results (share correct)
| test | n | untrained | v1 | v2 | v2 − untrained [95%] |
|---|---|---|---|---|---|
| redefine-math | 300 | 70.0% | 64.0% | 69.0% | −1.0 [−7.7, +5.7] |
| quote-repetition | 300 | 84.0% | 83.3% | 83.3% | −0.7 [−3.7, +2.3] |
| NeQA | 300 | 62.0% | 61.3% | 63.0% | +1.0 [−1.7, +4.0] |
| hindsight-neglect | 300 | 41.0% | 33.0% | **28.0%** | **−13.0 [−17, −9]** |
| GSM-Symbolic main | 100 | 91% | 90% | 93% | +2 [−3, +7] |
| GSM-Symbolic p1 (+1 condition) | 100 | 73% | 71% | 71% | −2 [−9, +5] |
| GSM-Symbolic p2 (+2 conditions) | 50 | 52% | 58% | 56% | +4 [−8, +16] |

## Conclusion
1. **No gain on independent tests.** Almost everywhere the difference is within chance. The skill we taught does not transfer to these kinds of traps.
2. **hindsight-neglect got worse (−13).** A guess, not a test: here the right thing is to hold on to the principle from the examples ("judge by the reasonableness of the decision, not by the outcome").
   The model learned to trust examples less and also got worse at holding on to a useful principle.
3. **Why.** We taught a narrow situation: the text has a history of cases and a message that something changed.
   In the independent tests the trap is different: memorized knowledge vs a new definition, a habitual phrase vs a modified one, negation.
   For now the model applies the attitude "check whether it applies now" only to its own kind of situation.
4. This agrees with the method (METOD.md): the skill becomes general only if taught on many different kinds of traps.
   These independent tests must not be used for training; they remain an honest check. Training needs our own tasks similar in meaning:
   redefinitions, modified quotes and texts, negations, examples with a useful principle and with a harmful one.

Files: modal_ext.py, score_ext.py, score_ext_out.txt, ext_Qwen2.5-7B-Instruct.jsonl.
