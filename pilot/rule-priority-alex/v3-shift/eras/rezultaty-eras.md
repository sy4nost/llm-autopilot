# Alex's formula: "vote strength = case similarity × sameness of conditions" (14B, 2026-10-17)
Qwen2.5-14B-Instruct, H100. 40 copying heads (b40). From the question onward, log(w) is added to the attention weight on PERIOD 1 examples (solved under the old rule); PERIOD 2 examples and everything else are untouched. w = 1 is the normal model, w = 0 is full closing. Sets 15–29 (heads were not selected on them). Cost < $1.
Files: make_eras.py → sets_eras.json, check_eras.py, modal_eras.py, score_eras.py → score_eras_out.txt, eras_Qwen2.5-14B-Instruct.jsonl.

## Correctness checks
- Data: check_eras.py re-derives each of the 70 labels in each of the 240 prompts from the rule text in the prompt itself (not through the generator code): 0 errors. If one example is corrupted, the check catches it.
- w = 1 gives exactly the normal model: maximum difference 0.000 on all 240.
- A60 with w = 0 repeats the earlier b40 silencing run: 60 of 60 identical answers, difference 0.000.
- Period boundaries are printed in words (first jsonl line): period 1 = "Below are 60 solved … -> FREEZE", question = "Classify the TEST case".
- Independent reviewer: no serious errors. Two remarks fixed: period 2 of E_copy and E_nocopy is now identical except for one position, and the copy replacement is a neutral case.

## Result (correct answers; D1+D3 are the cases where the old rule leads to an error)
| test | normal | w 0.3 | w 0.1 | w 0.03 | w 0 |
|---|---|---|---|---|---|
| E_copy: after the change there is an example of exactly this case under the new rule | 70% | 87% | **90%** | 93% | 93% |
| E_nocopy: 10 examples after the change, but not this case | 7% | 10% | 17% | 20% | 23% |
| A60: only old examples | 10% | 10% | 13% | 13% | 27% |
| N_same: rule unchanged (formula = 1) | 100% | 100% | 100% | 100% | 100% |
w 0.1 vs normal: E_copy +20% [+7, +37]; E_nocopy +10% [0, +20]. Old answers in E_copy 30% → 10%, in E_nocopy 93% → 80%.

## What it means
1. The formula works where the model has fresh experience under the new conditions: it starts relying on it instead of the old copy (70% → 90%).
2. If there is no fresh experience for exactly this case, the formula helps little (7% → 17%). The habit does not rest only on the copying heads: a rule learned from 60 examples sits deeper (layers below 29), and the voice of the new change line is weaker.
3. When conditions did not change there is no harm: even if old examples are weakened by mistake, answers are not spoiled (100%), because the same cases are also present in period 2.
