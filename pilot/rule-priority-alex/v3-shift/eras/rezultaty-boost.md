# Making the new rule louder (14B, 2026-10-17)
A mirror of Alex's formula: from the question onward, log(k) is added to the attention weight on the line "IMPORTANT UPDATE …". Variants: in the 40 copying heads (b40), in all heads of layers 29–47 (late), 14–28 (mid) or in all layers (all). Plus a variant without intervention: the same line repeated right before the question (_rep). The same 240 prompts from sets_eras.json (sets 15–29) + 120 with the repetition.
Files: modal_boost.py, boost_Qwen2.5-14B-Instruct.jsonl, score_boost.py → score_boost_out.txt. Cost ≈ $1.
Checks: k = 1 gives exactly the normal model (difference 0.000 on 300 prompts); line boundaries printed in the first jsonl line; normal mode and w0.1 match the earlier eras run.

## Correct answers on traps (D1+D3, the old rule leads to an error)
| | normal | louder, late layers ×8 | louder, mid ×4 | louder, all layers ×4 | all ×16 | quieter old (w0.1) |
|---|---|---|---|---|---|---|
| A60 (only old examples) | 10% | 13% | 27% | 30% (+20 [+10,+33]) | 37% (+27 [+13,+40]) | 13% |
| E_nocopy (fresh examples, but not this case) | 7% | 7% | 20% | 23% (+17 [+7,+30]) | 30% (+23 [+10,+37]) | 17% |
| E_copy (a fresh copy exists) | 70% | 60% | 77% | 77% | 60% | 90% |
| line repeated before the question, A60 | 10% | 17% | 23% | 30% | 40% | 27% |
Cases where the new rule adds a "stop" (D2+D4): 87–90% → 100% in almost all variants.
Old answers in A60 with "all ×16": 90% → 37%, but only 37% correct: the remaining 26% go to a third, also wrong, label.

## What it means
1. The new rule has to be "heard" in the middle layers (14–28), where the model infers the rule from the examples. Boosting in late layers and in the copying heads gives nothing: it is too late there.
2. A louder new rule helps where "quieter old" is powerless (no fresh example): 7–10% → 23–37%. Where a fresh example exists, "quieter old" works better (90%), and a too-loud new rule (×16) gets in the way (60%).
3. Simply repeating the line before the question gives nothing by itself (10% → 10%), only together with an intervention.
4. The ceiling so far is about a third correct on hard traps: the habit can be broken (old 90% → 37%), but the correct answer does not always appear.
5. Harm: the intervention touches only the change line, so where there is no change it does nothing by construction. MMLU and GSM8K were not checked with it.

## Addendum: does the model have the correct answer at all (on saved answers, no new runs)
- Only the new rule, no history ("fresh", match/results): D1+D3 correct 26/30. The model can solve these cases.
- Old rule + change line, no examples ("change_0"): 13/30. The "was — now" amendment confuses it even without habit. This is the ceiling a brake can bring it to.
- In normal mode the correct answer is the second strongest in 22/30 traps (A60).
- Our interventions give 30–37% with a ceiling of about 43%: the habit effect from examples is almost fully removed; the remainder is a weakness in applying an amendment to a rule.
- A caveat about tuning: one task, one model; the intervention strength was chosen after looking at several variants. To test generality: fix ×4 in advance and run on another task type and another model.
