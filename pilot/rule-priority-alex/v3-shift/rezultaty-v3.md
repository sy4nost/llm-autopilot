# 14B: will a new rule override a habit? (v3, 2026-10-16)

Alex's idea: 14B learned a two-condition rule (88–89% in v2). Give it a habit under this rule, then change the rule
in one place and see what it does.

## Setup

- Old rule: HOLD if seal=CLOSED; HOLD if waiver=YES and echo=ON; PASS if tag=BLUE; otherwise REVIEW.
- After the examples, one line: "IMPORTANT UPDATE: Rule 2 has been changed. New RULE 2: ... waiver=YES and echo=OFF".
  One word changes (ON to OFF). The change is announced openly, not hidden.
- History: 0, 20 or 60 solved examples under the old rule (about 15% of them HOLD via waiver+echo=ON).
- Control: the same history without the change line (the old answer is correct). Gate: the new rule given directly
  in the prompt, no history.
- 30 sets with random names, one case per prompt, 5 answers per prompt (Qwen2.5-14B, bfloat16, H100).
  Answer keys checked by a separate program, 0 errors. Files: `make_shift.py`, `sets_shift.json`, `results/`, `score_shift.py`.

Cases where the old and new rule disagree:
- D1, D3: waiver=YES, echo=ON. Old: HOLD. New: not HOLD (PASS or REVIEW).
- D2, D4: waiver=YES, echo=OFF. Old: not HOLD. New: HOLD.

## Result (accuracy under the current rule)

| | D1 | D3 | D2 | D4 |
|---|---|---|---|---|
| gate (new rule directly) | 93% | 73% | 55% | 73% |
| change after 0 examples | 27% | 61% | 64% | 75% |
| change after 20 examples | 21% | 22% | 83% | 100% |
| change after 60 examples | **11%** | **17%** | 91% | 87% |
| no change, 60 examples (old rule) | 100% | 97% | 97% | 97% |

Answers following the old rule after the change:

| | 0 examples | 20 | 60 | difference 60 − 0 (95% interval) |
|---|---|---|---|---|
| D1, D3 (old HOLD → new "not HOLD") | 26% | 74% | **82%** | **+56% [+42, +70]** |
| D2, D4 (old "not HOLD" → new HOLD) | 14% | 8% | 11% | −3% [−15, +9] |

## What it means

1. **The automatism is there, and it is strong.** After 60 examples, 14B keeps answering HOLD in 82% of the cases
   where the old rule gave HOLD, although the line says outright that the rule changed. Without history the error
   rate is 26%. The effect grows with history length (26% → 74% → 82%) and holds across 30 different sets.
2. **It adds the new rule but does not let go of the old one.** Where the new rule introduces HOLD (D2, D4), 14B
   applies it, even better with history (up to 87–91%). So it does not ignore the update: it accepts it, but on top
   of the old rule. As a result it answers HOLD for waiver=YES almost always, with both ON and OFF.
3. This is the first effect in the project that is visible directly in the answers of a large model, not only in
   its confidence.

## Caveats

- The gate under the new rule is below 85%: 73% on average (D2 only 55%). But for the main case D1 the gate is 93%
  and the drop there goes to 11%, so the effect is not explained by the new rule being hard.
- The change is announced openly. A version where the new rule quietly sits in the text without the word "changed"
  (a test of "will it notice") has not been done yet.
- One model (14B), one wording of the change line.

## Possible next steps

- A silent change (no "IMPORTANT UPDATE"): will it notice at all?
- The same design on 7B (and maybe on models from another family), to see whether the effect is general.
- A "lamp": a hint before the answer such as "check whether the rule has changed", to see whether the 82% goes away.

## Addendum: does it see the change or not? (v3b)

Alex's question: did the model not see the change, or did it see it and still act by habit? The same 30 sets and
histories (0/20/60), two new versions (`make_restate.py`, `sets_restate.json`, `results_restate/`, `score_restate.py`):
- "restate the current rule 2" (no task);
- "first restate rule 2, then give the answer".

| history | restated the new rule correctly | D1+D3: old-rule answer (restate, then answer) | D1+D3 without restating (main run) |
|---|---|---|---|
| 0 | 100% | 16% | 26% |
| 20 | 100% | 54% | 74% |
| 60 | 100% | 57% | 82% |

1. **The model sees and knows the change.** In all 450 × 5 answers it restated the new rule 2 correctly, for any
   history.
2. **And it still answers the old way.** Right after a correct restatement, one line below, 14B gives the old
   answer: after 60 examples in 57% of D1+D3 cases (16% without history). It knows the rule but acts on the beaten
   track. This is an automatism in its pure form, not "did not see it".
3. **Restating before answering helps partly.** After 60 examples, old answers are 57% instead of 82%: minus 25%
   (interval −37…−15). This is the first "lamp": making the model say the rule out loud removes a third of the
   errors, but not all of them.
