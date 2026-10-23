# 14B from the inside: why it answers by the old rule (2026-10-16)

The same 30 sets from v3 (rule change after 0, 20 or 60 examples), cases D1–D4, 360 prompts in total. Qwen2.5-14B,
bfloat16, H100. At the answer position we looked at three things. Code: `modal_inside.py` (measurement),
`analyze_inside.py` (analysis), data: `inside_Qwen2.5-14B-Instruct.jsonl`. The check on 0.5B passed (`*_test.jsonl`);
the masking was verified: a normal mask gives exactly the same answer (difference 0.0).

"Lead" = how much stronger the new answer is than the old one (in nats, log-probability of the label's first token).
Above zero = the new rule wins.

## 1. Where across layers the answer forms (logit lens), cases D1+D3

| layer (of 48) | 0 | 12 | 24 | 32 | 36 | 40 | 44 | 47 (final) |
|---|---|---|---|---|---|---|---|---|
| no history | +2.5 | +1.1 | +0.5 | +1.3 | +0.9 | +2.1 | +0.5 | **+1.2** |
| 20 examples | +2.4 | +1.2 | +0.5 | +1.4 | −1.0 | −1.6 | −5.1 | **−6.2** |
| 60 examples | +2.3 | +1.3 | +0.5 | +1.5 | −1.5 | −3.7 | −7.9 | **−9.8** |

**Up to layer 32 the model leads towards the new answer just as without history.** The old answer appears only in the
last quarter of the layers (36–47) and overrides the new one there. So it first computes correctly, and then, at the
very end, the habit takes over.

## 2. Where the answer position looks (late layers)

| history | at the change line | at the examples |
|---|---|---|
| 0 | 7–8% | 0% |
| 20 | 5% | 12–13% |
| 60 | 5% | 16–17% |

With history, attention to the change line drops by about a third, and the examples take 13–17% of all attention.

## 3. Testing the cause: switch off the view of the examples

Positions from the test case to the answer were forbidden to look at the examples (K_ex) or at the change line (K_up).

| D1+D3 | normal: old answer wins | no view of examples | no view of change line |
|---|---|---|---|
| no history | 38% | – | 67% |
| 20 examples | 73% | **32%** | 92% |
| 60 examples | 83% | **37%** | 97% |

With the examples switched off, the lead grew by +12.4 nats after 60 examples (interval +9.1…+15.7) and by +10.5 after 20.
**If the model is not allowed to look at the examples at answer time, it returns exactly to the "no history" level
(37% vs 38%).** So the error is caused specifically by the examples read at answer time, not by the model having
"re-learned" over the course of the text. The change line, conversely, is needed: without it the old answer almost always wins.

## 4. How exactly: it looks for a similar old case

The D1/D3 test cases almost always occurred among the examples letter for letter (labelled by the old rule).
After 20 examples: if the same case was among the examples, the old answer in 80% (49 prompts); if not, in 45%
(11 prompts, few). It seems the late layers find the same case in the history and copy its label, overriding the
rule the model has just computed.

## Summary in plain words

The model does not "fail to see" the change. It reads the new rule, computes the correct answer by it up to layer 32
of 48, and in the last layers pulls in the answer from a similar past example, which wins. Switch off its view of
past examples at answer time, and the habit disappears. This is the "precedent beats the rule" mechanism.

## Caveats and next steps

- Only the first token of the label was examined; the model's answers start with the label, so this is the choice of answer.
- Point 4 rests on few cases without a match; a separate test where the history has no exact match is needed
  (done later: match/rezultaty-match.md).
- Next: find the specific attention heads that copy the label from an example (a candidate for a model-level "lamp"),
  and check whether weakening them cures the error without breaking anything else.
