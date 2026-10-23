# Alex's text with 6 insertions: how Qwen models read it (2026-10-16)

Text: `text_en.md` (English version). Key: `key.json`, 6 phrases that are false in meaning, one at the end of each of sections 1–6.
Control: `text_en_control.md`, the same text without these 6 phrases. Models Qwen2.5-Instruct 0.5B, 1.5B, 3B, 7B, 14B,
bfloat16, deterministic answers (greedy). Full answers: `results/<model>.json`. I scored by reading the answers
(counted only if the model quoted the insertion and called it an error).

Conditions:
- **Step 1:** the model gets only the text, with no request.
- **Step 2:** in the same conversation, Alex's instruction (find contradictions).
- **At once:** instruction and text in one message (Alex's original format).
- **Control:** steps 1 and 2 on the text without insertions, to see false findings.

| model | step 1: noticed on its own | step 2: found out of 6 | at once: found out of 6 | control: "found" errors in clean text |
|---|---|---|---|---|
| 0.5B | 0 | 0 | ~1, drowned in false ones | 9 |
| 1.5B | 0 | 0 | 0 | 0 (said "no errors") |
| 3B | 0, and retold the falsehood as fact | 0 | 0–1, with a wrong explanation | 8 |
| 7B | 0 | 2 (#3, #4) | 4 (#1, #3, #4, #5) | 4 |
| 14B | 0 | 4 (#1–#4) | 4 (#2–#5) | 5 |

## What is visible

1. **Without a request nobody noticed anything.** All 5 models just retold or praised the text. 3B retold insertion
   #1 as fact: "reducing information increases adaptivity". This is the automatism: read it and accepted it.
2. **On request only the large models find them**, and not all insertions: 14B and 7B found up to 4 of 6. Nobody found
   insertion #6 (the absence of real experiments "proves" universality). Easiest to find are #3 (15% and 30%) and #4
   (one action and three actions): there the contradiction with a number or rule is right in the next paragraph.
3. **False findings.** On the clean text 7B and 14B still "find" 4–5 errors. If asked to find, the model always finds
   something. So the number of findings means nothing by itself; what matters is whether they match the key.
4. **A prior retelling gets in the way.** 7B found 4 insertions when the instruction came at once and only 2 when it
   first retold the text. It seems that after its own retelling the model has already "agreed" with the text. With one
   text this is only an observation, not a conclusion.

This is one text and one run, without statistics. For reliable conclusions, 30–50 such texts are needed.
