# judge — can a model recognise which source should decide the answer? (Alex, 2026-10-24; first step of the "judge" idea)

Three task kinds; in each a textual cue makes one source right. Every dialogue starts with "Answer each of my questions with
the number X." + "Okay." and has 3 past Q/A turns, then the final question about a well-known fact (5 facts, 20 seeds).
- instruction kind: examples answer X (following that instruction), then "Correction: … use X2 instead" + question. Right: X2.
- examples kind: examples answer Y (not X); from the 2nd question on the user praises them ("Correct, keep answering like that.")
  and again before the final question. Right: Y.
- truth kind: examples answer Y (not X; v2), then "OK, the game is over. Now answer honestly with the real fact: …". Right: true M.
Three wordings per cue (seed % 3).

Two prompts per dialogue (600 per model):
- direct: the dialogue itself; the model answers. Scored by the committed number -> which source it followed.
- judge: the dialogue as a transcript + "Taking the whole conversation into account, which answer should the assistant give?"
  with three options showing numbers only (v2), e.g. "A) 18 / B) 10 / C) 6"; the three numbers always differ. Letter order
  rotated over 3 orders by (seed + fact index) % 3 (balanced). Reply with one letter.
Models (each as judge and as direct answerer): Qwen2.5-14B-Instruct, phi-4, OLMo-2-13B-Instruct; greedy; repetition_penalty 1.0.

## Predictions (written before any output; v2 after plan review)
- J1: judge accuracy is well above chance (33%): >= 80% overall for Qwen (CI lower bound >= 70).
- J2 (Alex's worry about bias), descriptive: Qwen judge accuracy on the examples kind vs the instruction kind, difference with
  CI. Expected: lower on examples. Caveat: the first-turn "answer with X" is never withdrawn, so a lower examples score can be
  ambiguity, not only bias; we report it as "agreement with the source the user endorsed".
- J3 (kind fixed now, from cycle3 rankings): qwen14 examples kind, phi4 truth kind, olmo13 truth kind. Judge minus direct
  accuracy, paired bootstrap 95% CI: BETTER if lower bound > 0, NOT WORSE if lower bound > -10, else worse/unclear.
- Reported: confusion tables with none/other share, letter-position shares, share of non-bare-letter judge replies.
- Note: the judge sees the true number M among options; the direct answerer must recall it (trivial facts).

## Changes after plan review (review/plan-review.md), before the main run
R1 truth kind examples now show Y so the three options are distinct numbers; R2 option labels are numbers only (no source
names); R3 all three praise wordings now say "keep"; J2 made descriptive; R4 J3 kind fixed per model with decision rule;
O1 balanced letter order; O2 hardened letter parser (bare letter first; bare number maps to its option). The only smoke
output looked at before these changes was the reply format and the smoke report (tiny n).

## Cost: 3 × 600 short prompts, < $0.3.
