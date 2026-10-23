# Lamp "the past conflicts with the current rule" (14B, 2026-10-16)

Idea: a habit is not "the model looks at the past" (it always does) but a conflict in which the past beats the
current rule. The lamp should fire only when what the "copying" heads carry disagrees with what the model computed
under the current rule.

## Setup

- The same 540 prompts (A/B/C, history 60), tested only on the held-out half (sets 16–30), because the heads were
  found on sets 1–15. Heads: the same 20 (`top20.json`, layers 29–38).
- Two readings at answer time:
  - **what the heads carry**: their contribution to the answer, translated into the three labels;
  - **what the model computed before them**: logit lens at layers 20, 24, 28, 29 (raw and corrected for the mean
    skew), and also the model's answer with the heads switched off.
- The lamp fires if these two readings point to different labels. When it fires, the heads are switched off.
- Files: `modal_lamp.py`, `lamp_Qwen2.5-14B-Instruct.jsonl`, `score_lamp.py`, output `score_lamp_out.txt`.

## Result

**What the heads carry.** Confirmed: with an exact copy in history, the heads push the old label in 100% of cases
after the rule change, and without a change they push the same label, which is correct (100%). So the heads are
"the past answer to the same case", no more and no less.

**Where the lamp fires** (share of prompts):

| lamp by | A, change (habit 90%) | B, change (33%) | C, change (0%) | A no change (0%) | B+C no change (0%) |
|---|---|---|---|---|---|
| lens, layer 28 | 100% | 87% | 80% | 100% | 85% |
| lens, layer 28, corrected | 73% | 73% | 80% | 73% | 78% |
| answer without heads | 40% | 53% | 47% | **3%** | 38% |

**Correct answers on held-out sets:** normal 63%, heads always off 69%, by lamp 67–69%.

## What it means

1. **The reading "what the model computed by the rule" cannot be read yet.** The lens in layers up to 29 almost
   always shows the same label and does not reflect the rule. So a lamp based on it fires everywhere, even without a
   rule change. Earlier, from averages, it seemed that "the new answer leads up to layer 32", but this cannot be read
   off for an individual prompt: too noisy.
2. **The comparison "with heads / without heads" discriminates better**: in A without a change the lamp is almost
   silent (3%), and with a change it fires in 40%. But in B and C it still fires in 38–53% of cases without any habit.
3. **For this task the lamp is not needed yet.** Simply always switching off the 20 heads gives the same result
   (69%), and where experience is right (A without change) answers barely suffer (97%).

## Next

- To read "what the model computes by the rule" in the middle, a trained reader is needed instead of the lens:
  a small linear classifier on top of the internal states, trained on part of the sets. This is a standard technique
  and usually reads much more accurately than the lens.
- Check whether switching off the 20 heads spoils ordinary tasks (math, general knowledge). Without that, the +6–8
  point gain cannot be called final.

## Addendum 2026-10-17: a trained lamp on saved readings (no new runs)
`probe_lamp.py`, output `probe_lamp_out.txt`. A small classifier trained on sets 0–14, tested on 15–29. Target: "the answer is the old label and the correct one is different".
- Separates habit errors from other answers: AUC 0.91 (0.5 = chance). Almost all the signal comes from one reading: how much the model's answer without the 20 copying heads still supports the chosen label (AUC 0.86). The layer lens gives nothing (0.50–0.56).
- Fires: on traps 36%, on D2+D4 11%, without a rule change 8%.
- Lamp + brake (answer without heads when it fires): correct 63% → 70% (always switching off heads: 69%). Without a rule change it spoils nothing (57% → 57%).
- Limitations: only our tasks and 14B. A second model pass (without heads) is needed. Internal readings were not saved on public tests, so it was not tested there.
