# A pull dial inside 14B (2026-10-16)

Alex's idea: do not erase experience, but weaken the pull towards it, so that the model decides by what is in force now.

## Setup

- The same A/B/C sets (history 60), with a rule change (D1–D4) and without (D1, D3). 540 prompts, H100, bfloat16.
- From the test case onward, the model's "view" of past examples is weakened: the attention weight on them is
  multiplied by 0.5, 0.25, 0.1 or 0 (fully closed). Variants: only late layers 36–47, only middle layers 24–35, all layers.
  The model's knowledge does not change; only the strength with which it reaches for the examples at answer time changes.
- "Lamp": weakening switches on only if a takeover is caught, i.e. at layer 32 the model leaned towards one answer
  and at the end gave another.
- Answer = the label with the highest first-token probability. The dial at 100% gives exactly the normal answer (check 0.0).
  Files: `modal_damp.py`, `damp_Qwen2.5-14B-Instruct.jsonl`, `score_damp.py`, output `score_damp_out.txt`.

## Result (all conditions together)

| dial | old answer after change (D1+D3) | correct where the change added HOLD (D2+D4) | correct without change, experience right (D1+D3) |
|---|---|---|---|
| 100% (normal) | 48% | 79% | 61% |
| 50%, all layers | 41% | 97% | 62% |
| 25%, all layers | 37% | 97% | 67% |
| **10%, all layers** | **30%** | **97%** | **74%** |
| 0% (closed) | 35% | 94% | 51% |
| 0%, late layers only | 49% | 90% | 66% |
| 0%, middle layers only | 43% | 97% | 84% |

Paired by set, 10% vs normal: old answers −18% [−29, −8], correct on D2+D4 +17% [+9, +26],
correct without change +13% [+3, +22]. With an exact copy in history (A), old answers 93% → 42%.

## What it means

1. **The pull can be weakened without breaking experience.** At 10% there are a third fewer old answers, and where
   experience is right (no rule change) the model answers even better (61% → 74%). Fully closing experience is worse:
   without a change accuracy drops to 51%. So what is needed is not a cut but a lower volume, as Alex said.
2. **The pull does not live in one place.** Closing only the late layers gives almost nothing (48% → 49%). The model
   "reads" the precedent earlier, and the late layers only express it.
3. **The correct answer does not appear by itself.** Old answers on D1+D3 went away, but correct ones did not increase
   (46% → 45%): the model moves to a third label. Without the pull it is visible that the new rule is hard for it.
4. **The "lamp" does not yet work as a detector.** The layer-32 vs answer divergence occurs in 76–83% of cases even
   where there is no takeover. It is not a "habit is pulling now" signal, so the lamp almost always fires and behaves
   like a plain dial. A more precise signal is needed.

## Next

- Find a precise takeover signal: which specific attention heads look at the exact copy at answer time (candidates
  for "copying"), and switch on the dial only by them.
- Weaken not all examples but only these heads (as in "Overthinking the Truth"), and check whether specifically the
  correct answers go up.
