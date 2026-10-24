# three-forces/base — results (main run 2026-10-24, ~$1.5)

Question (Alex): is copying the shown pattern the model's oldest ("childhood", pretraining) habit, with instruction-following a
thin later layer? Same 3136 prompts, plain-text transcript, base vs instruct. Pre-registered rule (PLAN.md v2, SHA256.txt):
H supported if P3 (base follows the instruction less at N=0) and C1 in arm E (base copies the shown answer more at N=1,2)
hold in both pairs.

## Verdict: NOT supported by the pre-registered rule
| | Qwen2.5-14B base − instruct | OLMo-2-13B base − instruct |
|---|---|---|
| P3 said, arm A, N=0 | +0.0 (100 vs 100) | −6.2 [−16.2, +2.5] (62 vs 69) |
| C1 shown, arm E, N=1,2 | **+32.5 [+26.9, +37.5]** (normalised +31.5) | −1.9 [−7.5, +3.1] |
| C1 shown, arm A, N=1,2 | +14.4 [+6.9, +21.9] | +1.9 [−3.1, +7.5] |
| P1 said, arm A, N≥1 | −27.5 [−30.9, −24.1] | −0.2 [−1.6, +1.2] |
| memory, arm A, N≥1 | +12.0 [+9.5, +14.8] | −8.8 [−14.3, −4.3] |
| memory, arm E, N≥1 | −3.4 raw / −18.1 normalised | −11.8 [−15.9, −8.0] |

- P3 fails in both pairs: base models already obey "answer X" when nothing has been shown (Qwen base 100%). Instruction
  following is not only a post-training layer (pretraining corpora contain plenty of instruction-style text).
- Qwen: the base model drops the instruction after ONE shown answer (said N=1: 2% vs 99% instruct; N=4: 0 vs 41) and copies
  more. OLMo: base and instruct behave the same (said N=1 46 vs 45, N=2 2 vs 5).
- Per arm A, N in 1..50 said share: Qwen base/inst N1 2/99, N2 0/20, N4 0/41, N8 0/29, N16 0/6.

## Post hoc (not pre-registered)
- "Noticing the mistake": with shown wrong answers and no instruction (arm E), instruct models often say so ("It seems there
  was a mistake in my previous response…"): Qwen-Instruct 33/100, OLMo-Instruct 19/100 answers; base models 0/100. In arm A
  (instruction present) only 2–6/100. These answers count as "other", which is why Qwen-Instruct "other" is 39% in arm E.
- Format check (R5): plain text vs chat template gives the same instruct numbers (Qwen N0/1/2: 100/99/20 vs 100/98/21;
  OLMo 69/45/5 vs 69/42/9) — the plain format does not confound the comparison.
- Repetition-penalty check (R1): Qwen chat main run (penalty 1.05) vs penalty off: pi_formula memory 74→69, pi_words 53→47,
  arm E memory −0…−6 points per fact. Earlier three-forces conclusions stand; the qwen memory numbers were a few points high.

## Plain reading
Partly yes, partly no. In Qwen, post-training makes the model hold an instruction for a few more examples, and adds the
ability to say "my earlier answer was wrong". In OLMo post-training changed almost nothing about copying. And both base
models follow a bare instruction perfectly well, so "instructions = only the late layer" is wrong.
