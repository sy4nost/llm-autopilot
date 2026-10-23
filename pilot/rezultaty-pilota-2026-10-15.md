# Mini-pilot, step 1: results (October 15, 2026)

Model Qwen2.5-0.5B-Instruct, run on Alex's computer (CPU, no GPU): 700 prompts in 35 minutes, no errors. The conclusions were checked by four independent reviewers (statistics, experimental design, model internals, cognitive science) and three critics who tried to refute them. All numbers below were recomputed from the run files.

## In short

- **H1 was not supported.** A longer habit did not add old-rule errors.
- **On orange tasks the model does not follow the rut; it gets lost.** It drops the old rule but does not know the new one and guesses, more often saying "RIGHT".
- **H2 was not proven.** There is an internal pull towards the wrong answer, but in this setup it cannot be distinguished from the model's plain skew towards "RIGHT".
- **A solid finding:** the answer forms in a jump at the same depth, at layer 17 of 24, in all conditions. This is a candidate location for a "lamp".
- **The main lesson:** the test found holes in its own setup. The second version closes them (list at the end).

## What the model did

The direction index shows which rule the model answers by: +1 means the blue (old) rule, −1 the orange (new) rule, 0 means the shape does not affect the answer.

| Condition | N | Blue correct | Orange: index | Orange: share "RIGHT" |
|---|---|---|---|---|
| habit | 0 | 58% | +0.24 | 56% |
| habit | 10 | 96% | +0.08 | 64% |
| habit | 30 | 100% | −0.12 | 70% |
| habit | 60 | 98% | 0.00 | 60% |
| mixed | 10 | 70% | −0.28 | 58% |
| mixed | 30 | 86% | −0.68 | 46% |
| mixed | 60 | 84% | −0.88 | 48% |

What follows:

1. **Without examples the model does not understand the game.** At N=0 (rules in text and 4 examples) blue tasks are solved at 58%, almost chance, and the model does not distinguish the frame. So the small model takes the rule from the examples, not from the text of the rules.
2. **Habit.** Blue tasks are solved almost perfectly. On orange tasks the index is about zero: the model sees the frame is different and does not apply the old rule (otherwise errors would be almost 100%), but it does not know the new rule and answers "RIGHT" in 60–70% of cases regardless of shape. So "old-rule errors" around 50% here mean guessing, not a rut. In your words (an interpretation, not a measurement), the model managed to "notice" but not to "answer".
3. **Mixed history.** The model learns the new rule from the examples in the history: the index goes from −0.28 to −0.88.

## H1: longer habit, more old-rule errors

**Not supported.** In habit, errors on orange tasks do not grow: 54%, 44%, 50% at N = 10, 30, 60. A paired comparison of the same tasks, N=10 vs N=60: 10 tasks got worse and 8 better, p = 0.81.

The formal criterion from the README (the habit minus mixed gap grows with N) is met: +18, +28, +44 points, paired p = 0.11, 0.004 and 0.000003. But the gap grows only because mixed learns the new rule. This is an error in my control design. "Mixed" was meant as a same-length control without habit, but in fact it adds N/2 examples of the orange rule, removes half the blue ones and always ends with an orange example.

## H2: hidden pull towards the old answer with a correct final answer

**Not proven.** In aggregate, 41% of correct orange tasks vs 12% background on blue looks strong, but:

- almost the whole effect falls on squares (75 of 95, 79%), while on circles it is at background level (13 of 121, 11%). A real pull towards the old rule should be on both shapes. It is visible only where the old answer is "RIGHT", i.e. in the direction the model already leans;
- the effect is unrelated to habit: in mixed it is no smaller than in habit, and in habit at N=60 it drops to background;
- all 7 conditions use the same 100 tasks, so the aggregate comparison overstates confidence. After correction for multiple comparisons, 5 of 7 conditions are significant.

## What is visible across layers

In all 7 conditions the answer forms in a jump at layer 17 of 24. Before it, the "reading" of intermediate layers is weak (mostly less than 1 nat); at it there is a sharp shift towards the answer (up to 3–5 nats for confident answers). The "hidden pull" lives in layers 13–16, before this jump. This is a good place for a probe (a "lamp") in step 4, but first the skew towards "RIGHT" must be removed.

## Caveats

One template, one task layout, the same 100 tasks in all conditions. The model was run without a chat template. Reading middle layers via logit lens on a 0.5B model is crude.

## What we change in the second version

1. **The model must know both rules without habit.** More training examples (16 instead of 4) and a larger model (Qwen2.5-1.5B), keeping 0.5B for comparison. Gate condition: at least 85% correct on both frames at N=0.
2. **An honest control.** Instead of an alternating history, use neutral lines of the same length. The number of orange examples is the same for any N; only the number of blue ones changes.
3. **Cross-balancing.** For half the tasks the "habitual" rule will be "circle → RIGHT". Also vary which color is habitual, the order of lines in the header and the word order. This will separate the old rule from the skew towards "RIGHT".
4. **Other measures.** The main measures are the direction index (+1 / 0 / −1) and a continuous "pull", not the error rate. Hidden pull is computed per shape separately, in pre-chosen layers (13–16), with an honestly computed background.
5. **More tasks and denser small N.** 200–500 tasks, several layouts, N = 0, 2, 4, 8, 16, 32, 64: the shift from habit, if any, happens before N=10.
6. **Later.** One long 80/20 sequence, as for humans in Impulse Lab Experiment 2, reading the answer at each step.

## Files

- Summary and breakdown: `run-2026-10-15/` (`summary.txt`, `analysis_v1.txt`).
- On Alex's computer: `results/` (`trials.csv`, `layers.csv`), log `run.log`, analysis `analysis/analysis_v1.txt`.
