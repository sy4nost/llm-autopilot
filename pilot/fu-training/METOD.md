# The "fu / good" method: how to teach a model the right attitude to experience

Recorded 2026-10-17 after two training runs on Qwen2.5-7B (rezultaty-fu.md, rezultaty-fu-v2.md).

## Goal
Not to take experience away from the model and not to teach it ready answers. To change its **attitude to experience**:
the past is a hint, not the truth. The new is not the truth by default either.

## Skill ladder (Alex's wording)
1. **Notice**: the condition now differs from the past (in any words, not by a keyword).
2. **Stop**: do not answer by habit.
3. **Check whether the change applies to this case.**
4. **Answer.**
5. **Remember**: experience grows but does not become law.

## Training principles
1. **"Fu" for both blind errors, not just one:**
   - blindly copied the past although it no longer fits;
   - blindly applied the new although it does not apply to this case (decoys).
2. **"Good" for relying on experience when it fits.** Otherwise the model unlearns learning from examples.
3. **Experience can be wrong.** Some past examples are deliberately wrong; the rule matters more.
4. **Breadth matters more than quantity.** The skill becomes general only if it appears in very different kinds of tasks:
   rules, records, tariffs, schedules, instructions, texts, dialogues, math, code.
   A thousand pairs of one kind teach that kind, not the attitude.
5. **Different wordings.** The change is presented in different ways: announcement, hint, ordinary phrase, at the start, in the middle, at the end.
6. **Gradualness (as for a human, it takes time):** from simple to complex. First explicit changes, then hidden ones,
   then decoys, then a mix. Repeat earlier kinds so they are not forgotten.

## Testing principles
1. Test only on what was not in training: new words, new wordings, **new kinds of tasks**.
2. Always both sides: traps (experience misleads) **and** cases where experience is right (including decoys). Improving one side at the cost of the other is not success.
3. Plus general harm tests: MMLU, GSM8K.
4. Far transfer: free-form answers (not a choice of two), Alex's autopilot test, a self-made GSM-NoOp.
5. A small run first; scale up only if the trend is positive.

## What has been shown so far
- v1: old answers 86% → 0%; transfer to unseen tariffs 12% → 94%; but the word "NOTICE" was learned and blind trust in the new appeared.
- v2 (+decoys, +different wordings): on familiar kinds traps 88%, decoys 100%; on unseen kinds transfer is partial.

## Next step for the method
Broad training: 8–10 different kinds of tasks, gradually, with all kinds of pairs; testing on 3–4 kinds not in training and on free-form answers.

(Outcome: the broad v3 training failed its pre-registered test on public benchmarks; see v3/rezultaty-v3.md. The "fu" line is closed as a negative result.)
