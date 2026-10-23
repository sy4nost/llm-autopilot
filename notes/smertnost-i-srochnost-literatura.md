# "The model knows it will die soon": what has been done (search 2026-10-19)

Alex's question: give the model a "block" saying that time and resources are limited and it will die soon, so that it is maximally efficient and does not waste time on habits. A quick search; the authors' numbers, not re-checked.

## What has been done
1. **Token / time budget in the prompt.** TALE (Han et al., arXiv 2412.18547): "you have N tokens" shortens reasoning with almost no loss of accuracy; too small a budget hurts. EMNLP 2025 "LLM reasoning under strict output length" (30 models): the best prompt and model change with the budget. A Stanford CS224R student report: a model with a "clock" (how much time is left) allocates time reasonably well by itself. The goal everywhere is saving compute, not fighting habit.
2. **Emotions and stakes.** EmotionPrompt (Li et al., arXiv 2307.11760: "this is very important to my career") claimed +10.9%; a replication (arXiv 2409.20303) found on average about +2.6%, i.e. the effect mostly comes from picking the best phrase.
3. **Survival and shutdown.** Agents with "energy" whose depletion = death: a Sugarscape-like simulation (arXiv 2508.12920, agents start attacking for resources, up to 80% attacks under severe scarcity), The Energy Society (arXiv 2607.14865, large models spend more than they earn), Yerkes-Dodson for agents (arXiv 2603.07360: under strong pressure behavior narrows to a single action within 5–12 turns), "Survive at All Costs" (arXiv 2603.05028: the threat of shutdown pushes towards risky behavior). This is about safety, not answer quality.
4. **Humans.** Mortality salience (200+ studies) usually does not increase effort but strengthens attachment to habitual views; on a real task performance is lower than in controls. Time pressure on the CRT test is used specifically to suppress deliberation: people more often give the intuitive (habitual) wrong answer.
5. **Rational economy of thought** (resource-rational metareasoning, arXiv 2410.05563; "think / don't think" routers): they decide when to spend thought, but by task difficulty, not by a sign of "habit here".

## What was not found
- A prompt "you will die soon, be efficient" as a tool against habitual answers. Nobody has linked urgency to reducing automatisms.
- CRT-trap tests on LLMs with explicit time pressure in the prompt.

## What it means for us (inference, untested)
- In humans urgency strengthens habit rather than weakening it: no time to think means answering with the first thing that comes to mind. In models a budget makes reasoning shorter. Our think-aloud experience: reasoning aloud fixes traps (R 47→92). So "you will die soon" is more likely to push the model towards a short habitual answer.
- Our "shake" test (disguise, seed 90): BANG, "wake up", noise = placebo; a simple "follow the rules" instruction worked better. "You are mortal" looks like yet another shake.
- A variant that might work: urgency + a clear statement of what not to waste time on ("do not answer from memory, check the current rule"). But then it is the instruction that works, not "death"; the two need to be separated.

## A cheap test (not run, needs Alex's word)
On the existing thinkaloud set (the same R/N/K/D traps + T0 for harm), Qwen 14B, no new data. Arms:
- DIRECT (as before), DEATH ("you have little time, you will be shut down soon, answer as efficiently as possible"), DEATH_PLACEBO (neutral text of the same length), INSTR (as before), DEATH+INSTR, THINK and THINK+DEATH (does the threat shorten reasoning and bring the errors back).
- Prediction: DEATH ≈ DIRECT or worse; THINK+DEATH shorter and worse than THINK. If DEATH > PLACEBO on traps without harm on T0, that is news.
- Estimate: about $1–2 on Modal.

(Outcome: the test was run as cycle/ and failed; see `../pilot/rule-priority-alex/v3-shift/cycle/rezultaty.md`.)
