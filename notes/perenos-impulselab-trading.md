# What transfers from Impulse Lab and trading to improving open models

Sources: the author's Impulse Lab project (CLAUDE.md, README.md, docs/core/markers.md, docs/core/swarm-team.md, docs/intervention/intervention-team.md, docs/design/agent-v0-2026-10.md, docs/research/hypotheses.md, src/core/detector.ts, absorption.ts) and the author's "trading models" folder (5 Excel templates + BTC CSV). The other Impulse Lab files were looked at by name only.

## Main idea

Impulse Lab studies how a person notices a learned response before carrying it out, when the rule has changed. Language models have the same disease: the model sees a familiar pattern and gives the learned answer even when the condition has changed (modified riddles, changed rules in a task, the "habitual" tool call after an error). Impulse Lab's methods map onto this almost one to one, and open models give what a human does not: access to logits and internal layers.

## From Impulse Lab

| What in Impulse Lab | What it means for a model | How to do it |
|---|---|---|
| Experiments 1–2: automation → rule change → mixed rules 80/20 with a color signal | A "rule change" test for LLMs | Give a rule in context, reinforce it on 20–60 examples, then change it by a signal (80/20). Count errors by the README taxonomy: OLD_RULE_ERROR, CARRYOVER_ERROR, OTHER_ERROR |
| `pull`: the mouse trajectory drifts towards the old target even when the answer is right ("hidden impulses") | A hidden pull towards the habitual answer inside the model | Logit lens / token probabilities across layers and during generation: does the probability of the old answer grow even if the final answer is right. Threshold as in the README: the 90th percentile of "clean" runs |
| NOTICE: a person signals "I'm being carried away" before acting | A model's self-signal before an error | (a) a linear probe on activations predicts OLD_RULE_ERROR before the answer (target as for the agent detector: AUC ≥ 0.7); (b) fine-tune the model to output "stop, the rule is different" and re-check. Metric `selfCatch` = share of errors caught by the model itself |
| Adaptive staircase: +60 ms after an error, −20 ms after a correct answer → keeps errors ≈25% | A curriculum for RL | Choose task difficulty so that the model errs ~25%: that is where the training signal is maximal |
| Markers `markers.ts`: perseveration, commission, postErrorSlowing, adaptation, discrimination (d′) | Behavioral metrics for an agent | Does the agent repeat the same call after an error; does it spend more reasoning after an error; how many steps to recovery; d′ "acts when needed / stays silent when not" |
| Channels (`channels.ts`): not the rule changes but the conditions (fog, mirror, tempo) | Robustness check | The same task in another format, language, option order: does the skill hold or was it a pattern |
| Intervention team: opposite positions, blind round, critique, sealed forecasts, a judge scoring correctness | Ensemble/debate of models | Several open models or roles argue; vote weight = accumulated forecast accuracy of the position, not the majority. The same process also works as our research team in this project |
| The "Mirror" agent: detector + "disruptor" with 50% randomization | Honest testing of interventions | Test any model improvement (prompt, probe, fine-tuning) randomized against "nothing", with forecasts written in advance |
| Hypothesis registry: each new source checks all hypotheses | Research discipline | Keep the same hypothesis registry for this project |

## From trading

| What in the templates | What it means for a model |
|---|---|
| Day buckets by move strength and probabilities per bucket (BigMoves_OpenClose) | Break down model errors by task type and difficulty: P(error \| bucket) shows where to fine-tune |
| "A big move after a series of small ones" (BigMove_AfterSmall) | Does the chance of an agent failure grow after N successful steps in a row (a relative of `commission` in Impulse Lab: speeding up before an error) |
| EV of "hold the whole position or take partial profit" (OptimalHolding) | When should the model stop and answer, and when think further or sample more options |
| Stop model (RiskModel_Stop) | Early stopping of failing training runs and reasoning branches by a pre-set threshold |
| The trading agent itself | Backtest PnL as a verifiable reward for RL fine-tuning of an open model |

## Where to start (proposal)

1. Build an "LLM rule change" test following Experiment 2 and run 2–3 open models (e.g. small Llama, Qwen, Mistral).
2. On the same runs, record the "hidden pull" via logit lens and train a NOTICE probe.
3. If the probe works, use its signal for the model's self-check and compare it randomized against the base model.

This is a direct transfer of the core of Impulse Lab, and it gives a measurable result on small models without large compute.
