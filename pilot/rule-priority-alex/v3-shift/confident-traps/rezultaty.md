# Small pilot of "confident traps": result (2026-10-23)

**Why.** The fastthink test failed because the chosen independent tests were easy for the models: they had almost no
habit errors. Here we checked on which kinds of tasks the models have a **confident** habit: the model quickly and
confidently answers "as usual", although the task forbids it. The lamp was **not** looked at in this step: only fast answers.

**How.** 517 new tasks, 9 kinds. Each kind has 40 traps (the habitual answer is wrong) and 20 controls (the habitual
answer is right). Three models: Qwen2.5-14B, Phi-4, OLMo-2-13B. Each model picks an answer immediately, without
reasoning. Confidence is how strongly the model prefers the chosen option over the others. The run took about 15
minutes on Modal (L40S), about $0.5. I chose the sampling seed (20261023): the pilot is not frozen and nothing is
proven on its basis.

## Main finding

A confident habit exists **in all three models** on 4 kinds of tasks:

| kind of task | Qwen | Phi-4 | OLMo |
|---|---|---|---|
| hindsight (fresh items from the published template): "lost, so the decision was bad" | 60 | 8 | 25 |
| redefine (independent test): "π = 39+14, first digit of π?" | 58 | 50 | 12 |
| our π and e with new numbers | 28 | 44 | 3 |
| parcels: the rule changed after 60 examples | 25 | 55 | 0 |

The number is how many traps out of 100 the model solved correctly when answering immediately. Almost all errors here
are habitual (the model chose "as usual"), and the model is confident in them: on average 80–100 out of 100. All
three models solve the controls at 95–100 out of 100. So the models understand the format and fail exactly where the
habit is wrong.

In two more kinds the habit is confident in two models out of three:
- **definitions** ("an octagon has 9 sides"): Phi-4 60, OLMo 30, while Qwen makes few errors (80);
- **bonus** (the scoring rule changed): Qwen 2, OLMo 0. Phi-4 makes errors (40), but unconfidently.

Where there is no confident habit:
- **capitals** ("the capital of Sweden here is Gothenburg"): the models are almost always right (77–100).
- **memo-trap** (finish a famous phrase with a different word): Qwen 70, OLMo 82. Only for Phi-4 (52) does it look like a habit.
- **road words**: Qwen has a strong habit (18), Phi-4's errors are unconfident. OLMo also gets confused on controls
  (55), i.e. it understands the format poorly.

## As a picture

A student quickly blurts out an answer from an old notebook and is 95 out of 100 sure of it. Exactly these errors
cannot be caught by the simple check "where is the student in doubt": they are not in doubt. So these 4–6 kinds of
tasks are a fair testing ground for whether our lamp ("the fast answer differs from the answer after thinking")
catches them. fastthink had no such testing ground.

## Limitations

- 40 traps per cell, so each number is uncertain by about ±15.
- This shows that the model is confidently wrong. It does not show that it **can** solve the task if it thinks. For
  Qwen this was seen earlier on π, hindsight and parcels. For Phi-4 and OLMo it will be checked in the next step.
- redefine, memo and hindsight are older public tests. The models may have seen them in training. That is why we
  regenerated hindsight from the same template, and for memo and redefine we took items not used earlier in the project.

## Next (needs a separate "yes" from Alex, ~$4–6)

A frozen test of the lamp on these kinds: hindsight, redefine, our π, parcels, plus definitions and bonus. The same
3 models. The lamp must beat "where the model is unsure" by a pre-registered margin. My guess (untested): since the
errors here are confident, the confidence check will be weak and the lamp has a chance.

Files: make_ct.py (item set), modal_ct.py (run), score_ct.py (scoring), report_pilot.txt (full table), out_pilot_*.jsonl (answers).
