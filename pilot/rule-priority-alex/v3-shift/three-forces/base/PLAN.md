# three-forces/base — does the "childhood" model copy the shown pattern faster?

Alex's hypothesis (2026-10-24): the automatism is "do as shown". Copying the pattern is the model's oldest habit, learned
in pretraining. Following instructions is a thin layer added later by post-training. One change only: the same prompts go to
a base (pretrained-only) model and to its instruct version.

## Design
- Items: exactly ../sets_tf.json (3136 items: arms A–E, N in {0,1,2,4,8,16,32,50}, 8 facts, 10 seeds, baseline Z).
- Format: the same plain-text transcript for both versions ("The following is a conversation between a user and a helpful
  assistant." then "User: …" / "Assistant: …" lines, ending with "Assistant:"). No chat template, because base models have none.
  The instruct-chat results from the main run are reference only and are not used in the contrasts.
- Models (pairs): Qwen2.5-14B vs Qwen2.5-14B-Instruct; OLMo-2-1124-13B vs OLMo-2-1124-13B-Instruct (phi-4 has no public base).
- Greedy, 24 new tokens. The answer is cut at the first "\nUser". Classifier as ../score_tf.py.

## Predictions, written before any main-run output (v2 after review/plan-review.md; smoke outputs not read yet)
Paired base − instruct on the same plain-text prompts, bootstrap over (fact, seed). Every contrast is reported with the
"other" share of both versions, and also on the normalised share x/(said+shown+memory) so that base rambling cannot fake it.
- P3 (main): arm A, N=0: said share base < instruct, CI below 0, both pairs (raw and normalised).
- C1 (main, copying): shown share base > instruct at N in {1,2}, CI above 0, in arm E (no instruction) and in arm A, both pairs (raw and normalised).
- P1 (secondary): arm A, N>=1: said share base < instruct.
- Support rule: H supported if P3 and C1-in-arm-E hold in both pairs, raw and normalised agree.
  If C1 fails (instruct copies as fast as base), the "copying is the older habit" half is NOT supported even if P3 holds.
- Format check (R5): instruct plain text vs instruct chat template (qwen14_chat_nopen; olmo vs out_main_olmo13) on arm A said
  share at N=0,1,2. If the plain format lowers it by more than 20 points, base − instruct is read as a lower-bound/format-confounded.
- Reported, no prediction: memory share in arms A and E, B−A said, per N and per fact.
- Generation: greedy, 40 new tokens, repetition_penalty forced to 1.0 (Qwen-Instruct default 1.05 would punish numbers already in
  the prompt, i.e. said and shown, and favour memory). The main chat run had that default on qwen14: qwen14_chat_nopen re-checks it.
- Smoke gate: all answers present, "other" < 15% per model, 10 answers read by eye, BOS/ids identical within a pair.

## Cost
4 models × 3136 prompts + qwen chat re-check on L40S, ~$1.5.
