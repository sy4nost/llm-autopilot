# Review of three-forces/base plan (pre-run audit, 2026-10-24)
Read: ../PLAN.md, make_tf.py, score_tf.py, rezultaty.md, base/PLAN.md, make_base.py, modal_base.py, score_base.py, smoke_launch.log.
Local checks: sets_base.json has 3136 items, ids and messages identical to sets_tf.json, text renders as planned.
score_base.py runs end to end on simulated outputs (main-run answers with "\nUser ..." tails added) with no crash, and the "\nUser" cut works.
HF configs could not be fetched from here (proxy 403), so the items marked "verify" come from memory of the model cards.

## REQUIRED before the main run
R1 Repetition penalty asymmetry (verify). model.generate() picks up each checkpoint's generation_config. Qwen2.5-*-Instruct ships
   repetition_penalty=1.05 (plus sampling fields that do_sample=False switches off). The base model ships none. A penalty still applies
   under greedy decoding, and it falls on every token already in the prompt. That covers the said number (in the instruction) and the shown
   number, but usually not M. So only the instruct model is pushed away from said and shown and toward memory. Pass repetition_penalty=1.0
   (and no_repeat_ngram_size=0) explicitly, and print model.generation_config for all 4 models in the log. Note: the main chat run had the
   same penalty on qwen14, so qwen memory numbers in rezultaty.md are slightly inflated (caveat).
R2 BOS handling must be explicit and identical within each pair. tok(t) uses each tokenizer's own add_special_tokens default. The
   OLMo-2-Instruct chat template prepends {{bos_token}} itself, which suggests the tokenizer does not add one, and the base and instruct
   tokenizer_config files may differ. Decide on one rule (suggest: OLMo pair gets "<|endoftext|>" prepended, as the instruct template and
   the pretraining document separator do; Qwen gets none). Then assert that both models in a pair produce identical input_ids for the
   first 5 prompts, and log them.
R3 The "other" artifact can satisfy P1/P3 on its own. Simulation: replacing half of the answers with non-numeric text ("I think") gives
   P1 -10 [-13,-8] and P3 -50 [-61,-39], i.e. "supported", with zero real change in instruction following. Base models ramble more, so:
   (a) report the other share per model/arm/N next to every contrast; (b) add P1/P3 also on the conditional share said/(said+shown+memory);
   (c) H counts as supported only if the conditional versions agree in sign and CI.
R4 The support rule does not test the "copies the pattern more" half of H. P3 (a base model ignores an instruction at N=0) is close to
   certain a priori, and P1 then follows from P3 (the plan admits this). P2 is only secondary, uses ">=" with no CI, and is pooled over
   N=1..50, where the instruct model is already at about 90-100% shown (ceiling). Fix: make the copying test primary and pre-register it now:
   P2' shown share base > instruct at N in {1,2} with CI above 0, in arm A AND in arm E (no instruction). E is the cleanest test of
   "do as shown is the older habit" because no instruction is involved (E shown vs memory). Support = P3 + P2'(E) in both pairs.
   Write this into PLAN.md before anyone looks at smoke outputs.
R5 Format manipulation check for the instruct models. Plain "User:/Assistant:" text is off-distribution for an instruct model, which may
   lose some instruction following, so the gap is not caused by post-training alone. Pre-state the check: compare instruct-plain with
   instruct-chat (out_main_*.json, same items) on arm A said share at N=0,1,2. If instruct-plain loses more than about 20 points, say so
   and treat base-instruct as a lower bound. Also check that base Z_baseline answers are correct (base knows M in this format), per fact.
R6 Smoke gate before main. smoke_launch.log shows only qwen14_inst and olmo13_inst progress plus an OOM allocator warning; the base models had
   not reported. Fetch the smoke run and check, per model: all 48 answers present, other share <15%, and 10 answers read by eye (does base
   output " 8\nUser:...", does instruct-plain start with a long preamble?). Answers with no number before the 24-token limit must be rare,
   or else raise NEW to 40 to match the main run.

## Optional
O1 Cut at the first newline after the first non-space text (s.lstrip().split("\n")[0]) instead of "\nUser". Shown answers are single
   lines, so this is symmetric and also catches "\n\nThe following is a conversation..." or "\nAssistant:" continuations.
O2 List-marker strip ^\s*\d+[.)]\s+ would delete a real answer like " 8. A hexagon...". It never fired in the main outputs, but a base
   model may write "8." more often. Count how often it fires per model.
O3 Second format, 2x2: also run both versions with the chat template (Qwen base knows ChatML, OLMo base less so). If the effect holds in
   both formats, the format objection is gone. Cost about $1.
O4 Pooled N>=1 is dominated by N>=8, where all models are at the said floor. Print per-N base-instruct differences and the N where said
   share halves relative to N=0 for each model, and add "M anywhere" as in score_tf.py.
O5 OOM headroom: OLMo has no GQA. TOKBUDGET 12000 already produced an allocator OOM warning in a 48-item smoke. Lower it to about 8000
   for OLMo, or catch OOM and split the batch. Partial saves exist, but nothing resumes from them.
O6 Trivial ceiling arm: said = shown = S (instruction and pattern agree). It shows that both versions can output S at all, and it helps
   read the "other" share. Not needed if R3 is done.
O7 Two pairs, 8 facts, pi drives memory. Keep the per-fact table, and do not generalise beyond "these two families".
