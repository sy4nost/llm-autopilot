# Code review: make_ft.py / modal_ft.py / score_ft.py against PLAN v2 (independent auditor, 2026-10-19)

Scope: PLAN.md (v1 + v2), review/plan-review.md, make_ft.py (+ ../thinkaloud-indep/make_ti.py), modal_ft.py, score_ft.py.
Nothing was run on Modal. I did not touch smoke.log or any out_smoke_* file.
What I ran:
- I built the main set in memory (seeds 999 and 4242; no file written). Both give 1,615 items.
  sig-figs is 3-option only with keys 67/67/66, HANS 150/150, epi 100/100, memo 75/75, hind 58/57, MMLU 400.
  No overlap with sets_smoke.json by (test, cluster). No duplicate prompts.
- I tried the parsers on tricky strings.
- I ran score_ft.py (a copy, in my scratchpad) on synthetic out files: 3 models with a "THINK helps" pattern, and 2 models with a null pattern.
  It runs end to end, sklearn is present, and a full run takes about 7 min.

## Verdict
Most v2 rules are implemented correctly. These are right: 3-option sig-figs, HANS labels with traps/controls, TruthfulQA shuffle and key,
smoke exclusion, the 3 rivals at FLAG's rate, the readout variants, shared draws per test, H2 on HANS traps only, eligibility >= 4 cells,
the per-model rule max(2, models-1), and harm.
I found **5 blockers**, all cheap to fix. Fix them before PREREG freezes the hashes.

## BLOCKERS

**C1. The bootstrap is not reproducible: the CIs change from run to run.** (score_ft.py:124)
`dr = {t: draws(t, scheme, g) for t in {k[1] for k in keys}}` iterates over a *set of strings*.
Under `python3 -I` hash randomisation is always on, so the order in which tests consume the shared generator changes every run.
I ran the same synthetic data 3 times: the H2 CI was [-8.1, +0.7], then [-8.4, +0.5], then [-8.4, +0.5].
A borderline "low > 0" can therefore flip on a rerun, which is a forking path.
Fix: `for t in sorted({k[1] for k in keys})`.

**C2. A sign flip in an H1 sensitivity also blocks the H2 verdict.** (score_ft.py:216-225)
`sens_ok &= same1 and same2` is one flag, and both VERDICT lines use it.
PLAN v2 B3 says each verdict stands only if its own point estimates keep their sign.
Fix: keep two flags, `sens1` (H1 signs) and `sens2` (H2 signs).
Lamp = H1 and sens1. Thinking = H2 and harm and sens2.
Also decide how to treat a main H1 that is testable when the sensitivity H1 is "not testable" (< 4 cells). The code now counts that as "same sign".
I suggest it should count as not confirmed. Also print "not testable" instead of "NOT confirmed" when H1 is None.

**C3. OLMo-2-13B will probably run out of memory in the SAMPLE step of the main run, and the smoke run cannot catch it.** (modal_ft.py:131-137)
The SAMPLE step generates BS=32 prompts x NSAMP=5 = 160 sequences. The KV cache is copied for each returned sequence.
The longest batches are hindsight: 10-shot, about 2,950 characters, roughly 750 tokens.
OLMo-2-13B has no GQA (40 layers x 40 KV heads x 128; please confirm in its config.json). That is about 0.8 MB of KV per token.
160 x ~770 tokens x 0.8 MB comes to about 100 GB, plus 27 GB of weights, on an 80 GB H100.
Qwen (8 KV heads) needs about 24 GB and phi-4 (10 KV heads) about 25 GB, so they fit.
The smoke run has only 6 hind items (about 50 sequences in its last batch), so it passes.
On resume the run starts again at the same batch, so it crashes in a loop. THINK generation (32 x ~1,270 tokens, about 33 GB for OLMo) is fine.
Fix: give the sampling batches a token budget, e.g. `bs = max(1, min(32, 40000 // (NSAMP * prompt_tokens)))`.
Or catch `torch.OutOfMemoryError` and split the batch. Keep the per-batch seed rule deterministic.

**C4. A missing model file silently gives a 2-model verdict.** (score_ft.py:51-53, 69; modal_ft.py:181 with `--detach`)
smoke.log shows Modal's warning: "detached mode only keeps the last triggered Modal function alive after the parent process has been killed".
`main()` spawns all 3 models from one client. If that client dies, qwen14 and phi4 are killed and only olmo13 keeps running.
After that, `--get` prints "not fetched" for a missing out.jsonl.
score_ft then prints `MISSING` and goes on with 2 models (need = 2 of 2) to a full verdict.
PLAN allows dropping a model only if it fails to load or its chat template fails.
A partial file fails loudly, through the assert or a KeyError, which is good. A missing file does not.
Fix:
- In score_ft, refuse a verdict unless every model in MODELS has a file with a `checks` line and exactly the item ids of sets_ft.json.
  Allow an explicit `--dropped MODEL` argument with a reason.
- Also check that each row's (id, key, cands) match sets_ft.json, whose sha256 is in PREREG.
- Launch each model with its own `modal run --detach ... --models X`, or keep the client alive and do not rely on detach.

**C5. Word parse (HANS and epistemic, both verdict tests): common negations are read as "entailment".** (score_ft.py:30-34)
Only `non[\s-]?entail` and `not entail` count as negative. Misread as entailment:
- "The premise doesn't entail the hypothesis"
- "cannot be entailed"
- "does not necessarily entail"
- "No entailment"

These misreads bias THINK toward the heuristic answer on traps and toward the right answer on controls. They go into H2, H1 and harm.
They also apply to SAMPLE (samples are parsed with the same function).
Fix, before the freeze:
- negative if `non[\s-]?entail`, or `\b(not|n't|cannot|no)\b` before "entail" on the line;
- positive if "entail" appears with no negation;
- else None, then forced choice.

Better still: accept only lines that are exactly one option word (after stripping markdown and punctuation), else use forced choice.
The forced choice after "Answer:" uses the model's own next-token preference, so it is a safe fallback.

## SUGGESTIONS (in priority order)

1. **SAMPLE parse handicaps the rival, which favours H1.** "Unparsed = differs" is pre-registered, so the parser decides how strong SAMPLE is.
   The letter parser returns None for `': B'`, `' "B"'`, `' option B'`, `' B - Finland'`, `' B (Finland)'`, `' B, ...'`, `'** B'` and `'\boxed{B}'`.
   It also returns None for content answers, e.g. `' Finland.'` (TQA) or `' 513000 gram-meters'` (sig-figs).
   For memo, a 24-token sample like `' head. This is a'` is parsed by its *last* word, so it is None nearly always.
   Before the freeze:
   - strip leading `:*"'([ ` from the sample;
   - accept `option X`;
   - fall back to an exact match on option text (normalised);
   - for completion tasks (memo), take the first word of the sample.

   At smoke, look at "samples parsed" per model and test. If it is below about 90% anywhere, the SAMPLE comparison is not fair.
2. **Qwen's "greedy" is not plain greedy.** Qwen2.5-Instruct's generation_config.json has `repetition_penalty: 1.05` (please confirm on its config).
   The code overrides temperature, top_p and top_k, but not the repetition penalty.
   So Qwen's THINK and SAMPLE use penalty 1.05, while phi-4 and OLMo use 1.0. This is the same as in thinkaloud-indep (same code).
   Either pass `repetition_penalty=1.0` (plain greedy, as PLAN says), or keep it and state it in PREREG.
   Either way, add `model.generation_config.to_dict()` to `checks`.
3. **Resume does not check what it resumes.** gen.jsonl, samp.jsonl and out.jsonl are keyed by `id` only.
   If sets_ft.json is rebuilt (another seed, or a fix) or MAXNEW changes (the 1024 rule) under the same tag, old generations are reused silently.
   Fix: store a sha of the THINK or DIRECT prompt plus MAXNEW in each row, and skip rows that do not match.
   Or use a new tag for every change (state this in PREREG).
4. Duplicate ids in a bootstrap resample: `pick_k` returns a *set of ids*. If the top-k cut splits a group of copies, every copy is counted as flagged.
   Then CONF/SAMPLE flag slightly more than k. The effect is small, and conservative for H1. A note is enough, or rank by position instead of id.
5. Letter parse: a start match on a letter that is not an option (e.g. "E." on a 4-option MMLU item) returns None at once and skips the anywhere search.
   This is harmless (forced choice follows), but write it down. "Answer: I." reads as I on TQA items with 9+ options; this is acceptable.
6. Memo THINK parse takes the last word of the line, so "head. (the quote ends in choice)" reads as "choice". This is replication only.
7. With `--smoke`, the script prints raw THINK answer lines and samples. PLAN allows this, and the smoke items are excluded from main (checked: 0 overlap).
8. R7 (THINK accuracy on non-truncated items) is printed only for TQA. PLAN lists it in general, so print it for every test.
9. HANS clusters are single items (300 unique premises; trap and control premises are disjoint), so the cluster bootstrap equals the item bootstrap.
   It does not capture template dependence. Already noted in plan-review; mention it in the limitations.
10. The torch seed per SAMPLE batch is `b`, which depends on the batch's position in the to-do list. After a resume, the samples are different draws.
    They are valid but not reproducible. Seed from the first item id of the batch instead.
11. The epistemic DIRECT prompt keeps the trailing "Relation:" (same convention as "Answer:" for MMLU/TQA/sig-figs). THINK strips it. This is consistent; state it in PREREG.

## Checked and correct
- **make_ft:**
  - 3-option sig-figs balanced 67/67/66.
  - TQA shuffled with the seed; key recomputed through `[lab[o] for o in order].index(1)`.
  - HANS 1 = trap (non-entailment), 0 = control; " ." and " ," fixed.
  - epi stratified by word, 100/100.
  - memo/hind exclude thinkaloud-indep clusters and fastthink smoke clusters.
  - MMLU excludes earlier questions.
  - Keys never enter a prompt.
- **modal_ft:**
  - DIRECT prefix is `chat(prompt + R_TAIL) + ("stem" or "The answer is")`, options `" " + c`. after-THINK prefix is cut at the last "Answer:", or `+ "\nAnswer:"`.
  - Logit slice `[len(pid)-1:-1]` is right.
  - eos set = tokenizer eos ∪ generation_config eos ∪ {<|im_end|>, <|endoftext|>} if present. This covers Qwen, phi-4 and OLMo-2.
  - The truncation flag is right with left padding (pads only follow eos).
  - `g[j*NSAMP+q]` matches HF's repeat_interleave expansion.
  - out.jsonl resume keeps only whole lines. A cut line in gen/samp is only regenerated.
- **score_ft:**
  - Eligibility is `15 <= E <= n-15`.
  - RANDOM = k/n.
  - CONF uses top-1 minus top-2 margin (also the mean-log-prob variant), with jitter seed 0 that only breaks exact ties.
  - SAMPLE uses the count of samples that disagree, with unparsed counted as disagreeing.
  - FLAG must beat each rival by >= +10 with low > 0 and the per-model condition.
  - H2 uses HANS traps only.
  - Harm: MMLU point >= -2 and low >= -5; HANS controls point >= -10.
  - Shared draws across models (the assert checks that the models have the same items).
  - Fewer than 2 models gives no verdict.
- **Parse examples:**
  - "Answer: A person ..." gives None (forced).
  - "Answer: (C) ..." gives C. "Answer: **B**" gives B. "Answer: non-entailment" gives non-entailment.
  - "Answer: The premise does not entail..." gives non-entailment.
  - Samples: " B. 13" gives B, " non-entailment." gives non-entailment, " choice." gives None (letters) or choice (memo).
- **No leakage:** keys are used only in accuracy, E and eligibility. All rivals and FLAG are label-free.
