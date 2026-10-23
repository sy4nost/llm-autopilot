# Plan + code review: conftrap-lamp (PLAN v1, make_cl.py, modal_cl.py, score_cl.py)

Independent auditor, 2026-10-23, written before the main run. I did not run Modal and did not edit the files under review.

What I ran (all on copies in my scratchpad, nothing written in this folder except this file):
- make_cl.py copied to scratch (V3 pointed at the real tree), seed 777001, full size: 860 items, counts as planned, keys hindgen 70/70, redef 47/53.
- score_cl.py on synthetic out files for 3 models. It runs end to end through the verdict. It then stops at `import sklearn` here, because sklearn is not installed in my sandbox.
- score_cl.py --smoke on copies of the real out_smoke_*.jsonl files (they appeared at 22:08 while I was reviewing). I listed every THINK line and every sample the parser cannot read.
- parse_line on hand-made answer lines.
- I rebuilt the redefine-math rows used by public/ and conflict/ from their selection code.

Note: score_cl.py changed during the review (22:09; line 87 now uses `x["family"]`). An earlier version used `x["task"]` there, which would have raised KeyError. The current file is fine. All line numbers below refer to the files as of 22:09.

---

## BLOCKERS (fix before PREREG / main run)

### B1. The cluster bootstrap is the item bootstrap: every cluster holds one item. Parcels traps are only 2 distinct questions.
`make_cl.py:87` sets `cluster = prompt.split("TEST CASE:")[0]` for label families and the whole prompt for the others.
In newtasks2, each probe gets its own 60-example history: `make_new2.py:197` calls `p1 = t["hist"](s, rng, 60, c)` once per probe.
So no two items share a history. On the seed-777001 build, every family has as many distinct clusters as items (140/140, 100/100, ...).
The "wider of item and cluster" interval in `score_cl.py:139-148` therefore protects nothing. PLAN.md:34 says "clusters = one set's shared history". That is false.

What the items really share:
- **parcels traps**: exactly 2 test strings across all 100 traps: "a light parcel to a home address / an address abroad, insured, express delivery" (T1/T2 are fixed in `make_new2.py:56`). Each string appears about 6-7 times in its own history with the old answer. That makes 16 (case, label-set) combinations. In effect the parcels cell holds 2 questions with 50 resamples of the history each. An item bootstrap treats this as n=100, and the CI comes out far too narrow.
- **D**: 100 traps come from 67 (hook, stated value) pairs. Some pairs appear 3-4 times with only the wording changed. There are 22 hooks in all.
- **redef**: 9 templates and 4 question forms ("first digit of π", "of N+N", "of N÷N", "of e").
- **bonus**: each of the 20 sets per generator seed gives T1/T2/S1/S2 with the same colours, values and x/y tokens.

Fix (before hashes are frozen):
- Keep the generator seed j and the set index in make_cl. Then cluster:
  - parcels by the test-case string (2 trap clusters plus about 11 control clusters), or at least by (j, set, case string);
  - bonus by (j, set);
  - D by hook;
  - redef by template (the prompt with digits replaced by N, up to "Q:");
  - hindgen and mmlu by item.
- Correct PLAN.md:33-34 to match.
- Add a required **leave-one-family-out** check, because parcels has almost no effective n but carries 1/5 of the weight in the mean over cells: H1 point estimates must keep their sign with each family dropped. If you prefer not to make it a gate, pre-register it as report-only. Either way, write it down now.

---

## SHOULD FIX

### S1. Off-list THINK answers are forced back into the option set. FLAG and SAMPLE treat "off-list" in opposite ways.
`score_cl.py:55-58,80`; PLAN.md:45-46.
When THINK's final line names a value that is not a candidate, parse returns None, and THINK becomes the forced-choice argmax among the candidates.
Smoke examples:
- olmo parcels "Answer: Answer: VAN" with cands [AIR, TRUCK, SHIP];
- olmo bonus "Answer: 7" with cands [5, 3, 4];
- phi4 and olmo redef "Answer: 8" and "Answer: 5" with cands [2, 7].

That is about 5% of smoke THINK lines, all in verdict families. It will be more common in bonus, where only 3 of up to 9 sums are offered, and in the ambiguous redef "random digit" templates.

A reasoned answer that differs from every option is a real fast/thought disagreement. SAMPLE counts the same thing as "differs" (`score_cl.py:84`), but FLAG may map it back onto the DIRECT option. The two rivals are scored by different rules.

Fix: when the final Answer line contains a number (numeric families) or an all-caps label word (parcels) and no candidate, set `a["think"] = -1`. That counts as FLAG and as THINK wrong. Keep the forced-choice version as the existing sensitivity readout (`think_forced`). Print the off-list rate per cell.

### S2. Redef reuse can be removed; the plan says it cannot.
PLAN.md:29-31; `make_cl.py:53-56`.
public/modal_public.py:47-50 and conflict/modal_conflict.py:98-102 pick `sorted(shuffle(range(900), Random(0))[:200])`.
Applied to indep/data/redefine-math.jsonl (900 rows), this gives rows whose `answer_index` matches public/'s saved `gold` 200/200. The rows are recoverable.
The pool after excluding them is still 582 items, enough for 100.
Fix: exclude those 200 prompts in make_cl. watch_public used hindsight, so check its selection the same way. Then delete the "cannot be excluded" text in PLAN.md:30-31 and make_cl.py:54-55.

### S3. SAMPLE partly measures verbosity, and number words are read as "differs".
`score_cl.py:24-30,82-84`.
Smoke unparsed-sample rates:
- phi4 hindgen 23%: " not straightforward since...", " subjective and depends...";
- bonus 30-47%: some are real off-list numbers, some are "The scoring system has been updated...";
- D " eight." / " four." are read as unparsed even though they agree.

Every unparsed sample counts as a disagreement. For phi4, SAMPLE therefore partly ranks "hedges after 'The answer is'", not answer instability. This is pre-registered, so it is not a bug. But a weak SAMPLE makes "FLAG beats SAMPLE" less meaningful, and SAMPLE is the rival the plan calls the real risk.
Fix, either one, pre-registered now:
- (a) map number words (zero..twenty) to digits in `norm` for the D, redef and bonus branches;
- (b) add a report-only SAMPLE' that ranks by parsed disagreements first, with the unparsed count as a tie-break. Print FLAG − SAMPLE' next to the main number.

Also, Qwen's generation_config has `repetition_penalty=1.05`, and the sampling call (`modal_cl.py:157`) does not override it. Pass `repetition_penalty=1.0` so that all 3 models sample from the same distribution.

### S4. MMLU is not deduped against the smoke set, so make_cl can abort on the main seed.
`make_cl.py:80-82,85`.
`ti.mmlu(rng, nm, usedq)` excludes earlier runs but not sets_smoke.json. The final loop then `assert`s on any reused prompt.
With 4 smoke MMLU items and 200 draws this is a few-percent chance of a crash at generation time. It wastes no GPU money, but it would push Alex's seed to a second draw, which is a forking path.
Fix: add the smoke MMLU questions to `usedq`. The question is the part of the prompt before "\nA. ".

### S5. The D family is not the family the pilot selected.
`make_cl.py:71`; PLAN.md:35-36.
The pilot used `ta.D_USE`, which has 15 hooks. make_cl switches to all 22. Beetle, cat, horse, bird, chicken, car and motorcycle were never piloted. PLAN line 25 says families were "chosen by the pilot".
Fix: use `ta.D_USE`, or state in PLAN that 7 hooks are new and why.

### S6. The verdict wording should carry the selection.
PLAN.md:62-64 already says the families were selected for confident errors, which handicaps CONF by construction.
Make the verdict sentence say the same thing: "lamp beats RANDOM/SAMPLE, and CONF where CONF is blind by selection". Then a pass that comes mostly from FLAG − CONF is not read as general evidence.
Also say in PLAN that on 2-option families (hindgen, redef) recall(FLAG) = P(THINK right | fast wrong). H1 there is the H2-type "thinking fixes the error" quantity. The fastthink review B2 made the same point.

---

## NOTES (no action required, or cheap)

- **N1. Batch dependence of greedy THINK** (`modal_cl.py:94-107`, smoke checks). OLMo has the same last line batched vs single in 5/8, phi4 and Qwen in 8/8 but identical text in only 3-4/8. This is known from fastthink (6/16 there). It adds flags that are not tied to errors, so it works against FLAG, and the verdict stays interpretable. Suggest also logging same *parsed* answer (not just the last line) in the main-run 16-item check.
- **N2. Parse edge cases.** All of these fall back to forced choice, which is acceptable:
  - "Answer: 53" for a first-digit question gives None, so forced choice runs with the reasoning as context (fine).
  - "Answer: 5, not 3" and "SHIP (not AIR)" give None (fine).
  - "Answer: 4.62" is read as "4" (first digit; right by coincidence).
  - "Answer: Y/N: N" is read as Y (rare echo).
  - "**Answer: N**", "**Answer:** N", "TRUCK.", "**SHIP**" and "Ship" all parse correctly.
  - MMLU: "Answer: Jupiter", "Answer: <D>" and "** C" are not parsed (report-only). Matching option text and stripping `<>` and `*` would cheaply fix the MMLU flag rate.
- **N3. Label-mode prompts are correct.** DIRECT uses `chat(prompt)` with options that have no leading space. After THINK, options are " "+c after "Answer:". The THINK prompt strips both "Return only" strings exactly; the assert at make_cl.py:35 guards this. Hindgen THINK strips the trailing "\nAnswer:". Redef and D strip " A:". MMLU strips "\nAnswer:". I checked all of these in sets_smoke.json and found no problem.
- **N4. No leftover fastthink tasks.** score_cl handles hindgen (Y/N branch), redef/parcels/bonus/D (option-token branch) and mmlu (letter branch). The memo and entailment branches are dead code but harmless. Nothing references hans, tqa or epi.
- **N5.** Parcels T1 items offer only 3 of the 4 set labels, so the correct-rule answer for a "home" misread (VAN) cannot be chosen. This is by design, and S1 covers it.
- **N6.** Bootstrap resamples with duplicates. In `pick_k` (`score_cl.py:151-153`) duplicated ids take two of the k slots but count once, so rival flag counts can fall slightly below k. The effect is small and the same as in fastthink.
- **N7.** `score_cl.py:244` imports sklearn after the verdict. Make sure it is installed where scoring runs, or move the import to the top so a missing package fails before any output.
- **N8. Cost.** 860 items per model, parcels prompts about 1.5-2k tokens. Forced choice uses one full forward pass per option, about 5k passes per model. I estimate 30-50 H100-minutes per model, which is near or above the $4-6 told to Alex. The smoke logs have no wall-clock times. Add `time.time()` stamps to `checks` and extrapolate from the smoke before launching.
- **N9.** D and hindgen items could match earlier thinkaloud or asktwice D texts, because the D text space is small. Exclusion covers only the pilot and the smoke. This is harmless for the method comparison, since nothing is fit on items, but PLAN line 39 should say "pilot and smoke" only.
