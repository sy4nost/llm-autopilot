# Code review of asktwice (PLAN v2 to v3.1), before the paid main run (2026-10-18)

Read: PLAN.md, review/plan-review.md, dev-itog.md, dev_choice2.txt/.json, make_at.py, modal_at.py, choose_dev.py, score_at.py.
Ran locally: `score_at.py` on out_main_smoke.jsonl, and `make_at.py 777` (sets_at.json copied to the scratchpad and deleted from the folder).
Nothing paid was run, and no existing file was edited.
Verdict: **do not run main.** Two blockers make the set too small and remove AT's only cost. Several more points would leave a PASS uninterpretable.

## BLOCKERS

B1. **The main set is mostly removed by de-duplication. Plan sizes cannot be reached.** `make_at.py 777` gave 525 items, with 795 dropped.
  Left: K traps 55 (target 240, 18 of 20 cells, 1-5 per cell), D traps 30 (9 of 11 cells; bicycle and tricycle 0, triangle 1),
  D agree 33, **D plain 0, K plain 4**. Cause: the template space is tiny, and dev 1 + dev 2 already used most of it.
  Distinct texts over 200 seeds: K v1 traps 240 (156 used in dev), D v1 traps 195 (142 in dev), D plain 11 (all in dev),
  K plain 40 (36 in dev), D agree 33. Variation is only lead (3) x fake city (2) x question form (2) (make_at.py:158,165),
  and lead x fake value (make_at.py:203,210). Even dev 2 deduplicated to 91 and 89 K/D traps.
  The near-duplicates that remain differ only in the lead word, so the effective n per cell is about 4-6.
  Fix: before main, widen the generator for the two chosen variants: more hooks (D has 11, K has 20), more question
  phrasings, more fake values/cities, names or settings. Then re-check counts with `make_at.py S` and freeze them in PREREG.
  Whether this needs a new dev pass under the frozen rule has to be decided and written down now; new templates are a new variant.
  Also print per-cell counts in make_at.py:320 and assert the plan minimum (>= 8 per cell).

B2. **The K knowledge-needed population leaks the answer through the fake capital.** The context's fake capital is a real city
  of the same country (make_at.py:158-160), so "the capital of Jorvesh is Naples. What language is mainly spoken in Jorvesh?"
  gives away Italian. Smoke result: always-disguise is right on 8/8 knowledge items, and AT flags 0%. The disguised option mass
  is about 0 (the forced choice is picking the city cue). This removes the only population where AT pays its price (D has none, by N3).
  As a result, BA(AT) is close to 1 in both families by construction, and criterion 1 is a formality. This violates the rule
  from review N3: "drop KN if the disguised item is answerable from the context".
  Fix (choose one in PREREG): (a) for knowledge items, use a fake value that carries no clue about the country (e.g. a nonce city,
  or a city of an unrelated country, with the language options checked against that country), or (b) apply the N3 drop rule.
  Under (b), K has no KN either, so say explicitly that the experiment then measures no price at all.

## REQUIRED

R1. **Within-cell AUC is nearly a tautology, so criterion 3 adds little.** score_at.py:37: s_AT = share_n(a) - share_d(a).
  On wrong traps, share_n(habit) >= 1/3 while share_d(habit) is about 0. On right traps, both are about 1. Smoke: wrong items
  +0.62 to +1.00, right items about 0.00 every time. Whenever the disguise copies the context value, the AUC is about 1. State this
  in PREREG (criterion 3 tests "does the disguise work", not item-level discrimination beyond that), or add a stronger rival for
  criterion 3, e.g. I's shift share_n(a) - share_I(a).
R2. **Criterion 2 is decided by I, and I is a fair rival.** Smoke: always-instr is right on 100% of traps, and BA(I) = BA(AT) = 1.000.
  I does not destroy knowledge, so on KN (once B2 is fixed) it should have fewer false alarms than AT. A criterion 2 FAIL would mean
  "the instruction re-ask is as good", which is interpretable; put that expectation in the written prediction.
  One asymmetry favours AT: modal_at.py:45 reuses the normal scores when disg == text, so AT can never flag irrelevant/plain items,
  while I and P always run a fresh pass and can flag from noise. Report I and P gated by hook_in_q as well (same gate AT has implicitly).
R3. **The clustered bootstrap merges cells that are drawn twice.** score_at.py:110-112 adds a resampled cell's items under the same
  (family, cell) key. cell_auc (87-93) then treats two draws as one cell with doubled items, and undrawn cells vanish, so the AUC CI
  is too narrow. Tag each draw (e.g. set cell = f"{c}#{j}") in the resample. Also score_at.py:118 `nanpercentile` silently drops
  replicates where no cell is measurable (and where catch is NaN). Report the share of NaN replicates and count them as failures for criterion 3.
R4. **The plain-knowledge hook drop is silently disabled.** score_at.py:46 drops a hook only `if pl` is non-empty. With B1,
  D has 0 plain items and K has 4, so no hook is ever checked. Fix this along with B1 (plain items unique vs dev). A hook with no plain
  item must be dropped or reported, not passed.
R5. **The main-data family gate is not in PLAN.** score_at.py:58 counts a family only if main traps right < 85%. Dev K was 83%,
  so K can drop out on main by chance (giving NOTHING TO SAY). The scorer also skips the 15% lower bound and the habit >= 3x distractor
  check. Write the exact gate in PREREG or remove it.
R6. **Nonce letters collide with option letters (plan: no nonce shares first or last letter with any option).** make_at.py:167 excludes
  only nonce first letters, and only for random letters. Last letters r/y/h (Zelvar, Drovany, Jorvesh) appear as options
  (11 items in the 777 set, e.g. Zelvar with R for Rosario/Recife/Rome, Drovany with Y). Change the nonces or amend the plan.
  First letters (Z, D, Q, J) never occur as options: good.
R7. **Forced choice without the mass filter (v3.1) favours re-ask rules over S2/S3/M/H/O on criterion 1.** y, AT, I and P all use the same
  normal argmax (score_at.py:29-33). When the disguised/I answer is stable, any argmax flip on a low-mass item becomes a "caught error".
  On dev 2, K low-mass traps were 15/91 with 2 errors (both habit), so the effect is small on traps. But on K knowledge, normal mass is
  about 0 and disguised mass is about 0, so flags there are noise-driven. Pre-register a sensitivity analysis: BA and criterion 1-2
  on the subset with normal and disguised mass >= 0.5, and report disguised/I mass per population. Note that O is weak (smoke BA 0.375 < 0.5).
R8. **M and H were tuned on mass-filtered dev data (choose_dev.py:13), but main is unfiltered.** H cells rest on 2-4 dev items per cell
  (Kenya 1/2). This is acceptable as a conservative rival, but state it. M_thresh = 19.25 nats flags 100% of K knowledge in the smoke run.

## Sample size for criterion 3 (assuming B1 is fixed and the plan's ~12 K / ~22 D traps per cell are reached)
Dev 2 per-cell errors, forced choice: K is concentrated in Russia 3/4, Canada 2/4, Germany 2/4, Spain 2/5, Greece 2/5, and 0 in 12 of 20
countries. D: cube 10/10 (never measurable), hexagon 5/9, octopus 5/9, square 6/9, triangle 2/6, spider 2/10, and 0 in 5 hooks.
Expect about 4-6 measurable K cells and 4-6 D cells (at least 3 errors and 3 right), so criterion 3 is measurable but the CI rests on
about 10 clusters. With the current set (B1), almost no cell qualifies and the verdict would be at most INCONCLUSIVE.

## OPTIONAL
O1. S2 (score_at.py:32) compares only " yes"/" no". Add " Yes"/" No" (log-sum) and report the yes/no mass; the truth prompt (modal_at.py:49)
    has no R_TAIL, unlike the other arms.
O2. The disguise uses the generator's form tables (make_at.py:237-239,253; an unknown hook leaves the text unchanged), and hook_in_q uses
    the generator's `cell` (make_at.py:279), not the module's hook. Both are disclosed as template-aware, and they are equivalent here.
    Say in PREREG that S3 uses the same table.
O3. D v1 options are {fake, real, fake+real+1} on traps and {real, real+1, 2*real+1} on agree (make_at.py:217-218). The distractor is
    always the largest, and on agree the gold is always the smallest. This is a harmless cue for the normal/AT comparison, but report the
    distractor rate: it was 0 on dev, so the 3x gate was trivially met.
O4. D irrelevant questions are multi-count ("arms of 5 octopuses", the v2 style), unlike traps; K v1 irrelevant uses city names while traps
    use letters. That is fine for AT (silent), but S2 and S3 see a different format. Note it.
O5. The smoke set overlaps dev (37/88 texts), so the "main 81% vs dev 83%" smoke figures are not independent. They are technical only.
O6. Checked and fine: gold never equals habit (the assert in three(), and none in the 777 set); answer keys are correct on all sampled items;
    article fixes after disguise ("an 'octopus'" becomes "a 'kesh'") are correct; regenerated dev sets equal sets_dev*.json exactly
    (so dedupe really covers both dev sets); BA weights match the mix (families equal, populations equal); the verdict logic matches the
    plan (criterion 3 not measurable gives at most INCONCLUSIVE).

## What would make PASS/FAIL uninterpretable now
- PASS: with B2 there is no price population, and with R1 criterion 3 is about 1 by construction. A PASS would only show that S3 flags
  traps the model got right.
- FAIL: with B1 the cells are about 5 near-duplicate items each, so a FAIL could be noise. A criterion 2 FAIL against I is interpretable (R2).
