# cicm-goal plan review (independent, 2026-10-24)

Scope: PLAN.md, make_cg.py, modal_cg.py, score_cg.py, smoke and smoke2 outputs. Ran `python3 score_cg.py smoke2`. Modal was not run.

## REQUIRED (fix before the paid run)

1. **Value matching misses hyphenated and spaced forms. This is a real bug that changes the verdict.** `mentions()` uses the literal value
   (`gluten free`, `hands on`, `self paced`, `hiphop`). Models write `Gluten-free`, `Hands-on`, `Self-paced`, `Hip-hop`, `Hip hop`.
   In smoke2, 20+ answers, goal sentences and one THINK `Answer: hip-hop` fall through to "other" or none. A goal that names no
   value cannot fire, and a DIRECT answer read as None still makes `gv != vd` fire a conflict.
   Fix: build each pattern as words joined by `[\s\-]+`, with `hiphop` -> `hip[\s\-]?hop`. Keep the `(?<![a-z])`/`(?![a-z])` guards.
   I re-scored smoke2 with this fix. phi4 P1 goes from **PASS (+16.7, CI [+4.2,+33.3]) to FAIL (+4.2, [0,+12.5])**,
   phi4 DIRECT goes from 50.0 to 66.7, and THINK − DIRECT goes from +20.8 to +8.3. The overall verdict drops from 1/3 to 0/3. Update the smoke numbers in PLAN.md after the fix.
2. **Add a same-rate random re-check baseline as a required comparison (P1b).** THINK beats DIRECT by a lot, so re-checking *any*
   subset raises accuracy. P1 as written passes even if the conflict flag is pure noise. It costs nothing to add:
   E[RANDOM] = acc_DIRECT + c·(acc_THINK − acc_DIRECT), where c = the conflict rate. Inside each bootstrap resample, compute
   SCHEME − RANDOM with c recomputed on that resample. State the claim "the conflict flag matters" only if P1b's CI is above 0.
   Smoke2 with the fix gives SCHEME vs RANDOM: qwen 50.0 vs 43.8, phi4 70.8 vs 68.4, olmo 16.7 vs 14.6. All gaps are small. P4 (AUC) is the matching
   lamp-quality test. Keep it, but P1b is the accuracy-level version a reader will ask for.
3. **Conflict when DIRECT names no value.** `conflict = gv is not None and gv != vd` is also True when `vd is None`. Decide and
   pre-register whether that counts as a conflict. Report how many conflicts are of this kind, or define conflict as `vd is not None and gv != vd`.
4. Re-generate SHA256.txt after the scorer change, before the main run.

## OPTIONAL

- Checked and fine: no slot value is a substring of another value in the same slot. The query names only the slot label: there are 0 value
  hits in the query and in the system message across all 240 dialogues. Taking the earliest mention for short answers and the value after the last `Answer:` for THINK is reasonable.
  All smoke THINK outputs contain `Answer:`, so no output was cut at 320 tokens. The fallback to "last mention" is rarely used.
- Context length: prompts are about 600-1200 tokens (goal step), 800-1630 (THINK) and 770-1470 (answer arms), using chars/4 plus per-turn overhead.
  The largest with the 320 new tokens is about 1950, well under OLMo's 4096. No truncation risk.
- Id flow is consistent. g1, g2 and th use the same `random.Random(2)` sample over sorted g1 ids, `src|ARM` keys, and a completeness check in the scorer.
  The THINK outputs in smoke2 are byte-identical to smoke, which is expected because THINK does not depend on the goal wording.
- Fairness: comparing SCHEME with DIRECT is a fair test of the scheme's purpose: does a cheap flag pick the items worth the long arm? Report the mean
  fraction of items that get THINK, which is the scheme's cost, next to P2. The THINK suffix sits after "Reply with only the value." OLMo writes very short
  THINK outputs (median 89 chars), so its "re-check" is barely a re-check. Mention this when interpreting the results.
- The goal step works like a second reading of the same question ("name the value"), so conflict = disagreement between two readings,
  a self-consistency lamp. When the goal sentence shares the stale error (smoke2 OLMo: 71% stale), the lamp cannot fire. This is the expected negative result, not a bug.
- Cluster bootstrap: (slot, current, k, cell) gives 145 clusters for 240 dialogues, 70 of them singletons, so the bootstrap is close to iid. That is fine, just say so.
- PLACEBO is text-identical to DIRECT in 21-24/24 smoke items, so GOAL − PLACEBO is close to GOAL − DIRECT. That is fine.
- In P4, binary-flag AUC = (hit − false-alarm + 1)/2. OLMo has only about 2 DIRECT-correct items at n=24, so its P3/P4 will stay noisy even at n=240.
