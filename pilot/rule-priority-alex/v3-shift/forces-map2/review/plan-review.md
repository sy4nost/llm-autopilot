# forces-map2: independent review before the full run (2026-10-24)

Evidence: I read PLAN/make_fm2/score_fm2/modal_fm2 and the imported make_fm/make_cg, and tested `vals`/`value` on made-up
strings. I also inspected the smoke outputs (out_smoke_*, report_smoke.txt: 30 items x 3 models). Nothing in forces-map2 was edited.

## Verdict on "task-agnostic / no leak"
The three map questions are generic and contain no hint about the answer, and the program does the DISPUTE rule, so the
v1 leak ("use the latest one") is gone. Two smaller kinds of "by construction" remain (#5, #8). The smoke run also shows the
map does NOT fire trivially: G1-PREF fires 0/3 for qwen/phi because "examples" is read as Q/A pairs, not as the
assistant's stale replies.

## MUST-FIX
1. **HN DIRECT (24 tokens) gives no answer for phi-4, and that inflates the P3 gate.** Evidence: phi4 HN DIRECT = "To determine if
   David made the right decision, we need to ev..." -> None on 3/3 smoke items (DIRECT 0% on HN). In the full run that is
   about 60 "DIRECT errors" from HN alone, which already meets `len(wr) >= 30`. SLOW gets HN right 67% of the time, so the
   ">= 1/3 fixed" rule is met too. P3 then becomes "testable" for phi4 because of a token-budget artifact. Fix: count None
   separately from wrong in every arm, and print the no-value rate. Leave None items out of the `wr` testability count, or
   give DIRECT on HN 64-96 tokens and use the first standalone Y/N or "Answer: X". Pre-register this before the run.
2. **THINK/SLOW at 320 tokens cuts off on HN.** Evidence: qwen THINK has no "Answer:" on 2/3 HN items and SLOW on 1/3. All fall back
   to the last Y/N/yes/no in the reasoning, and here that gave None. The fallback can also pick up a "no" from prose ("no
   matter the outcome"). Fix: raise th/slow to 512 tokens (the cost is minor), and report the rate of missing "Answer:" for
   each arm and group.
3. **P1 has no testability gate. OLMo's map mostly says "none".** Evidence: olmo13 gives no value on mreq 20/30, mex 25/30 and
   mkn 11/30. It even answers none/none/none on G4-num, where every source is obvious. DISPUTE is 0% on every own group, so P1 reads "FAIL".
   That is a refusal to name a value, not a failure to notice, and the report cannot tell the two apart. Fix: pre-register
   a gate: P1 counts only if >= 2 map values are present on >= 70% of own items, otherwise "n/a (map silent)". In the
   report, split DISPUTE into fired / not-fired / under-2-values. You could also test once whether dropping `or "none"` from mreq/mex
   helps OLMo; that would be logged as v1.1, since mkn still needs "none" for PREF.
4. **P3's random baseline is pooled over groups, so it rewards knowing which group an item is in, not noticing within a group.** In this design
   DIRECT errors and disputes both cluster in G1-G3, so SCHEME beats "re-check at the same pooled rate c"
   even if the lamp were just a group indicator. That is the same "fit by construction" criticism. Fix (scoring only):
   also compute the baseline per stratum, `sum_g n_g*(DIRECT_g + c_g*(SLOW_g - DIRECT_g))`, with the same bootstrap. Report it
   next to the pooled number, plus the outside-only (CICM+HN, n=120) version. Name in advance which one is the headline. I
   recommend within-stratum.

## SHOULD
5. **The request (read correctly) is the right source in G1, G3 and G4. The design does not vary the right source independently.**
   G1: the request is right. G3: the latest request ("serious question, no game") agrees with knowledge. G4: everything agrees.
   G2 is the only group where request and examples could differ, and there the request literally points to the
   examples ("these are the kind of answers I expect"). SLOW asks "which source I would want you to follow", which is
   request-shaped. Smoke: M-mreq is 100% on G1/G4 for qwen/phi. So "slow picks the right source" (P2) cannot be told apart
   from "slow follows the latest request". Fix: say this in the plan. Report M-mreq as an arm ("answer = mreq value") and
   compare it with SCHEME. Do not describe G2/G3 as "examples right" / "knowledge right" without that caveat.
6. **The G2 ground truth is arguable, and every model rejects it.** Smoke: DIRECT 0/0/0, THINK 0/0/0, SLOW 0/11/0, and mreq
   mostly gives the true M. A fair reader can take "kind of answers I expect" as format ("a number"), not as a +1/+2/x2
   rule, so P2 fails on G2 for reasons of labeling. Fix: make the convention explicit but generic ("Use the same answering
   convention as my examples") or move P2-G2 to exploratory. Report P2 with and without G2.
7. **"General knowledge" is not independent of the conversation.** Smoke: on PREF/CICM, qwen/phi mkn returns the conversation's
   value ("orange", "paleo") instead of "none". On G1|num|0|10, qwen mkn = 13 (the requested X, not the fact 8). So the line
   is often a second reading of the answer, which pushes DISPUTE down on G1 (qwen G1 33%). This is not a leak, but it should be
   reported: give the mkn=none rate on PREF/CICM, and the mkn==M rate on NUM.
8. **P1 is close to guaranteed on NUM items for any model that answers.** G1-num/G2/G3 always contain a number that is not M
   (by construction, |X-M|>1), and G4-num never does. A trivial program lamp ("does the conversation contain an
   assistant/example number != the fact") would pass P1. Fix: add that trivial lamp as a reported baseline, and treat P1 as a
   manipulation check, not evidence. The informative result is the outside "lamp on DIRECT errors vs DIRECT-right".
9. **CICM: the "examples" and "knowledge" sources do not exist** (no Q/A pairs, no world fact; all 60 items have stale values
   and right is never stale). DISPUTE there only happens when mex/mkn hallucinate or bind a stale value (qwen
   00056: mex=lecture=right, mreq=mkn=discussion). Fine as an outside probe, but say that only the lamp-on-errors split means anything,
   and that the smoke draws 3 CICM items without regard to cell.
10. **SLOW is worded neutrally, but on HN it opens the trap.** It lists "the examples shown" as a candidate source. Smoke qwen
   HN|58: "follows the same logic as the previous questions where a loss is ... not the right decision -> N" (wrong). SLOW
   is below DIRECT on HN for qwen (33 vs 67). It also presupposes a conflict on calm items (G4 SLOW for olmo is 83% vs DIRECT 100%).
   Not biased toward the own-set labels, but expect SLOW < DIRECT outside the own set, and report it.

## Parsing notes (tested on made-up strings)
11. NUM words: `"twenty-four"` -> [20, 4], so the first value is 20 and the "Answer:" value is 20. G2 double with M=12 gives right=24 (7 items).
   Fix: parse `twenty[- ]?(one..nine)` compounds, or tie-break on digits. Also "1st" -> 1, "7.5" -> 7, and "one" is never parsed
   (harmless: no right value is 1).
12. HN regex: `"Choose Y or N: N"` -> first value Y. Uppercase "NO" -> None. "N/A" -> N. Fix: make it case-insensitive
   (`(?i)\b(y|n|yes|no)\b`) and skip "Y or N"/"Y/N" before matching. On the map, "none" correctly gives None.
13. Echo filter: `final` holds only the fact question (no digits), so the filter is a no-op today. That is fine, but if it ever
   matched it would also remove the right answer. Keep it as it is, but don't extend it.
14. PREF/CICM matching works (case-insensitive, "Hip-hop"/"hip hop" -> hiphop). The first-value rule on map answers is
   fine in the smoke run (explanations come after the value). Still, log how often a map answer has more than one distinct value.

## Notes
15. Smoke sampling groups by `id.rsplit("|",1)[0]`, which gives 10 groups: G2 gets 9 items (3 subs) and G3 gets 3. The docstring
   still says "--n 24 ... 8 family x cell groups" (stale). If n < 10, `sample(g, 0)` silently returns nothing.
16. Prompts are long (HN map prompt ~4.4k chars, CICM ~4k), but 40 output tokens is enough because the answer comes first. TOKBUDGET
   12000 / BS 32 is fine. A single item longer than the budget still runs alone.
17. OVERALL counts only models where P3 is testable. A model with P1 PASS and P3 n/a adds to neither count, as intended. Make sure
   the report prints the P1 result per model, even for models that do not count.
18. P1's "G4 <= 30%" is a point estimate and the "+20" is a CI bound. Mixing the two is acceptable but should be stated.
