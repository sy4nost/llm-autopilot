# many-instr/mix: review of the plan before the main run (2026-10-24)
Read: PLAN.md, make_mix.py, sets_mix.json, modal_mix.py, score_mix.py, ../review/plan-review.md. I tested score_mix.cls on made-up answers in a
scratch copy. I did not run Modal or look at any outputs. Checked and OK: 1600 items (100 N0 + 3 arms x 5 N x 100); the question list per
(fact, seed) is identical across arms, and N is a strict prefix (0 mismatches); arm values are exchangeable draws from one pool; no digits
or "one" in test questions; bank answers never equal a drawn value or M+-1; "one" is no longer parsed; bootstrap drops NaN resamples.
Verdict: the design is sound. Fix R1 to R3 before the paid run. They are scoring and decision rules, so no regeneration is needed.

## REQUIRED
R1 `multi` counts as NOT last in P1 and as NOT ex in P2, and the multi rule fires on answers that clearly commit. Checked:
   "Following your latest correction: 14 (not 9)." -> multi, and "Answer: 14\n\nNote: earlier answers used 9 by mistake." -> multi.
   At N=0 the sibling run had ~0% multi, but shown examples invite "not 9" asides. So P1 can "hold" just because the model mentions the
   example. Fix (pre-register it): score the committed value, i.e. the number after "answer is/Answer:/:", else the LAST candidate number.
   Use multi only when no commitment can be found. Also report P1 with multi/other removed from the denominator.
R2 The first-number rule lets a non-candidate number decide the label. "A spider has 8 legs, but per your instruction: 14" -> memory;
   "Based on your 5 corrections, the answer is 14." -> other; "1. 14" -> other. Fix: same committed-value rule as R1. At minimum, prefer
   candidates (vals + ex) over other numbers, and label "M stated + candidate committed" by the candidate, reporting its share.
R3 The headline metric does not match the question. "Example beats the last instruction" is ex% > last%, but the code reports "first N with
   last% < 50", which multi/other/old can trigger with no example win. Report the first N where ex% > last% (point estimate, plus the
   paired ex-minus-last bootstrap CI per arm). Also missing: an across-model rule for P2 (e.g. "P2 holds if it holds in >= 2 of 3 models"),
   and a floor/ceiling clause. If ex% is ~100% (or ~0%) in all arms at N=1..3, P2 is "untestable", not "fails". Note that boot() gives a
   two-sided 95% percentile CI, so the "one-sided" test is really at 2.5%. That is fine, but say so.

## OPTIONAL / a skeptic would ask
O1 P2 baseline: in E1/E4, "ex" equals an old instruction, so a spontaneous slip to vals[0]/vals[3] (present at N=0, absent for x in EX) counts
   as copying. Subtract the N=0 rate of answering vals[0] / vals[3] from the E1/E4 ex% (small if N=0 is ~100% last, as in the sibling run).
O2 Instruction 1 is the only one worded without "Correction" (sibling R4). P2 compares each arm with EX, so it is unaffected, but do not read
   E1 vs E4 as primacy vs recency.
O3 PLAN says same-topic bank questions are excluded, but TOPIC only covers week/year. The hexagon fact gets polygon-side questions (square,
   octagon, decagon ...; 4/20 seeds at N=3), and spider gets dog/insect/starfish legs. Arms are unaffected (same questions in every arm), but
   fact-level curves vary with it. Either fix PLAN's wording or extend TOPIC (and regenerate before the run).
O4 Spider M=8 vs 6 (insect legs) is a plausible memory trap; 6 is allowed in the pool. Flag answers of 6 for spider (sibling O2).
O5 score_mix: `base[(fact, seed)]` raises KeyError on partial outputs. Use .get and skip the cluster. P1 at N=3 uses only 1 item per cluster
   per arm (n=100). That is fine, but also show P1 at N=1 and N=2, because the prior expects the drop before N=3.
O6 No smoke outputs are in the folder yet. Fetch the smoke run and read ~20 answers per model (style: bare number vs recap) before main.
   The R1 rule depends on it.
O7 Batched, left-padded bf16 greedy generation can differ slightly from unbatched runs (same caveat as the sibling run).
