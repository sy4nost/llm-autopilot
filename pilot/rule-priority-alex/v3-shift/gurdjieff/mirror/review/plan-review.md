# Mirror PLAN v1: independent review (2026-10-19)
Read: mirror/PLAN.md, lamps/*, conflict/rezultaty.md, rezultaty.md, modal_g.py, make_g.py, items_g.json (R/D counts below), ../cycle, ../thinkaloud.
Facts from items_g.json: y is almost fully set by the template. R/traps T6 104/115 flips (DIRECT right on only 6), T5 47/120, T3 19/119, T7 0/24;
R/cfright: all 35 flips in T0, where THINK is the wrong one (DIRECT 47/50 right); R/compute C5 46/120; R/agree and D/agree/plain 0 flips.
R clusters are one item each (378 clusters for 378 traps), and 352 R/traps items share their expression with an R/compute twin ("Redefine π as 988+30" vs "988+30") in another cluster.
D has 15 concept clusters shared across traps/agree/plain (good).

## BLOCKERS
1. **MIRROR wording is the FACTS hint.** Saying "the text conflicts with what you know" on R/D traps means "follow the text". FACTS-style lines already took D 78->100, π 47->68 (thinkaloud), +6.5 over a filler line (cycle).
   GENERIC has no content, so M3 measures "a hint beats no hint" and says nothing about the sensor. **Fix:** add one GPU pass with HINT = same length, same content, no sensor framing
   ("Note: what is written in this task may conflict with what you already know. Notice this before you answer."). Put it on the lamp items. Rewrite M3 as MIRROR - HINT (sensor framing). Keep MIRROR - GENERIC as a check on the content.
   If MIRROR ≈ HINT, the result is "a conditional hint", not "self-regulation". Write that outcome into the plan now.
2. **RANDOM is too weak a placement baseline. M2 passes if the lamp only knows the template.** The lamp is trained on the whole family with y set by template, so it will mostly fire on T6/T5/C5/T0 and D/traps. Unstratified RANDOM puts about 70% of its lines on non-trap items, so M2 is close to guaranteed. **Fix:** every arm can be built offline from the ALL pass at no GPU cost.
   Add (a) a stratified RANDOM: within each population, the same count as the lamp; take the expectation over all draws (= BASE + k/n·(ALL−BASE) per population), not one seed;
   (b) a TEMPLATE-prior placement: fire on the cells with the highest out-of-fold flip rate, matched in count;
   (c) a CONF placement: top 30% by −margin, which beat the sensor in rezultaty.md.
   M2 should be "MIRROR beats the best of (stratified RANDOM, TEMPLATE, CONF) by ≥ 3, low > 0". Without this, a "pass" tells us nothing new.
3. **ALL is report-only, so "the lamp is worth having" is never tested.** If the line does no harm anywhere, ALL ≥ MIRROR and the lamp adds nothing. **Fix:** add M5: MIRROR ≥ ALL on the equal-weighted traps+harm score, or MIRROR beats ALL on harm with traps no worse than ALL − 3.
4. **M4 is diluted, so it cannot fail.** The lamp barely fires on R/agree and D/agree/plain (y = 0 there). A population-level MIRROR − BASE ≈ 0 by construction. D/plain has 27 items, so −3 points is 0.8 of an item: either it never moves, or one item fails it.
   **Fix:** judge harm on ALL − BASE per harm population, which is the line's real harm (the "conflict" claim is false on agree/plain/compute). Report MIRROR − BASE on the fired harm items with counts. Use a CI rule (low ≥ −5) instead of a point estimate. Merge D/agree+D/plain, or give a minimum number of items.
5. **Twin leakage in the R lamp CV.** With single-item clusters, a T3/T5/T6 trap and its C3/C5/C6 compute twin (same expression, correlated y: C5 vs T5 both hard) land in different folds. **Fix:** set the R cluster to the expression key (twins together) for both the CV and the bootstrap. Freeze lamp_scores.json and its sha256 before the GPU pass, as planned.

## MINOR
- BASE should be the recomputed no-line pass, not the saved picks: same code and hardware as the line passes. Keep ≥ 98% agreement per population as an abort gate, and write the abort action.
- The label includes harmful flips: on R/cfright T0, y = 1 means THINK is wrong. So the lamp is aimed at items where DIRECT is right. Say this in the plan, and report MIRROR on fired cfright-T0 items separately. The "THINK where lamp fires" reference will look bad there by design.
- R/traps T7 (24 items, DIRECT and THINK both 0/24 right) has no headroom. Report it, but it dilutes R/traps.
- Bootstrap: D has only 15 clusters, so its interval will be wide. That is fine for traps (expected effects +15–20), but write it down. For R, the "wider of item/cluster" rule is the same as item-level unless (5) is fixed.
- Equal weighting of R/traps and D/traps: D/traps may already be near ceiling once a hint is added (100 in thinkaloud), so R will decide the outcome. Report per population next to the pooled number.
- The line sits inside the user turn, between the task and R_TAIL. The cycle +12 "format effect" came from system text, so GENERIC controls for it only roughly. Add one sentence saying where the line goes, with its exact string.
- R/compute: the line falsely claims a conflict on fired compute items. Report this as a harm-type number, not just "reported".
- Cost: 4 passes x 1399 items x 2–3 options is still about $1. The added arms are offline. No reason to cut them.
