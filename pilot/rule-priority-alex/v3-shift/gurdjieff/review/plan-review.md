# Review of gurdjieff/PLAN.md v1 (2026-10-19), before any paid run

Reviewer: independent auditor (Claude). Everything below was computed with `python3 -I` from the saved outputs
(thinkaloud/out_main.jsonl, thinkaloud-indep/out_main.jsonl) and the sets files, using the frozen answer rules
(own: forced choice for THINK, as score_ta.py; public: parsed "Answer:" line with forced fallback, as score_ti.py).
Nothing from the outputs was executed (`res` decoded with `ast.literal_eval`). Scratch scripts:
/tmp/claude-0/-home-user/1814f85d-e03a-5c27-8b66-9c6b11e6a758/scratchpad/g/{load,ev,sim}.py.

"Rival gates" below were simulated on the saved data exactly as the plan describes rivals: score fitted/ranked on the
training pool, threshold = 70th percentile of training scores, applied to the test pool, families weighted equally
(own families R, N, K, D with all their populations; public families = the 10 tasks including MMLU).

**Verdict: not ready to run. 6 blockers; most are fixes to the criteria/analysis text and the extraction spec, costing nothing.**

---

## Key numbers

### y = (THINK != DIRECT) per population

| own pool (y 24.0%, 494 items) | n | y | y rate | share of all y | THINK fixes / breaks |
|---|---|---|---|---|---|
| R-traps | 378 | 170 | 0.45 | 34% | 169 / 1 |
| N-traps | 120 | 110 | 0.92 | 22% | 110 / 0 |
| R-compute | 354 | 60 | 0.17 | 12% | 59 / 1 |
| K-traps | 242 | 60 | 0.25 | 12% | 28 / 32 |
| D-traps | 258 | 56 | 0.22 | 11% | 56 / 0 |
| R-cfright | 210 | 35 | 0.17 | 7% | 0 / 35 |
| N-confirm, K-plain | 82 | 3 | | 1% | 3 / 0 |
| R-agree, N-agree, K-agree, D-agree, D-plain | 416 | 0 | 0 | 0 | |

| public pool (y 15.7%, 361 items) | n | y | y rate | share of all y | THINK fixes / breaks |
|---|---|---|---|---|---|
| mmlu | 1000 | 160 | 0.16 | 44% | 74 / 56 |
| hind-trap | 120 | 75 | 0.63 | 21% | 67 / 8 |
| pattern-trap | 100 | 36 | 0.36 | 10% | 15 / 21 |
| redef-trap | 100 | 24 | 0.24 | 7% | 16 / 8 |
| memo-trap | 120 | 20 | 0.17 | 6% | 19 / 1 |
| neqa-trap / neqa-control | 120 / 80 | 16 / 7 | | 6% | 14 / 9 |
| unknown-trap | 100 | 14 | 0.14 | 4% | 12 / 2 |
| quote, modus | 200 | 9 | | 2% | 3 / 6 |
| alg (trap+control), hind-control, memo-control | 360 | 0 | 0 | 0 | |

- Of all y, THINK actually fixes DIRECT in 425/494 (own) but only 220/361 (public): 31% of the public y are THINK breaking a right answer or both wrong (MMLU alone: 56 breaks).
- Under family-equal weighting the whole THINK gain is small and concentrated: own gap (always-THINK minus DIRECT) = +17.5 points, of which N is 53%, R 27%, D 21%, K -1%. Public gap = **+5.2 points**, of which hindsight is 57% (memo 17%, unknown 19%, redef 15%; pattern, modus, quote negative).
- The oracle needs only 20.6% (own) / 9.6% (public) think share.

### Trivial features already explain y within a pool
- Pooled AUC for y from **population identity alone**: own 0.807, public 0.736. From own **template/hook cell identity**: 0.925.
- Prompt length: own N prompts are ~1,000 chars vs <= 106 for R/K/D; public hindsight ~2,700-2,900 chars vs < 700 for most others. So "long prompt" is "N family" in own and "hindsight" in public: the two families that carry most of the THINK gain.
- Inside the two gain-carrying families, y errors are **confident**: within-population AUC of low DIRECT margin for y = 0.24 on N-traps and 0.28 on hind-trap (i.e. inverted). N-traps has 110 y vs 10 non-y: there is essentially nothing to discriminate inside it; catching it = recognising the family.

### Simulated rival gates (plan's own rules; no new data)

| split | gate | test think share (train-threshold) | acc(gate) - acc(DIRECT) | R | at a 30% test-quantile threshold: dD / R |
|---|---|---|---|---|---|
| S1 own->public | low-confidence (-margin) | 0.29 | +3.6 | 0.69 | +3.7 / 0.70 |
| S1 | prompt length | **0.86** | +3.5 | 0.68 | +3.6 / **0.69** |
| S1 | surface LR (len, "not", digits, "Answer") | 0.18 | -0.9 | -0.17 | +1.1 / 0.21 |
| S1 | random 30% | 0.30 | +1.6 | 0.30 | |
| S2 public->own | low-confidence | 0.31 | +11.5 | 0.66 | +11.3 / 0.64 |
| S2 | prompt length | **0.15** | +9.3 | 0.53 | +9.5 / **0.54** |
| S2 | surface LR | **0.00** | 0 | 0 | +9.2 / 0.52 |
| S2 | random 30% | 0.30 | +5.3 | 0.30 | |

Bootstrap (items within family, 1000): R of the low-confidence gate S1 [0.51, 0.93], S2 [0.57, 0.71]; paired difference of
two 30%-share gates (margin vs length) S1 [-1.5, +1.7] points, S2 [+0.2, +3.4].

---

## BLOCKERS

### B1. G1 and G3 do not test the sensor; prompt length and low confidence already pass them in both splits
- G1 (gain > 0) is passed by a **random** 30% gate (S1 +1.6, S2 +5.3 points): every y=0 item is unaffected by thinking, and always-THINK beats DIRECT in both pools, so any gate that thinks on anything gains.
- G3 (R >= 0.5 at share <= 40%) is passed by the low-confidence rival (R 0.69 / 0.66) **and by "think if the prompt is long"** (R 0.69 / 0.54 at a 30% test share). So a G3 pass is fully explainable by a trivial feature that a residual-stream probe encodes trivially (length, position, family format).
- Only G2 discriminates. As written, the verdict "sensor works" rests on G2 alone, and the surface rival in G2 is too weak (see B3).
- **Fix:** keep G1/G3 as report-only, or restate them as margins over the strongest trivial rival (low-confidence and a length-only gate), at the same actual test share. Add a within-family criterion (B2).

### B2. No within-family criterion; gate accuracy can be won by recognising families/templates
- y is concentrated by family (population-identity AUC 0.81 / 0.74; template-cell AUC 0.93 in own). Within the pool, any probe that encodes "this is an N/hindsight-style prompt" earns most of the gain. Across pools family identity cannot be copied, but generic format cues can (long prompt, many in-context examples, rule lists, "Answer:" scaffolding): length does exactly this (B1).
- The plan lists "AUC of sensor for y within family" as a metric, but no criterion uses it. general-lamp already established (its plan v2, point 2) that only within-cell measures are meaningful because cell identity alone gives 0.87-0.98.
- Gain-carrying families where within-family discrimination is measurable (>= 10 y and >= 10 non-y): public hind-trap (75/45), memo-trap (20/100), pattern (36/64), redef (24/76), unknown (14/86), neqa-trap (16/104), MMLU (160/840); own R-traps (170/208), D-traps (56/202), K-traps (60/182), R-compute (60/294), R-cfright (35/175). Own N-traps (110/10) is not measurable.
- **Fix:** add a pre-registered criterion: mean within-population AUC of the sensor for y over test populations with >= 10 y and >= 10 non-y >= the low-confidence rival + 0.05 with low > 0 (cluster bootstrap), plus a within-family gate result (threshold applied, gain computed after removing between-family differences, e.g. family-demeaned scores, or "gain over a family-oracle gate that thinks on whole families"). Report a "family-identity gate" (think on whole families/populations ranked by training y-rate where they exist, or by length) as a named rival.

### B3. Rivals are not compared at the same actual think share; threshold transfer is arbitrary under pool shift
- The plan recomputes each rival's threshold "on the training pool the same way", so actual test shares differ wildly: length 0.86 (S1) vs 0.15 (S2); surface LR 0.18 vs **0.00**. The margin gate transfers well (0.29 / 0.31) because log-probs are calibrated; standardised residual-stream scores are not, and own vs public prompts differ by an order of magnitude in length and format.
- Consequences: (a) G3's 40% cap can fail or pass on calibration luck, not detection quality; (b) G2 compares the sensor with rivals at different shares, which is not "the SAME thinking share" the plan promises; (c) the surface rival collapses to 0% share in S2 and becomes a straw man, so "sensor >= surface" is near-automatic.
- Base rates of y differ (own 24.0%, public 15.7%, public without MMLU 15.5%). This by itself is not the problem: on y=0 items thinking changes nothing, so a share above the y rate only costs compute. The problem is purely score-distribution shift moving the actual share.
- **Fix:** primary comparison at a **label-free test-quantile threshold** (each gate thinks on exactly 30% of test items; uses no labels, so it is legitimate) and, better, the gain-vs-share curve over shares 10-40% (area) for sensor and every rival on the same grid. Report the train-transferred threshold and its actual share as a secondary "calibration transfer" result. Include length-only (both signs, sign fixed on training) as a separate surface rival.

### B4. Leakage: public redef uses the same templates as own R
- All 9 public redef templates ("Redefine π as #+#", "# mod #", "Redefine π as #", "Redefine e as #+#", "View + as the digit #", "View + / ÷ as a random digit…", "View number as text…") occur in own R (R-traps/cfright/agree/compute were built by copying redefine-math templates). In S1 the redef test family (1/10 weight, 24 y) is a seen template; in S2 own R (1/4 weight, 34% of own y) is tested after training on public redef.
- Removing redef barely moves the trivial rivals (S1 margin dD +3.6 -> +3.3), so it costs little.
- **Fix:** drop public redef from both training and testing in S1 and S2 (keep it report-only), and say so in PREREG. Also report S2 without own R as a sensitivity.

### B5. DIRECT-prefix reconstruction and read-out position are under-specified and differ across families
- Own N: `score(chat(text), cands)` - no R_TAIL, no "The answer is", options **without** a leading space; prefix ends at `<|im_start|>assistant\n`. Own R/K/D: `chat(text + R_TAIL) + "The answer is"`, options `" " + c`. Public: `chat(text + R_TAIL) + (stem.strip() or "The answer is")`, options `" " + c` (memo/quote/pattern end on the stem's last word). The plan's one-line spec ("same chat template, tail, 'The answer is'/stem") omits the N exception; code written from the plan would get N wrong (y for N is 22% of own y).
- The last-prefix token therefore differs by family ("\n" after assistant, " is", a stem word). Token identity at the read-out position is itself a family signature, and own N's position type never occurs in public training (S2 reads N out of distribution; S1 trains on it).
- **Digit answers: the first option token is the space.** In Qwen2.5 `" 3"` tokenises as `" "` + `"3"`: saved DIRECT scores have two tokens per option and identical first-token log-probs for all 1,026 own R and 373 own D items, and for public alg (195/200) and redef (100/100). So for 68% of own items (R+D) and two public families: (a) "final-layer logits over the options' first tokens" are identical across options, making the output-only sensor blind there; (b) the residual at the last prefix token is one position before the token where the answer is decided.
- **Fix:** specify the prefix per family by copying `direct()` from each modal script verbatim (N included). Append the options' longest common token prefix (e.g. the space) to the prefix and read the state and logits at the position that predicts the first *distinguishing* token. Report a sensitivity reading at one common position for all items (the final token of `<|im_start|>assistant\n`, before any prefill). Build the output-only sensor from the saved full-option log-prob sums/margins, not first-token logits.

### B6. The sanity check is weak and would not catch the errors above
- On the saved data, first-token argmax vs saved DIRECT pick (items with distinct first-token log-probs): own N 297/297, K 358/358; public 1,927/1,962 = 98.2%, but **memo only 164/197 = 83%** (multi-token words). A global ">= 90%" passes while a whole family is off, and R, D, alg, redef are not checked at all (shared first token).
- The saved `res` already holds the exact log-prob of every option token under the DIRECT prefix. A same-model forward pass of the correct prefix must reproduce them up to bf16 noise.
- **Fix:** sanity = recomputed log-prob of each option's first distinguishing token vs the saved value, per family: median |diff| < 0.05 and >= 98% of items within 0.2, and argmax agreement >= 98% **per family**; stop if any family fails. Run without padding (or with checked attention masks), batch size 1, as in the original `score()`.

---

## SUGGESTIONS

S1. **Harm (G4) is measurable almost nowhere and is pooled away.**
- y = 0 on 416 of the 498 own control items outside R-cfright/R-compute (the other 82, N-confirm and K-plain, have 3 y) and on memo/hind/alg controls: a gate cannot harm there. Harm is only measurable on own R-cfright (35 harmful y), MMLU (56 harmful y) and neqa-control (4).
- Pooling hides it. In S2 the margin gate gives R-cfright **-6.2** while R-compute gives +13.8. R-compute is an arithmetic-confound population where THINK helps; it is not a harm control.
- S2 has no MMLU at all.
- Fix: G4 per population (each >= -2 point, and low >= -5 or similar) on R-cfright, MMLU and neqa-control; exclude R-compute from harm; state that S2 harm rests on R-cfright only; report the gate's fire rate on MMLU where DIRECT is right.

S2. **The teacher label includes harmful disagreements.** 31% of public y are THINK breaking or both wrong (MMLU 56 breaks, pattern 21), so a perfect y-detector is not a good gate there. Training S2 on public y means 44% of the positives are MMLU, which pushes the probe toward generic uncertainty, i.e. toward the low-confidence rival. Report (diagnostic only) the gate with y+ = (THINK right and DIRECT wrong) as the label, and S2 trained without MMLU.

S3. **G2 feasibility.** Low-confidence already gets +3.6 (S1) and +11.5 (S2); G2 needs +6.6 and +14.5. The ceilings are the oracle +10.3 (public) and +20.7 (own); always-THINK is only +5.2 in public. In S1 the sensor must therefore beat always-THINK by about 1.4 points at 30% share, within paired CIs of about ±1.6 points. That is fine as a strict test, but say in the prediction that G2 in S1 requires near-oracle selection. Also note why: the gain-carrying errors (hindsight, N) are confident, so low confidence cannot catch them while a format/length cue can (B1/B2).

S4. **The R ratio is unstable in S1** (denominator 5.2 points; margin-gate R CI [0.51, 0.93]). Prefer acc(gate) - acc(DIRECT) with a CI as the primary quantity, with R reported. Or require R's lower bound >= 0.5.

S5. **Inner CV must be grouped by family** (leave-one-family-out within the training pool) for both layer and C. Random k-fold rewards layers that memorise family/template, which is exactly what does not transfer. Report the CV AUC by layer.

S6. **Define the populations precisely.**
- Which public families enter S1-test and S2-train: are the report-only modus/quote/pattern included, and is MMLU one family of 10? This changes the gap (public gap 5.2 with all 10).
- Own: the 4 families with all populations pooled?
- Bootstrap clusters: own as in score_ta.py (R by template with twins, N by set, K/D by hook); public by the `cluster` field.
- State that the S2 y uses forced choice for own THINK and parsed THINK for public (they differ on 22 public items, 10 of them memo).

S7. **Save per-item sensor scores locally** (a few hundred KB) so the analysis can be recomputed, as general-lamp did. "Only reports come back" prevents audit.

S8. **Label noise.** The public THINK generations came from a resumed run with batch drift (smoke: 15/20 batched vs single identical last line). y has some noise; mention it. This is not a blocker.

S9. **Prior result to state in the prediction.** general-lamp found that confident habit errors on road words defeat probes (margin 0.89 there vs probe 0.59), and that a C->P probe on redefine scored 0.51. The present data show the same pattern (margin AUC 0.24 / 0.28 inside N-traps and hind-trap). Write the expectation per family, not only per split.

S10. **Cost estimate.** The plan's estimate is plausible: about 4,360 single forward passes of 8 layers x 5,120 bf16 values is about 360 MB. Cap the Modal function timeout as before.

---

## What a pass would mean if the plan were run unchanged
A G1+G3 pass is expected from low confidence or length alone. A G2 pass could come from rivals evaluated at mismatched shares (surface at 0% share in S2) and from the redef/R template overlap. Read-out positions differing by family (B5) could make the probe a format detector. After B1-B6, a pass would mean: at equal think share and within families, the internal state picks the THINK-worthy items better than the model's own confidence and better than format cues, on a pool the probe never saw.

---

## v2 check (PLAN v2, 2026-10-19)

Blocker status:
- **B1 fixed.** G1/G3 are out of the verdict. The recovered share R is report-only. The accuracy gain is the main number.
- **B2 fixed at population level.** See N1 for the part that remains.
- **B3 fixed.** All gates use the same 30% test quantile, and the 10-40% curve is reported.
- **B4 fixed.** Public redef is dropped from both splits.
- **B5 fixed.** The prefixes are copied verbatim from each run. The read-out is at the first token where the options differ. The output-only sensor is read at that same position.
- **B6 fixed.** The sanity check is now per family, with >= 98% agreement, on full-pass forced-choice picks. Report-only suggestion: also print the median |recomputed - saved| log-prob.

### Is P1's population filter workable? Yes.
Populations with >= 10 y and >= 10 non-y:
- **S1 test (public, no redef), 6 populations:** hind-trap 75/45, memo-trap 20/100, MMLU 160/840, neqa-trap 16/104, pattern 36/64, unknown 14/86. neqa-control (7 y) is out.
- **S2 test (own), 6 populations:** R-traps 170/208, R-compute 60/294, K-traps 60/182, D-traps 56/202, R-cfright 35/175, N-traps 110/**10** (exactly on the limit, so very noisy).
- **Low-confidence rival, mean within-population AUC:** S1 0.653, S2 0.638.
  - These are inverted on hind-trap (0.28) and N-traps (0.24).
  - If the rival's sign could be chosen per test population, the means would rise to 0.736 and 0.725.
  - The plan should state that each rival's sign is fixed on the training pool, as for the sensor.
- **Bar for the sensor:** about >= 0.70 and >= 0.70 (rival 0.65 + 0.05), and low > 0. That is strict but attainable.

### Is P2's label-free 30% test quantile sound? Yes.
It uses no labels, so it is legitimate. Two caveats:
- **It is transductive.** The gate needs to see the test batch's score distribution. The verdict should say so: "works at a calibrated share", not "works with a transferred threshold". Keep the transferred-threshold share as a report-only line.
- **Specify a tie-break.** Length, surface counts and margins near 0 have ties. To make each gate think on exactly 30%, break ties at random with a fixed seed, or think on all tied items and report the actual share.
- For P2, bootstrap the difference against the best rival with that rival re-chosen inside each resample, as thinkaloud's `best(s)` does.

### New issues
- **N1 (required fix to the plan text; costs nothing).** Within an own population, template or hook identity still predicts y:
  - R-traps AUC 0.85 (T3 0.16, T5 0.39, T6 0.90, T7 0.00)
  - R-cfright 0.96 (only T0 flips, 35/50)
  - R-compute 0.77, D-traps 0.88, K-traps 0.73
  - So an S2 pass of P1 could still be "recognises the T6 / T0 template". The risk is lower than within a pool, because public training has no redefine items now. Still, add a within-cell check on S2:
    - mean AUC over own cells with >= 5 y and >= 5 non-y (18 cells: 5 D hooks, 6 K countries, T0, C3, C5, C6, T3, T5, T6);
    - requirement: >= 0.60 and >= low-confidence within-cell, or else report "population-level only".
  - Public populations have no comparable templates, so this concerns S2 only.
- **N2 (note, not a blocker).** P1 rewards predicting all flips, including harmful ones: R-cfright y are all THINK breaks, as are about a third of MMLU y and 21/36 of pattern y. A sensor that is good on P1 will therefore fire on R-cfright and may fail P3 there. The cutoff is -3 points on 210 items, about 6 items; the margin gate already gives -6.2. This is the correct behaviour of the criteria, but write in the prediction that P1 and P3 can conflict in S2.
- **N3 (note).** P3 point-only on small populations is noise-prone: neqa-control (80 items, 4 harmful flips) and modus (2 harmful). Zero-flip populations (all agree/plain, memo/hind/alg controls) pass automatically. Say that the only informative P3 populations are R-cfright, MMLU, neqa-control, quote and modus.

**Verdict:** v2 fixes B1-B6. Add N1 (within-cell check on S2), the sign rule for rivals, and the tie-break rule, then it is ready for code.
