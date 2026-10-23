# Mirror test — PLAN v1 (2026-10-19). Alex said yes (~$1).

Question (Alex): if the model is SHOWN its own lamp reading, does it regulate itself better?

Model Qwen2.5-14B-Instruct, H100, bf16, forced choice exactly as in gurdjieff/modal_g.py (DIRECT prefix chat(user + tail) + "The answer is",
options " "+c, summed log-prob, each option its own pass). Only the user text changes.

Items: own families where a lamp exists (lamps/rezultaty.md): R (1026 items: traps, cfright, compute, agree) and D (373: traps, agree, plain).
Saved DIRECT answers = BASE arm (sanity: recomputed BASE pick == saved pick >= 98% on the no-line pass, see below).

Lamp (frozen before the GPU run): per family, logistic regression on the saved layer-32 snapshot (C=0.003, balanced), label y = THINK != DIRECT,
5-fold cross-validation grouped by cluster -> every item gets an out-of-fold score. The lamp FIRES on the top 30% of scores within the family
(label-free quantile). Saved to mirror/lamp_scores.json with sha256 in PREREG.

Lines (inserted between the task text and the reply tail):
- MIRROR line: "Note from your internal sensor: right now, what is written in this task conflicts with what you already know. Notice this before you answer."
- GENERIC line (same length, no sensor content): "Note: please read this task carefully and take a moment before you answer, then give your answer as requested."
GPU passes: every item x {no line (sanity), MIRROR line, GENERIC line} = 3 x 1399 items.

Arms (assembled offline from those passes):
- BASE: no line anywhere.
- MIRROR: MIRROR line only where the lamp fires.
- RANDOM: MIRROR line on a random 30% of each family (seed 0), same count as the lamp.
- GENERIC: GENERIC line where the lamp fires.
- ALL: MIRROR line on every item (report-only: is the line itself helpful/harmful).
- Reference (saved, no GPU): THINK where the lamp fires.

Populations, weighted equally: traps = R/traps, D/traps; harm = R/cfright, R/agree, D/agree, D/plain; R/compute reported.
Criteria (bootstrap item + cluster within population, wider interval):
- M1 MIRROR - BASE on traps > 0, low > 0.
- M2 MIRROR - RANDOM on traps >= +3 points, low > 0 (where the line goes matters).
- M3 MIRROR - GENERIC on traps >= +3 points, low > 0 (what the line says matters).
- M4 every harm population MIRROR - BASE >= -3 points.
Verdict "the mirror helps" if M1-M4 pass. Prediction: M1 passes (any instruction helped before: FACTS +6.5), M3 fails or small, M4 R/cfright risk.
Caveat stated: lamp is post hoc and only exists on our task types; this does not test other tasks.

# PLAN v2 (after review/plan-review.md; overrides v1)
Passes on GPU: every R/D item x lines {NONE, MIRROR, HINT, GENERIC} (lines.json), line placed in the USER turn between the task text and
"\n\nReply with only the answer, nothing else.", separated by a blank line. ~5600 forced-choice items, est. ~$1-1.5.
BASE = recomputed NONE pass. Abort (no verdict) if NONE pick != saved DIRECT pick on > 2% of a family (exact ties excluded).
Clusters: R = arithmetic expression key (trap and compute twins together; plain numbers keep their own item), D = concept (hook).
Used for the lamp CV and every bootstrap. Lamp: per family, 5-fold GroupKFold, layer 32, C=0.003, y = THINK != DIRECT; fires on the
top 30% of out-of-fold scores within the family. lamp_scores.json frozen with sha256 before the GPU pass.
Placements (MIRROR line unless named; same number of lines per family as the lamp):
- MIRROR: lamp items. HINTL: HINT line on lamp items. GENERIC: GENERIC line on lamp items. ALL: MIRROR line on every item.
- RANDOM-S: random items stratified by population with the lamp's per-population counts (expected value over placements, exact).
- TEMPLATE: items of the cells with the highest leave-one-out cell flip rate. CONF: items with the lowest BASE margin.
Populations: traps = R/traps, D/traps (equal weight). Harm = R/cfright, R/agree, R/compute, D/no-conflict (D/agree + D/plain).
Criteria (bootstrap: items and clusters within population, 2000 draws, wider interval):
- M1 MIRROR - BASE on traps > 0, low > 0.
- M2 MIRROR - best of {RANDOM-S, TEMPLATE, CONF} on traps >= +3, low > 0 (the lamp places the line better than simple rules).
- M3 MIRROR - HINTL on traps >= +3, low > 0 (sensor framing matters). If |MIRROR - HINTL| < 3 points: "the content of the hint does the work,
  the sensor framing adds nothing".
- M5 MIRROR >= ALL on traps + harm populations together (equal weight), point estimate; difference and CI reported.
- Harm: every harm population MIRROR - BASE low >= -5; ALL - BASE per harm population reported with CI; counts of lamp firings per population.
Verdict "the mirror helps": M1, M2, M3, M5 and harm all pass. Separately reported: does HINTL help (HINTL - BASE on traps).
Reported separately: R/cfright T0 (where THINK is wrong), R/traps T7 (no flips), each population next to pooled numbers.
