# Lamp = "the answer flips in the last layers" — PLAN v1 (2026-10-24), exploratory

Alex 2026-10-24 02:38-02:40Z: isn't "the old answer overtakes the right one in the last layers" itself the lamp? Go.
Earlier: only seen on HOLD/PASS (v3-shift/inside, Qwen 14B); a "layer-32 divergence" signal there fired 76-83% everywhere,
incl. right answers (damp test, 2026-10-16). Here: does a late flip separate fast habit errors from fast right answers on the
five confident-trap families and three models, better than low confidence?

## Data
conftrap-lamp items (seed 57, ../conftrap-lamp/sets_cl.json) and saved DIRECT answers (out_main_*.jsonl): hindgen, redef,
parcels, bonus, D (traps + controls). Models Qwen2.5-14B, Phi-4, OLMo-2-13B. Reused items: exploratory.

## Measure (one forward pass per item, the exact DIRECT prefix of modal_cl.py)
Logit lens at the answer position: for every layer l, logits = lm_head(final_norm(h_l)); last layer = the real logits.
Per option one distinguishing token: the first token of the option string as DIRECT scored it (" "+c, or c for label items);
if all options share that first token (e.g. a bare " " before digits), that token is appended to the prefix and the options'
second tokens are used; items whose distinguishing tokens still coincide are dropped (count reported).
leader(l) = option with the highest lens logit among the options at layer l. Final leader = leader at the last layer.
Scores per item (depth d = l / (L-1)):
- SETTLE (primary) = the largest d at which leader(l) != final leader, over layers with d >= 0.5 (0 if the final leader leads
  on all of them). High = the answer was decided late (a late flip).
- FLIP75 (report) = 1 if some layer with 0.5 <= d < 0.75 has a leader different from the final one.
- LEADMID (report) = lens margin of the final answer at d = 2/3 (low = it was not leading in the middle).
Rival CONF = 1 - normalized probability of the fast answer (from saved DIRECT scores); also LEN.

## Questions (as surprise-lamp v2, same code paths)
Q1 (primary). Traps only: fast habit error (+) vs fast right (-), within-template AUC (pairs from the same template:
conftrap cluster; hindgen by key), eligible cell >= 30 pairs and >= 2 templates. "Promising" needs ALL: mean |AUC-0.5| of
SETTLE >= 0.12; beats CONF by >= 0.05 with clustered bootstrap low > 0; beats LEN likewise; same direction in >= 2/3 of cells
and in >= 2 models (OLMo only with >= 2 eligible cells). Expected direction: habit error -> later settle (AUC > 0.5).
Q2 (report). Fire rate of "late flip" (SETTLE >= 0.75) on traps-habit-error / traps-right / controls, per model x family:
the "smoke alarm" view Alex asked about.
Q3 (report). Mean leader curve by depth for habit errors vs right answers: does the old answer take over late (like inside/)?

## Anti-fitting
Formulas, the 0.5 / 0.75 / 2/3 depths and thresholds fixed here; scorer frozen with hashes before opening main outputs.
Prediction (Claude): Q1 "no lamp here"; late flips frequent on every population (alarm rings too often); maybe some signal in
parcels/bonus (the inside/ pattern), not in D/redef/hindgen.
Cost ~$0.5 (L40S, ~760 forward passes x 3 models).

# PLAN v2 (after review/plan-review.md; overrides v1). Written before opening any main output (the main GPU pass, which makes no
# choices, was launched with modal_lf.py sha in main_launch_hashes.txt; its per-layer option log-probs are all saved).
- B3: already as required: layers 1..L-1 through model.norm + lm_head, last layer = real logits (smoke: max diff <= 0.0005).
  d = l/(L-1), l = 0 is layer 1.
- B1 token prior: before taking leaders, each option's lens log-prob at each layer (except the last) minus the mean for that token
  in that model x family over items of OTHER clusters. MAIN = SETTLE on the debiased lens. New rival TOKPRIOR (mean SETTLE of
  other-cluster items whose fast answer is the same token); SETTLE_RAW (no debias) report-only.
- B2 ties: a tie with the final answer counts as "final leads".
- S1: rival FINMARG (final lens margin, = confidence on the token) added; report SETTLE_C = SETTLE with CONF ranks regressed out
  within model x family x pop. "Promising" needs MAIN to beat EACH of CONF, LEN, TOKPRIOR, FINMARG by >= 0.05 (clustered bootstrap
  low > 0), strength >= 0.12, direction in >= 2/3 of cells and >= 2 models. Verdict printed with and without hindgen.
- S2: FRAC (share of upper-half layers where the final answer does not lead) report-only; SETTLE distribution reported.
- S3 deviation: the GPU pass dropped every item whose distinguishing tokens coincide (any pair), e.g. 9 Qwen D traps where "13"/"1"
  clash with the third option; counts printed. Not rerun.
