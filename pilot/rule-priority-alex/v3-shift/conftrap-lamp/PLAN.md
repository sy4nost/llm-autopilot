# Lamp on confident traps, 3 models — PLAN v1 (2026-10-23)

Alex: "OK, go ahead and launch it" (2026-10-23 21:58Z), answering the proposal of a frozen lamp test on the families found by the
confident-trap pilot (../confident-traps/rezultaty.md). Budget stated to Alex: ~$4-6 (Modal H100).

## Why this test
fastthink (../fastthink/rezultaty.md) failed H1 because its new public tests had almost no confident autopilot: fast errors were
unconfident, so "flag the least confident answers" (CONF) did as well as the lamp. On the old memo/hindsight items (report-only) the lamp
beat CONF by +36. The pilot found 5 families where all or most of the 3 models make CONFIDENT habit errors on fast answers
(wrong-answer confidence ~80-100%, controls 95-100% right). This test asks the fastthink H1 question on exactly those families,
on fresh items, with everything frozen before the run. The pilot looked only at fast answers; no lamp, no thinking was computed.

## Question (H1, the only verdict)
When the model's FAST answer and its THOUGHT answer differ (FLAG), does that point at fast-answer errors better than three cheap rivals
that flag the same number of items: RANDOM@r, CONF@r (smallest DIRECT margin = top-1 minus top-2 summed log-prob) and
SAMPLE@r (weight-free: 5 short fast answers sampled at temperature 1, ranked by how many differ from the DIRECT answer)?
Scope, said up front: FLAG needs the thought answer, so it saves no compute versus "always think". Its use is a behavioural autopilot
meter that works on any model (API too). On two-option families recall(FLAG) = P(thought right | fast wrong), so H1 is linked to
"thinking fixes the error"; H1 asks whether this signal finds fast errors better than cheap signals.

## Models (fixed)
Qwen/Qwen2.5-14B-Instruct, microsoft/phi-4, allenai/OLMo-2-1124-13B-Instruct; bf16, greedy, H100, own chat template.
A model is dropped only for a load or chat-template failure, never after seeing output.

## Families (fixed; chosen by the pilot from fast accuracy only)
Per family 100 traps (habit answer wrong) + 40 controls (habit answer right), one cell = model x family, traps and controls together.
- hindgen: hindsight-neglect, freshly generated with the published template (10 shots where outcome and expected value agree; trap =
  outcome contradicts expected value). Controls: outcome agrees with expected value.
- redef: published redefine-math items not used in thinkaloud-indep, fastthink or the pilot. 100 traps, no controls (dataset has none).
  Earlier 14B runs (public/, conflict/, watch_public/) saved only row numbers, so some items may repeat those runs; nothing in this test
  is fit on items, so reuse cannot favour any method.
- parcels, bonus: newtasks2 generator, arm A60 (60 examples under the old rule, then the amendment note). Traps = probes T1/T2 (old rule
  fires, new does not), controls = S1/S2 (change does not touch them). Generator seeds SEED*10+0..2. Note (audit): the parcels trap
  cases are two fixed case types per set; clusters = one set's shared history.
- D: stated definitions contradicting the real count ("In this world, a 'cube' has 7 faces. How many faces...?"), thinkaloud generator,
  all 22 hooks. Controls: stated count = real count.
- mmlu: 200 questions unused by earlier runs, report-only (false alarms, harm).
Not used: R (our pi/e) duplicates redef templates; N, K, memo had little confident habit in the pilot.
No item of the pilot or of the smoke is reused (asserted by prompt).

## Arms per item (identical to fastthink except the label families)
DIRECT: chat(task + "\n\nReply with only the answer, nothing else.") + "The answer is", options " "+c, summed log-prob, argmax.
  parcels/bonus prompts already end with "Return only the answer." -> chat(task) and options c (as thinkaloud N).
THINK: chat(task without "Do not explain / Return only" lines + "\n\nThink it through step by step. Then give your final answer on the
  last line as: Answer: <answer>"), greedy, 512 new tokens; answer = the one option named on the last "Answer:" line, else forced
  choice after "Answer:".
SAMPLE: 5 samples, temperature 1, 24 new tokens, from the DIRECT prefix; unparsed sample = differs.

## Measures and criterion (as fastthink H1)
Fast error E = DIRECT wrong. FLAG = THINK != DIRECT. r = flag rate of the cell. Recall = share of E flagged.
Eligible cell: E >= 15 and non-E >= 15. Need >= 4 eligible cells of 15, else "not testable".
H1: for EACH rival, mean over eligible cells of recall(FLAG) - recall(rival) >= +10 points, bootstrap 2000 (item and cluster schemes,
one draw per family shared by models, wider 95% interval) low > 0, and per-model mean > 0 in >= 2 of 3 models.
Readout sensitivity: H1 recomputed with DIRECT length-normalised and THINK by forced choice; the verdict stands only if every rival's
difference keeps its sign in both.
Verdict: "lamp confirmed on confident traps" if H1 passes and signs hold; otherwise "not confirmed".
Report-only: THINK - DIRECT on traps per family; controls fast vs thought (harm where habit is right); MMLU flag rate and recall;
predicted-class prior rival; CONF AUC; think only on the 30% least confident; disagreement direction.

## Threats named before the run
- Trap-ness is visible from the text to anyone who knows the real fact; the test is about finding a model's ERRORS with model-only signals,
  so all rivals use only the model's outputs.
- Families were chosen because fast errors were confident; that favours FLAG over CONF by design. This is the stated question
  (does the lamp work where confidence is blind), not a hidden advantage. RANDOM and SAMPLE are not affected by that choice.
- One set of prompts and one THINK wording; thresholds (+10) carried over from fastthink, not tuned.

## Anti-fitting
Families, models, prompts, parse rules, counts and criteria fixed here before any THINK output on these families. Smoke (seed 12345,
32 items) prints pipeline checks only. PREREG.md with sha256 of make/modal/score and the items file is frozen before the main run;
Alex gives the seed. Nothing is added or removed after the main run; later analysis is labelled post hoc.
Prediction (mine): FLAG beats RANDOM and CONF clearly; SAMPLE is the risk (sampled answers may also wobble on habit items).

# PLAN v2 (after review/plan-code-review.md; overrides v1 where they differ). Written before any main-run output.
B1 clusters: v1's clusters held one item each. Now: parcels by test case (the traps are only 2 fixed case strings), bonus by
  (generator seed, set), D by hook, redef by template (prompt before "Q:" with digits -> N), hindgen and mmlu by item.
  New REQUIRED gate: leave-one-family-out. With each family dropped, every rival's point difference FLAG - rival must stay > 0.
  Verdict = H1 pass AND readout signs hold AND leave-one-family-out holds.
S1 off-list THINK answers: a final Answer line that names no option at all (e.g. "VAN" when the options are AIR/TRUCK/SHIP, "8" when
  they are 2/7) counts as THINK = "other": it differs from the fast answer (FLAG) and is wrong. The forced-choice readout stays as the
  sensitivity (THINK_forced). Off-list rate printed per cell. (Same treatment as SAMPLE, where an unparsed sample counts as differing.)
S2 redef: the 200 rows used by public/ and conflict/ (sorted(shuffle(range(900), Random(0))[:200])) are excluded too.
S3 SAMPLE: number words are read as digits; sampling uses repetition_penalty 1.0 for every model; report-only SAMPLE' ranks by readable
  disagreements first (unreadable samples only break ties).
S4 MMLU excludes the smoke questions. S5 D uses the 15 hooks of the pilot (not 22).
S6 wording: CONF is weak on these families BY SELECTION (fast errors were picked for being confident); a pass that rests mainly on
  FLAG - CONF is not general evidence; the verdict line says so. On the 2-option families (hindgen, redef) recall(FLAG) equals
  P(thought right | fast wrong), i.e. "thinking fixes the error".
Also: the main set excludes the items of the first smoke run (its file was regenerated); smoke2 (seed 12345, new code) logs
  wall-clock seconds for a cost check before the main run.
Cost check (smoke2: 32 items + 16 single generations took 190-330 s per model): main estimated ~25 min H100 per model; MMLU cut from 200 to 100 (report-only) to stay near the stated budget.
