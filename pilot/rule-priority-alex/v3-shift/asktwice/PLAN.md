# Plan v2: "ask twice" checker (2026-10-18). v1 = PLAN_v1.md, review = review/plan-review.md (4 blockers, 9 changes).

## Origin (post hoc, posthoc.txt)
Disagreement between normal and disguised answers flagged habit errors on redefine-math (pooled AUC 0.92), but much of it is
template level; within template T3 0.94, T6 0.96, T5 0.53; road words (oracle twin) 0.60/0.63 with 22% false flags.

## The question, sharpened by review blocker 1
A cheap rival knows which contexts are counterfactual (S2: "is sentence 1 true in the real world?"). So the checker is only
useful if it flags **the cases where the model actually fails**, not every counterfactual: (a) among counterfactual traps,
separate the model's errors from its right answers (within-cell); (b) stay quiet on counterfactuals the question doesn't use;
(c) the price: false alarms where the nonce removes knowledge the question needs.
**Flag action = "send to review"** (not "replace by disguised answer"; "always disguise" is reported as an answer policy).

## Model, format
Qwen2.5-14B-Instruct bf16 H100; F2 (chat + "Reply with only the answer, nothing else." + prefix "The answer is"); each
option its own full pass, summed log-prob; full per-token log-probs saved. **3 options**: gold, habit (real-world), distractor.
Option-mass filter >= 0.5 on BOTH normal and disguised (else the item is dropped for every method).

## Candidate families and difficulty variants (all listed now; dev on seed 12345 picks one variant per family)
- U units ("a week has 5 days"; week/days, hour/minutes, dozen/items, year/months, minute/seconds, pair/items, ...).
  v1 "How many days are in 3 weeks?"; v2 two-step "3 weeks and 2 days"; v3 implicit: story "Ana worked every day for 3 weeks.
  How many days did she work?"; v4 = v3 with 3 filler sentences between context and question.
- K capitals ("the capital of France is Lyon"; fake capital = another real city of that country).
  v1 "first letter of the capital of France?"; v2 implicit: "Marie flew to the French capital. Which city did she land in?";
  v3 = v2 + 3 filler sentences; v4 two-step "Which river flows through the city Marie landed in?" (dropped if no clean river).
- D defined words ("in this game, a 'triangle' has 4 sides"; triangle/sides, spider/legs, cube/faces, hexagon/sides, ...).
  v1 "How many sides does a triangle have?"; v2 "How many sides do 3 triangles have in total?"; v3 implicit story with counting;
  v4 = v3 + filler.
Frozen dev rule: per family take the variant whose normal right on traps (dev, after filters) is closest to 50% within
[15%, 85%]; ties -> lower variant number. Family enters only if (i) such a variant exists, (ii) habit-option choices on dev
traps >= 3x distractor choices, (iii) plain control right >= 90%. Need >= 2 families, else stop ("nothing to say", no main run).

## Populations per family (main run)
- traps: counterfactual context, question uses it. Error = normal answer not gold.
- agree: true context, same question.
- knowledge-needed: counterfactual context about hook X, question about ANOTHER property of X that the nonce destroys
  (K: "What language is spoken in France?"; U: "How many weeks are in a fortnight?"; D: "Is a triangle a shape, an animal or a food?").
- irrelevant: counterfactual context about X, question about a different entity Y (module replaces X only).
- plain: no context; knowledge check only (items with plain wrong are dropped from all populations of that hook).
>= 8 distinct items per cell (cell = family x hook word); target 240 traps, 120 each control per family.

## Module (template-aware, disclosed): per family one frozen regex finds the hook word (U: "a(n) (\w+) has", K: "capital of
(\w+) is", D: "a '(\w+)' has"); hook replaced whole-word everywhere incl. plural/adjective forms listed per hook
(France/French, week/weeks). 4 nonces per family, assigned by hook word round-robin; no nonce shares first or last letter with
any option; results reported per nonce.

## Rivals (each a "flag" rule)
- AT ask-twice: flag if disguised answer != normal answer.
- S surface: flag if a context sentence exists.
- S2 truth check (one extra pass): flag if the model answers "no" to "Is the first sentence of this task true in the real
  world? Reply yes or no." (model's own judgement, F2 scoring of yes/no).
- M low confidence: flag if normal margin (top1 - top2 log-prob) below its median on dev.
- P paraphrase second ask (context moved after a neutral lead-in, hook kept): flag if answers differ.
- I instruction second ask (normal + "Follow the facts stated in this task exactly, even if they contradict what you know."):
  flag if answers differ.
- H dev hook prior: flag the cells whose dev trap error rate >= 50%.
Answer policies (report only): normal, always-disguise, always-I.

## Metrics
Mix = traps, agree, knowledge-needed, irrelevant weighted 1:1:1:1 per family, families equal weight.
- BA = balanced accuracy of "flag" vs "normal answer wrong" on the mix (mean of catch and 1 - false-alarm rate).
- within-cell AUC on traps (cells with >= 3 errors and >= 3 right; cell-weighted), AT vs M (continuous: disguised-vs-normal
  log-prob shift of the normal answer for AT; margin for M).
- false-alarm rates per population.
CIs 95%, clustered bootstrap: resample hook words within family, then items; 2000 reps.

## Criteria (PASS needs all; families with a habit only)
1. BA(AT) >= BA(S2) + 0.10, low > 0.
2. BA(AT) >= max(BA(M), BA(P), BA(I), BA(H)) + 0.05, low > 0.
3. within-cell AUC(AT) >= 0.75, low > 0.6, and >= AUC(M) + 0.10, low > 0. Not measurable -> verdict at most INCONCLUSIVE.
4. false alarm of AT on irrelevant <= 10% (high <= 15%).
Report only: knowledge-needed false alarms (expected high: the price), per family, per nonce, per variant, accuracy when not flagged.

## Written expectation (before data)
AT ~ S2 on traps vs agree; AT loses on knowledge-needed (many false alarms); AT wins on irrelevant; within-cell AUC ~0.7.
My guess: criterion 1 fails because of knowledge-needed. That would mean the checker is useful only paired with a filter.

## Order and cost
Code -> dev on 12345 (variants, gate) -> code review -> PREREG.md with sums -> Alex's number -> run -> frozen score -> report.
~2500 items x (normal, disguised, P, I, S2) x 3 options ~ 35k short passes + dev ~ $2.5-3.

## v2.1 changes after review/plan-review.md "v2 check" (these override the text above where they differ)
- N1: new rival **S3** = S2 flags AND the hook word (any listed form) appears in the question part. Criterion 1 becomes
  BA(AT) >= max(BA(S2), BA(S3)) + 0.05, low > 0. Criterion 4 (irrelevant false alarms) is report only (AT is silent there by construction).
- N2: **H** = S3 restricted to cells whose dev trap error rate (chosen variant) >= 50%. Dev has 120 traps per variant per family
  (>= 6 per hook word).
- N3: knowledge-needed only for U and K. D's category question leaks from the context ("has 4 sides" -> shape), so D has no
  knowledge-needed population; its mix is traps, agree, irrelevant 1:1:1. Options are never disguised (e.g. "French" stays).
- N4: CIs = both the clustered (hooks, then items) and the plain item bootstrap; the wider of the two is used.
  Hooks: U 10, K 20, D 11 -> ~24/12/22 traps per cell.
- N5: distractors (fixed in make_at.py): U (n+1)*value or n*value+n*real; K v1 = first letter of a third city of that country
  (or a random letter), v2/v3 = the third city; D (n+1)*value (v1: value+real+1). For controls the "habit" slot is replaced by
  a second distractor (gold + n). **M** threshold = the margin cut that maximises BA of "flag" vs "normal wrong" on dev traps+plain
  of the chosen variants.
- N6: nonce chosen by the module from the text (sha256 of the text mod 4), so nonces vary within a hook; report per nonce.
- N7: main items whose text equals a dev item or an earlier main item are dropped; count reported.
- Report only: length-normalised scoring check; dev vs main habit rate per family.
- v2.2 (found while coding, before any main data): U knowledge-needed questions are themselves changed by the counterfactual
  ("a century has 103 years; centuries in a millennium?" is ambiguous), so knowledge-needed exists only for K (language of the
  country). Mix: U and D = traps, agree, irrelevant; K = traps, agree, knowledge, irrelevant. Generator asserts that the hook
  word is in the question for every population except irrelevant. Hooks with plain right < 80% (main) are dropped everywhere.

## v3 (2026-10-18, after dev 1 stopped the run, dev-itog.md; Alex chose "redo the trial")
Dev 1 showed the habit only when the familiar fact is asked back directly. New variants (same populations, same criteria):
U v5 "a day has 23 hours. How many hours does a day have?"; K v4 "the capital of Kenya is Kisumu. What is the capital of Kenya?".
D v1 is already direct. Dev 2 on a fresh seed 12346, all variants, same frozen dev rule (choose_dev.py), which now decides.
Main items are de-duplicated against both dev sets. Dev 1 data are not used for the choice (seen already).

## v3.1 (2026-10-18, after the technical smoke on seed 12345; before any main data)
Greedy continuations (gen_check.py) show why option mass is low on many K items: the model writes "The answer is incorrect based
on the given premise..." - it OBJECTS to the counterfactual instead of answering. The mass filter therefore removed exactly the
cases where the model notices (all smoke knowledge items, half of irrelevant). Change: main scoring is **forced choice with no
option-mass filter**; new rival **O "model objects"** = flag if the normal option mass < 0.5 (free, no extra pass). O joins
criterion 2: BA(AT) >= max(BA(M), BA(P), BA(I), BA(H), BA(O)) + 0.05, low > 0. Dev variant choice (made with the filter) stands.
