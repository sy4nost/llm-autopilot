# Review of asktwice/PLAN.md (v1), before code (2026-10-18)

Reviewer: independent methods audit. Read: PLAN.md, posthoc.txt, disguise/PLAN.md, disguise/rezultaty.md, audit/audit-report.md.
No code was run and nothing paid was run. Verdict: **do not code yet.** Four blockers and nine required changes are below.

## Short version
The plan's two headline criteria (catch on traps, false alarms on agree controls) can both pass for a boring reason.
On these families "trap vs agree" means "the context sentence states a false fact vs a true one". A one-pass question
("is sentence 1 true in the real world?") or the model's own knowledge separates them perfectly, and that rival is not in the plan.
The control that actually tests the harm seen on road words, where the disguise removes knowledge the question needs, is also missing.
Separately, earlier data (explicit redefinitions were 100% right) suggest these families may show no habit at all.

## BLOCKERS

B1. **The strongest trivial rival is missing: a "counterfactual-context detector".** S ("flag every item with a context sentence")
is a strawman, because it fails agree controls by construction. The real cheap rival is S2: one extra pass,
"Is the first sentence true in the real world? Answer Yes or No.", flagging on No. A rule rival does the same job:
flag when the context value differs from the plain-control answer (one pass on the plain item, which the plan already has).
On traps vs agree controls, S2 should score close to catch 100% and agree false alarms 0%, so ask-twice can at best tie it.
Fix: add S2 (both the model-asked version and the plain-answer-mismatch version) as pre-registered rivals at equal cost.
Ask-twice must beat S2 on a population where S2 can fail (see B2), not on traps + agree.

B2. **Agree controls are the wrong false-alarm test. They will pass trivially.** The disguised agree item ("a vorn has 7 days...")
is as easy as the normal one. This is why R-agree had a 0% flag rate post hoc. The harm the earlier experiment found
(N-agree 23%, N-confirm 22%) came from a disguise that *removes knowledge the question needs*. Agree controls cannot produce that.
Fix: add two control types per family, with >= 150 items each, and make criterion 2 apply to them:
- **KN "knowledge-needed"**: the context changes fact A about the hook, and the question asks about fact B, which the context does not touch.
  The habit is right here, and the nonce destroys the knowledge. Examples: "In this story, the capital of France is Lyon.
  What is the first letter of the continent France is in?" (E / A). "In this world, a week has 5 days. How many hours are in 2 days?"
  (48 / other). "In this game, the opposite of 'hot' is 'blue'. What is the last letter of 'hot' spelled backwards?" If B can be
  answered from the nonce item alone, the item is not KN; the generator must check this.
- **IC "irrelevant counterfactual"**: the context sentence is counterfactual, but the question does not use it. S and S2 both fail here (they flag).
Then "practical gain over S/S2" is measurable. Ask-twice should not flag IC or KN items, while S and S2 flag all of IC.

B3. **Habit may be absent, so the run may produce "nothing to say".** Earlier data: simple explicit redefinitions ("Redefine π as 600")
were 100% right. The habit showed up only where a computation step or vagueness carried the hook (T6 e, T7 ÷, road words).
K and O are explicit restatements followed by a trivial lookup ("Lyon" -> L), so the habit is probably near 0.
U (5 x 3) is the only plausible candidate. Under the current gate, 1 of 3 families with a habit means paying for a null.
Fix: make the dev habit check on seed 12345 (not Alex's seed) **decisive for family selection** before PREREG.
Pre-list harder variants now and pick by the dev habit rate alone, never by flag performance:
- the hook is used implicitly ("How many days until the end of 3 weeks?", "the city where France's government sits");
- vague phrasing ("people here count a week as 5 days");
- the context sentence is far from the question (filler sentences between them);
- two steps (U: "3 weeks and 2 days").
Record the dev habit table in PREREG. This is a design choice made before the frozen seed, so it is not tuning on test data.

B4. **With 2 options, any error counts as a habit error, and noise can open the gate.** In O especially, last-letter extraction
fails for tokenization reasons, not because of the habit. With 2 options, that failure falls onto the "habit" option.
Fix: use 3 options (gold, habit, a distractor of the same kind, e.g. 15 / 21 / 18, L / P / M, e / d / s). Define a habit error
as choosing the habit option. The gate becomes: habit-option rate on traps >= 15 points **and** >= 3x the distractor rate.
Also require the plain control to be right (the model knows Paris and knows 'cold'). Drop items where plain is wrong:
there is no habit to speak of there. Report the drop rate.

## REQUIRED CHANGES

R1. **Catch is uninformative when nearly every trap is an error.** If a family has about 95% habit errors (like T7 and D1),
catch mostly measures whether the disguise works at all, and S/S2 match it. Report precision (habit errors / flagged) and
flag rate on a **pre-specified pooled mix**: traps : agree : KN : IC = 1:1:1:1. Report each population separately too.
Add the criterion "precision on the mix beats S2 by >= 15 points (CI low > 0)".

R2. **PARTIAL must not be reachable without item-level evidence.** Within-cell AUC is the only test that separates item-level
from template-level flags. The post hoc showed the pooled 0.92 was largely template-level. If cells come out all-wrong or all-right
(likely for "week" or "France"), criteria 3 and 5 are "not measurable", and PARTIAL can then rest on catch + agree false alarms,
which pass trivially (B1, B2). Fix: with criterion 3 not measurable, the verdict is at most "INCONCLUSIVE", not PARTIAL.
Add a one-pass rival "dev hook prior": flag every item whose hook word's dev-seed habit rate is >= 50%. Ask-twice must beat it on catch at equal false alarms.

R3. **The hook extraction rule is underspecified and family-specific in practice.** "Subject of 'In this world, (the) X ...'" yields
"week" in U, but "capital" in K ("the capital of France") and "opposite" in O. The plan's intended hooks (France, hot) are not
the subject. A per-family rule amounts to knowing the generator's map (the "ideal translator" caveat from the disguise experiment).
Fix: freeze one regex for all families in PREREG with dev examples. If K and O need a different rule, say so and label the
module "template-aware". Also run the variant where the literal subject ("capital", "opposite") is replaced and report it.

R4. **Nonce leakage and confounds.** One fixed nonce per family turns any effect of that nonce into a template effect.
In K and O, a nonce whose first or last letter equals an option letter biases the answer.
Fix: use >= 4 nonces per family, assigned by item, none sharing a first or last letter with any option. Keep nonce token counts equal.
Report results per nonce. Check that "keeping plural/possessive forms" never produces a real word (e.g. "vorns").

R5. **The action on a flag is undefined, so "practical gain" is undefined.** If a flag means "take the disguised answer", ask-twice
equals "always disguise" (as in disguise/PLAN.md) and the flag adds nothing. Fix: pre-register the action as "send to review"
(cost = flag rate). Report "always disguise" accuracy on every population as a rival. The flag is worth something only if
always-disguise loses on KN while ask-twice + review does not.

R6. **CIs should be clustered by template, not only by item.** The claim is that this generalizes to new templates and hooks.
Items within one hook word are not independent. Fix: use a cluster bootstrap over hook words (~40 per family) within family,
families weighted equally. Keep the item bootstrap as secondary. Note: with the 15% upper limit, agree/KN false alarms need a
true rate around <= 9% at n = 150. Raise KN to 200 if possible.

R7. **Option-mass filter on both versions.** If the disguised version's option mass collapses, the flag becomes noise.
Fix: also require mass >= 0.5 on the disguised version, or count a low-mass disguised answer as "no flag", whichever is pre-registered.
Report the drop rate per family and population, and apply the same filter to every arm (as before).

R8. **Rival P is weak; I is under-specified.** Beating P by 20 points on catch is easy and says little.
Fix: keep P as a sanity check. Define I as a second ask (flag = normal and I disagree) and add it to criterion 4 at equal weight.
Pre-register that S2 (B1) is the main rival.

R9. **Drop "plain" as a false-alarm population.** Its flag rate is 0 by construction. Use it only as the knowledge check in B4.
Specify the template count and per-template variation (counts 2..9, multiple fake capitals) so each cell has >= 8 distinct items.

## OPTIONAL

O1. Add a fourth family that mirrors the post hoc's strong case (an implicit symbol inside a computation), for example
"In this world, a dozen is 10. How many eggs in 3 dozen plus 2?". This gives one test that is close to isomorphic to the post-hoc family, next to the farther ones.
O2. Pre-register the expected result now: e.g. "catch ~90%, KN false alarm ~20%, S2 ties on traps/agree". Compare at report time, as the disguise run did.
O3. Report the agreement-conditional accuracy (normal right when not flagged), per population. It was the post hoc's most useful number (96%).
O4. Log full option log-probs for all arms, so a margin-threshold version of ask-twice (a soft flag) can be reported without a rerun.
O5. Cost stays under $3 even with the added controls (about 4 populations x 4 passes x 3 options). Cost is no reason to skip B2.

## v2 check (2026-10-18)
Verdict: **needs changes before code.** One new blocker-level issue (N1) and six smaller fixes. Everything else is resolved.

Status of v1 items:
- B1 S2 rival: resolved. The model-asked truth check was added. The plain-answer-mismatch version was not; that is acceptable.
- B2 knowledge-needed (KN) and irrelevant controls: added, and they enter BA through the mix. But one example question leaks the answer (N3).
- B3 dev variant selection: resolved. The rule selects only on habit rate, never on flag performance, and is frozen.
- B4 3 options and the gate: resolved, except that distractors are not specified (N5).
- R1 resolved (BA on a 1:1:1:1 mix). R2 resolved (INCONCLUSIVE). R3 resolved (disclosed as template-aware).
  R5, R7, R8 and R9 resolved. R4 partly: see N6. R6 partly: see N4.

New problems:
N1. **(Blocker) Criterion 1 can pass for a surface reason.** AT is silent on irrelevant items by construction: the hook is not in
    the question, so the disguised question equals the normal one. S2 flags all of them. So BA(AT) - BA(S2) is driven largely by the
    irrelevant population, and a cheap rule gets the same benefit: **S3 = S2 AND the module's hook word (or its listed forms) appears
    in the question**. S3 is silent on irrelevant items and, like AT, flags KN items. AT and S3 differ only on traps the model got
    right (and agree items), which is exactly the item-level claim. Fix: add S3. In criterion 1, replace S2 with max(S2, S3), or add
    "BA(AT) >= BA(S3) + 0.05, low > 0". Keep criterion 4 report-only or drop it, since it passes by construction.
N2. **Rival H is underspecified.** Does it flag every item of a high-error cell, agree and KN included, or only counterfactual
    contexts? Fix: H = S3 restricted to cells with a dev trap error rate >= 50% (its strongest form). Dev must cover every hook
    word with >= 6 trap items, or H falls back to S3 for that hook.
N3. **The KN examples must not be answerable from the context.** D: "a zib has 4 sides. Is a zib a shape, an animal or a food?"
    The context gives away "shape". Fix: a generator check that drops a KN item if the disguised version is answerable from the
    context alone, judged by a rule fixed in advance, not by the model. Also specify whether options are disguised. K options such as
    "French" (language) or "French capital" contain the hook adjective; replace consistently or forbid hook forms in options.
N4. **Cluster counts and cell sizes do not add up.** U and D may have only about 8-10 hook words, and a percentile cluster bootstrap
    with fewer than about 15 clusters gives intervals that are too narrow. K has about 40 countries for 240 traps, about 6 items per
    cell, which violates the ">= 8 per cell" rule and makes criterion 3 cells (>= 3 errors and >= 3 right) rare. Fix: state the hook
    count per family in PREREG. Require >= 15 hooks or widen the CIs (e.g. report the item bootstrap too and take the wider one).
    Set K traps per hook to >= 8 (fewer countries or more traps).
N5. **Distractor construction and M's threshold.**
    - Distractors: give the rule per family and variant (e.g. U: count x (true unit size +/- 1); K: a third real city of that country).
      The 3x gate depends on it.
    - M: "median on dev" fixes M's flag rate at 50%, which is arbitrary and can handicap M. Pick M's threshold as the one that
      maximises BA on the dev mix. A dev-tuned rival is conservative for AT.
N6. **Assigning nonces by hook word makes the per-nonce report equal the per-hook report.** Fix: assign nonces by item, balanced
    within each hook (round-robin over items).
N7. **Seed overlap.** With small hook lists, the dev (12345) and main seeds can produce identical items, especially in K.
    Fix: deduplicate the main set against dev by exact text, and report how many items were removed.

Minor: report length-normalised scoring as a check (K options are whole city names of different token lengths), and record dev
vs main habit rates per family (regression to the mean after variant selection).
