# Independent review of disguise/PLAN.md ("disguise" and "shake")

Reviewer: independent agent, 2026-10-18. No GPU, no Modal. All checks are small `python3 -I` scripts in this folder;
their outputs are saved next to them:

| script | output | what it checks |
|---|---|---|
| `check_twin.py` | `check_twin_out.txt` | native vs fictional twin, same seed, each built in a fresh interpreter |
| `check_redefine.py` | `check_redefine_out.txt` | 9 redefine templates: gold, R-same key, the plan's substitution rules |
| `check_rsame_distinct.py` | `check_rsame_distinct_out.txt` | how many *distinct* R-same prompts there are |
| `f3_per_template.py` | `f3_per_template_out.txt` | normal-model behaviour per template / probe in the general-lamp main run (F3, seed 88) |
| `sim_disguise_N.py` | `sim_disguise_N_out.txt` | the disguise module on N, simulated from answers already on disk (seed 20261020) |

**Overall verdict: NOT READY for PREREG.** There are 4 blockers. Each one can be fixed without new compute,
or with one cheap dev-only run. As written, method 1 (disguise) is close to a **FAIL that is fixed by
construction**: R-same harm is built in, and existing data already predict about +2.5 points on N. The shake arms
have a degenerate R-same control and a placebo that is not neutral. A PASS on a single family is allowed, and that
family could be N on its own.

---

## 1. Disguise on R and the R-same control

**Data sanity: OK.** In all 900 rows the gold option equals the redefined answer (the first character of the
expression for T0/1/2/7/8, the first digit of the redefined value for T3–T6). The non-gold option equals ordinary
math. No item has identical classes. No division gives a non-integer. No redefined value starts with the habit digit
(3 for π, 2 for e). T5 has 8 items whose value is 0, with gold "0", which is fine. The `\be\b` rule hits only T6
("Redefine e as", "of e?"). It hits no other word.

**Substitution rules taken literally (applied to every prompt): BLOCKER.**
- **T3 and T6 break.** The definitions contain "+" ("Redefine π as 715+245", "Redefine e as 10+36"). A global
  "+"→"⊕" rule turns them into "Redefine zorp as 715⊕245". The gold (960 → 9) is then undefined. This hits all
  200 items (180 in test).
- **T0 has no redefined symbol.** The rule there is "View number as text". The "+" rule rewrites the *question*
  ("first digit of 50⊕47"). That changes the task and removes the computation cue by construction. It is a
  different manipulation from the one used on T3–T6.
- "⊕" is not a neutral symbol. It is XOR or direct sum. "⊘" is less loaded. An unknown symbol such as "§" or a
  made-up token would be more neutral.
- **Fix:** write the map per template, from text the module can see: T1/T2 replace "+"; T7/T8 replace "÷"; T3–T5
  replace π; T6 replaces e. Never touch "+" inside a definition. For T0, say explicitly either "no disguise
  (identity)" or "excluded from method 1". Do not leave it to whatever the code does. Write the rewritten dev
  example of each template into PREREG.

**R-same as a harm control for disguise: BLOCKER (the result is fixed by design).**
- Once the definition is removed, there is no "redefined symbol" left. A module that looks only at the prompt has
  nothing to disguise. The plan still applies the fixed map. This gives "Q: What is the first digit of zorp?"
  (T3–T6), which has no answer, and "first digit of 10⊕19" (T1/2/7/8), where the only sensible reading is the
  first character, i.e. the wrong answer. A loss far above 5 points is close to certain: 4/9 of items become
  coin-flips, and most of the rest are pushed toward the wrong option. A FAIL on criterion 3 therefore says
  nothing about the idea. It only reflects one implementation choice.
- **Fix:** define the module as an algorithm that works on the prompt text. (a) Detect a redefinition statement
  ("Redefine X as", "View X as"). (b) Only then disguise X. On R-same the module then changes nothing, which is
  honest for such a module. Add a **harm control with the redefinition kept, where gold = habit**, for example
  generated items "Redefine π as 3NN" or "Redefine e as 2N+N", or a T4-like "Redefine π as 3.14". This catches a
  module or shake that just pushes the model away from its prior whenever a redefinition is present.

**R-same is degenerate for T3–T6, for every method: BLOCKER.** In the test split, T3, T4 and T5 each produce one
single prompt ("Q: What is the first digit of π? A:", repeated 90 times). T6 produces one ("…of e?"). That is 360
of 810 R-same items made from **2 distinct prompts**. Overall there are only 363 distinct R-same prompts. Only the
distractor option differs between items. For any non-noise method, harm on these 44% of items is all or nothing.
A bootstrap over "questions" treats them as independent, so the intervals are far too narrow.
**Fix:** run R-same only on T0/1/2/7/8, which have distinct arithmetic, or use the generated "gold = habit"
items above. Bootstrap over distinct prompts.

**Ambiguity of the R-same key: CONCERN (minor).** "First digit of 50+47" with no context: the plan takes 9 (the
computed answer) as correct. A literal reading gives 5. This is a convention, not an error. State it, and report
T0/1/2 R-same separately.

**Template heterogeneity: CONCERN.** In the general-lamp main run (F3), normal accuracy per template was t4 100%,
t6 0%, t7 0%, t3 83%, t8 90%. The share of items with option mass ≥ 0.5 ranged from 0% (t5, t6) to 100% (t4). A
pooled +10 can come entirely from one or two templates (e.g. t6/t7 going 0 → 100%), while the "within-template
question" bootstrap reports a tight interval. **Fix:** say how templates are weighted (equal weight per template
is recommended). Require the gain to have the right sign in at least k of the templates that have a habit, or add
a leave-one-template-out check, which should also pass. Exclude no-habit templates such as t4 at the template
level, not only at the family level.

**F2 baseline is unknown: CONCERN, close to BLOCKER.** The habit that has been measured, and the "confident
habit, no lamp" result, both come from F3. devformat_out.json reports only option *mass* for F2. It does not
report accuracy. Under F2, R may be ≥ 85% correct and drop out. Then a method can PASS on N alone (see section
4). **Fix:** before PREREG, run normal F2 on the **dev** split (90 items, a few minutes) and record the per-template
habit rate and mass retention. This is allowed because it is dev data.

## 2. The N twin and N-same

**Twin claim: OK (verified).** For seeds 20261020, 12345, 777, 778 and 779, each family was built in its own
interpreter. Arms, sets, probes, label positions of key and old, field order, label-triple index and the A/B/C
histories all match exactly. After every set-specific name, value and label is replaced by a positional
placeholder, the **prompts are identical** for all 1680 items per seed. No native word appears inside a native set
in two roles.

**Implementation trap: CONCERN (must be checked in code).** `make_native` patches `make_match`'s module globals
in place. If the fictional twin is built in the same process after `make_native` is imported, the "twin" is the
native task. The key-equality check still passes, and the disguise becomes the identity. make_gl.py avoids this
only through the order of imports. **Fix:** build the twins in separate processes, or reload the module. The twin
check must also assert that the words *differ*: no "pedestrians", "crosswalk" or "barrier" in any fictional
prompt, and at least one differing word per set.

**Is the twin really neutral? CONCERN.** The fictional pools are only "less native". In 25–29 of 30 fictional sets
at least one word has road or real meaning (STOP/GO/CHECK or HALT/PROCEED labels in 8–12/30 sets, gate, signal,
pass, ticket, ACTIVE, PRESENT/ABSENT, GREEN). The labels also tokenize differently from the native ones. Scoring
is by *summed* log-prob, so label length bias differs between the normal and the disguised versions. Report
first-token or length-normalised scoring as a sensitivity check.

**"Guaranteed win" on N: CONCERN. Not guaranteed, but largely predictable.** I simulated the module (twin answer
whenever it differs) on the stored change_0 greedy answers (`sim_disguise_N_out.txt`):

| probe | normal | twin = module |
|---|---|---|
| D1 | 10% | 23% |
| D2 | 100% | 83% |
| D3 | 13% | 47% |
| D4 | 100% | 80% |
| **pooled D1–D4** | **55.8%** | **58.3%** (gain **+2.5 [−4.2, +9.2]**) |

On the probes with a habit (D1/D3), disguise gains about +23 points. On D2/D4 it loses about 18 points. There the
pretrained road sense *agrees* with the new rule, and the model is 100% correct only because of it. Pooling D1–D4
mostly measures this trade-off. **Fix:** define N-traps = D1+D3, and treat D2+D4 as a harm control *with the note
present* ("N-agree"). Then pre-register the expected result from these numbers. When the claim is written, call
disguise on N an **oracle** disguise: the module is handed the generator's word map, which a real system would
not have. A PASS would show that the habit comes from the words. It would not show that an external rewriter is
feasible.

**N-same (note removed) as a harm control: CONCERN.** It is valid for one failure only: noise from the disguised
pass when everything needed is already in the prompt. It cannot detect loss of world knowledge, because none is
needed. For shakes it misses the important failure, "push away from the habit whenever a note is present". Every
N-same item has no note. **Fix:** add N-confirm (a note that re-states the unchanged Rule 2, key = old rule),
and/or use D2/D4 as above. Keep N-same as a secondary check.

## 3. Shake arms

**Placebo: CONCERN.** "The room is quiet, and the afternoon light is soft and calm" is a *calming* line, the
opposite of a shake. It is not neutral. If 2a beats it, the reason could be that the placebo makes things worse.
2b is also an *instruction* ("look at this task as if for the first time"). The placebo controls for length, but
not for the line being an instruction. **Fix:** use a neutral line matched in token count, e.g. a dull factual
sentence. Better, use two placebos and compare against their mean. Add an instruction baseline (section 5).
Report the token counts of all lines.

**Insertion point: CONCERN (needs exact spec).** R prompts are one line. "Insert a line before 'Q:'" needs exact
whitespace, for example `Redefine π as 462.\n<LINE>\nQ: …`. This should be identical for R-same, where the line
then comes first. For R, the line also separates the definition from the question. That distance may weaken the
redefinition by itself. The placebo controls for this only if it is placed in the same position. Give the exact
string for one N, one R and one R-same item in PREREG.

**Noise arm: CONCERN (several details not yet defined).**
- "Output of layer 20": HF `hidden_states[20]` is the output of `model.model.layers[19]`. Name the module index
  and the hook point (residual output `out[0]`).
- "α × mean state size": the mean over which positions? Qwen has very large norms on the first or sink tokens. A
  mean that includes them makes the noise many times larger at every other position. **Fix:** scale per position,
  `h_t += α·‖h_t‖·z_t/‖z_t‖`, with z drawn as Gaussian on the CPU from `torch.Generator(seed)`. Exclude, or
  report, the sink position.
- KV-cache scoring: options are scored in extra passes from a copied cache. Noise must also be added to the option
  tokens, and must be a function of (item seed, absolute position), not of call order. Only then does the
  existing cache-vs-full check still hold under noise. Add that check for α > 0.
- "Question number": define a stable id per family and version (e.g. a hash of family|uid). Then N, N-same, R and
  R-same seeds are unambiguous and do not depend on how the filter orders items.
- Sanity checks: α = 0 must give scores that are bit-identical to the run without the hook. Two runs with the same
  α and seed must be identical. Add a third check: a non-zero α gives some flips on dev. If α = 0.1 is a random
  direction at 10% of the norm in 5120 dimensions, it may change nothing, and the result would be a FAIL that
  tells us nothing. Report the flip rate per α on dev before PREREG, and fix α there.
- Noise has no matched control. A gain from noise may just be flattening toward chance. **Fix:** report the
  flips (wrong→right on traps vs right→wrong on controls) and a "random flip at the same rate" baseline.

## 4. Filter, 85% rule, criteria, multiplicity, bootstrap

- **Mass filter on the normal version only: OK in principle, CONCERN in detail.** Say whether N-same and R-same
  are filtered on their own normal versions. They have different prompts, so the trap set and the control set
  will contain different items. Report n per template and probe. For disguise, the disguised version may have low
  mass (e.g. "zorp"). Its argmax is then close to noise. Either use the disguised answer only when the disguised
  mass is ≥ 0.5 (fix this rule in advance), or report the result both ways.
- **85% "no habit" rule: BLOCKER in combination.** With this rule, a method can PASS on one family. N is the
  family where disguise works as an oracle, and R under F2 has never been measured. A PASS on one family must be
  labelled "partial" and not counted as a PASS. Apply the rule per template for R and per probe for N. On the
  general-lamp data, D2/D4 and t4 have no habit at all.
- **Thresholds are asymmetric: CONCERN.** A gain needs ≥ +10 points and a lower bound > 0. Harm needs only a
  point estimate ≥ −5. **Fix:** harm should also need the lower bound of the change to be > −10 (or a
  non-inferiority bound fixed in advance), checked per family and reported per template.
- **Multiplicity: CONCERN.** There are 4 methods, each with 2–3 criteria, and a one-family exit. Use 98.75%
  intervals (Bonferroni over 4) or Holm, or say plainly that each method is tested separately and that no "best
  method" claim will be made.
- **Bootstrap units: CONCERN.** N by set is right. Draw from the 90 sets across the 3 seeds. R "by question within
  template" ignores variation between templates and the duplicated R-same prompts (see above). Criterion 4 pools
  N and R: give the weights (by n would let R dominate, 810 vs 360) and say that the two families are resampled
  separately.

## 5. Room for tuning after seeing data, and missing baselines

**"Prefer disguised only on disagreement" = "always use the disguised answer": CONCERN.** These are the same rule.
If the two answers agree, you keep the disguised one; if they differ, you take it. So the normal pass has no
effect. Say this in the plan. The interesting use of two passes is **disagreement as a detector**, i.e. the lamp
that general-lamp did not find. Pre-register the AUC of "disagreement" for habit errors, and a selective rule
(flag or abstain on disagreement) with coverage and accuracy. It costs nothing extra.

**Missing baselines: CONCERN.**
- An **instruction baseline**: "The rules/definitions in this prompt may contradict what you already know; follow
  them exactly." This is the cheapest external fix. Run it with the same harm controls. Without it, a 2b PASS
  cannot be told apart from plain instruction-following.
- A **reword without neutralising** control for disguise: native → other native synonyms, or π → another familiar
  constant name. It separates "the words no longer carry meaning" from "any second pass on rewritten text".

**Things not yet fixed, which leave room for tuning after seeing data:**
- the substitution scope per template and the handling of T0;
- exact insertion strings and whitespace;
- the noise hook index, scale, positions and seed ids;
- template weights and the units for pooling N+R;
- whether filters are applied per version;
- how the 85% rule is computed (pooled or per template);
- tie-breaks;
- generator seeds: `make_gl.USED` lacks general-lamp's seeds 88–95. They must be skipped too, or N items repeat.

Note also that the R test split is the same 810 questions as general-lamp. Their per-template F3 behaviour has
already been seen (above), and Alex's number S does not change R at all. Say so in the plan.

---

## Required changes, by priority

1. **(Blocker)** Make the R disguise a per-template algorithm on the prompt text. Do not touch "+" inside a
   definition (T3, T6). Handle T0 explicitly. Show the rewritten dev example of each template.
2. **(Blocker)** Replace or add to the harm controls:
   - R: run R-same only on T0/1/2/7/8, add a "redefinition present, gold = habit" set, and bootstrap over
     distinct prompts;
   - N: split into N-traps = D1+D3 and N-agree = D2+D4 (both with the note), and add N-confirm.
3. **(Blocker)** No PASS on one family: when one family drops out, the result is "partial". Before PREREG, run
   normal F2 on R-dev and publish the per-template habit rate.
4. **(Blocker)** Twin built in a separate process, with an assertion that the words differ (the import trap).
5. Neutral placebo matched in token count (two if possible), plus an instruction baseline arm. Exact insertion
   strings in PREREG.
6. Noise: per-position relative scale; named module index; noise also on the option-token passes; RNG keyed by
   position; α checked on dev for a non-zero flip rate; three sanity checks (α = 0 identical, same seed repeats,
   α > 0 flips).
7. Statistics: equal-weight templates plus a k-of-templates or leave-one-out check; a confidence bound on harm;
   multiplicity correction or a statement that each method stands alone; separate resampling of N and R in
   criterion 4.
8. Pre-register disagreement-as-detector (AUC, selective accuracy). State that "on disagreement" = "always
   disguised". Call N disguise an oracle and write down the expected N result (+2.5 [−4, +9] pooled; D1/D3 about
   +23, D2/D4 about −18 on the existing data).
9. Report sensitivity to length-normalised or first-token scoring for N (label tokenization differs between twins).
10. Add 88–95 to the seeds that are skipped.
