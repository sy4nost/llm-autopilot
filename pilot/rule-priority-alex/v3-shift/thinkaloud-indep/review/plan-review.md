# Review of thinkaloud-indep/PLAN.md v1 (before the paid run)

Reviewer: independent session, 2026-10-18. I ran nothing on a GPU. I only read the plan, the earlier thinkaloud files and the audit, and ran data checks with `python3 -I` (scripts in my scratchpad; nothing from the data files was executed).
Paths are relative to `v3-shift/`.

## Bottom line

The plan is careful about wording freeze and seeds. But its main safeguard against a trivial explanation does not work. Balancing by `answer_index` balances the **position or label** of the right answer (A/B, Y/N, list order). It does not balance **"habit answer right" vs "habit answer wrong"**. In every one of the five main families the right answer is *always* the non-habit option, exactly like the three "report-only" families. So a THINK that just learns "pick the unusual option" would pass C1/C2. Nothing in the plan could tell it apart, because the only harm check (MMLU) contains no habit at all.

There are 5 blockers and 9 suggestions. Most fixes are mechanical and cheap.

---

## BLOCKERS

### B1. The main families are single-type traps; "class-balanced" does not stop an odd-option shift

The plan's own reasoning for demoting modus-tollens, quote-repetition and pattern-matching-suppression to report-only is: *"only one correct class in the whole dataset, so a gain could be a plain shift toward the odd option."* The same holds for all five main families once "class" means "habit vs non-habit" rather than "list index":

| family | trivial rule | accuracy of the rule (my check, full file) |
|---|---|---|
| memo-trap | pick the option that appears in the instruction | right-only-match **833/936**; the other 103 are ties, e.g. "need passports" → " passports" |
| hindsight-neglect-10shot | answer the opposite of the outcome (lost → Y, won → N) | **315/315** (test items are always EV-neg & won → N, 173; or EV-pos & lost → Y, 142) |
| repetitive-algebra | pick the answer repeated in the examples (never the last example's) | **1000/1000**. The odd example is always position 9 (the last), and the wrong class is always that last example's answer |
| into-the-unknown | pick the option with less word overlap with the context | **69.3%** with a crude bag-of-words rule. By design the right answer is always "not already stated" |
| NeQA | pick the anti-common-sense option | 100% by design (all 300 are negated questions) |

- In memo-trap and repetitive-algebra, `answer_index` is just the order of the `classes` list. The model never sees that order (no A/B labels), so "100 + 100 by answer_index" balances nothing the model can use.
- There is **no within-family control** where the habit answer is right. The earlier thinkaloud run had one ("agree", "cfright"), and that is exactly where THINK failed (cfright 98.6 → 81.9).

**Fix (before running):** add mechanical controls built from the same public items, with no new wording. Then add a pre-registered within-family harm criterion, like C3 but per main family, or at least pooled:
- **memo-trap:** swap the instruction word to the famous ending ("ends in the word "fonder"", right = " fonder.").
- **NeQA:** remove the single negation (general-lamp already did this as "aux"; use the plan-literal " not " subset, see general-lamp/review/code-review.md §1). Right = the common-sense option.
- **hindsight:** move one in-context example (outcome agrees with EV; there are 3150 of them) to the test slot. Right = outcome-consistent.
- **repetitive-algebra:** make the test question's answer equal the last (odd) example's answer. Right = the "recent" one.
- **into-the-unknown:** no clean mechanical control exists. State that its gain is bounded by the odd-option explanation.

Without these, C1/C2 PASS can only be reported as "THINK moves answers toward the non-default option", not "THINK removes habit errors without harm".

### B2. hindsight is not class-balanced, which contradicts the plan's own check

- The plan says main families are "sampled class-balanced 100 + 100 so always one class = 50%". It also says hindsight uses "All 315 (173/142)".
- With all 315, **always N scores 54.9%**. That is above the known DIRECT level on this task: 45.4% in watch_public and 46.0% in public/. The audit (§6.2) already found that the previous "gain" here was a one-way class shift.
- **Fix:** sample 142 + 142, or score hindsight by balanced accuracy (mean of per-class accuracy). Pre-register per-class accuracy for every family.

### B3. The C4 detector is close to a restatement of THINK accuracy on 2-option tasks

- Every main task has exactly 2 options. So when DIRECT is wrong, "THINK ≠ DIRECT" is the same as "THINK right". When DIRECT is right, "no flag" is the same as "THINK right".
- Hence balanced accuracy = ½·[P(THINK right | DIRECT wrong) + P(THINK right | DIRECT right)]. This is THINK's accuracy reweighted over DIRECT's strata, not an independent detector. On single-type traps (B1), an odd-option shift drives it to about 1 automatically.
- The prior 0.90 likely has the same mechanics.
- Families near DIRECT ceiling give AUC on a handful of positives. repetitive-algebra DIRECT was 98.1% (watch_public), so about 4 positives in 200.
- **Fix:**
  - Either drop C4 as a verdict criterion, or state it explicitly as "THINK accuracy conditional on DIRECT correctness".
  - Add the prior run's minimum-count rule: ≥ 5 DIRECT errors and ≥ 5 DIRECT right per family, else exclude the family from the mean.
  - Keep "low DIRECT confidence" as the only genuine rival.

### B4. The C3 (MMLU harm) bound is set so that a harmless arm fails about half the time

- With n = 300 paired items and a typical 20% rate of answers changing between CoT and direct, the 95% half-width is about ±5.1 points.
- "low ≥ −5" then needs a point estimate ≥ about +0.1:
  - P(pass | true Δ = 0) ≈ **0.49**;
  - P(pass | true Δ = +2) ≈ 0.77.
- With n = 1000:
  - half-width ±2.8;
  - P(pass | true Δ = 0) ≈ 0.92.
- Cost is about +700 items × 2 generations, roughly +$1.5 by the thinkaloud rate (2,060 items ≈ $4).
- **Fix:** raise MMLU to ≥ 1000, or widen the bound to low ≥ −7 and pre-register the reason. As written, a C3 FAIL is mostly noise.

### B5. The forced choice after "Answer:" is invalid for completion tasks and biased by multi-token classes

1. **memo-trap and quote-repetition are "complete this text" tasks.**
   - After THINK the model will very likely write the whole sentence: "Answer: Absence makes the heart grow heavy."
   - The scorer uses the prefix up to the last "Answer:" and scores " heavy." / " fonder." as the *first* word after it. Both are unlikely there.
   - The choice then reflects a residual word preference, not what the model wrote.
   - The prior "parsed = forced 100%" check came from one-word answers and does not carry over.
2. **Summed log-prob (`score_ta.py` line 10) penalises longer classes.**
   - quote-repetition: the right class is the made-up word. It is longer in 127 of 300 items and shorter in 40 (examples: " questrom", " musculature", " dinnertime"). These are multi-token in Qwen's vocabulary, so DIRECT is pushed toward the famous word.
   - THINK has already written the odd word in its reasoning, so copying it is cheap. That gives THINK a scoring-only advantage.
   - memo-trap has multi-word classes with unequal length in some clusters (" whites and negroes" vs " cats and dogs"; (3,4) and (2,4)-word pairs, etc.).
   - Almost all memo classes end with "." (837/936), which is part of the scored string.
3. **Fix:**
   - For THINK/THINKI, make the primary answer a string match of the classes against the final "Answer:" line (last occurrence, word-boundary, case-folded). Use the forced choice only as a fallback when exactly one class is not found.
   - Or score the class at the end of the model's own final line: prefix = everything up to the class position.
   - For both arms, either score the first distinct token or also report length-normalised scores. Pre-register which one is primary.
   - In the smoke run, report parsed-vs-forced agreement **per family**. Set a go/no-go rule, e.g. ≥ 95% agreement on parsed items.

---

## SUGGESTIONS

### S1. The DIRECT format may handicap DIRECT on completion tasks. Make the published protocol primary, or require a gain over both
- The Inverse Scaling protocol is a raw-text completion with class log-probs.
- `chat(P + "Reply with only the answer, nothing else.") + "The answer is" + " pango"` turns "Repeat my sentences … plays many" into "what is *the answer*?". A natural reading of "the answer" is the famous ending.
- The plan already reports raw-completion DIRECT for families 1–5, but only as a sensitivity check. Pre-register instead one of:
  - C1/C2 must hold against **both** chat-DIRECT and raw-DIRECT; or
  - per family, against the better of the two.
- A third, natural chat analogue for completion tasks: prefill the assistant turn with the unfinished text, e.g. "Absence makes the heart grow", then score " heavy."/" fonder.".
- INSTR carries the same handicap, so the THINK − INSTR comparison does not fix this.

### S2. The THINK suffix changes what memo-trap and quote-repetition ask
- P' removes nothing here, because there is no trailing "Answer:". "…heart grow\n\nThink it through step by step. Then give your final answer…" turns a continuation task into a meta-question.
- That reframing may *be* the effect. If so, it is a "reformulate as a question" effect, not "thinking". A cheap rival would separate them: DIRECT with the instruction restated as a question ("Which word should come next: X or Y?"), no reasoning.
- At minimum, interpret per-family memo/quote gains with this caveat.

### S3. into-the-unknown clustering is wrong: cluster and sample by story
- 912 exact pairs differ only in the prefix "We know:" vs "Context:", with the same `answer_index` (912/912).
- There are 570 distinct context stories: 456 of size 2 and 114 of size 8 (4 option pairs × 2 prefixes).
- "question+options" gives 999 clusters, so it under-clusters the 8-blocks.
- A random 200 contains on average only about 163 distinct stories.
- **Fix:** sample at most one item per story (570 available, enough for 100 + 100 by position), or cluster by the context text.

### S4. Other clustering notes
- memo-trap by text after the colon is right: 877 clusters, max size 7. The size 2–7 clusters are "same quote, different target word" (Irish sign, separate-cars law, etc.).
- hindsight: all 315 test questions unique, so clustering by item is fine.
- repetitive-algebra: 999 unique test lines (one duplicated), so by item is fine.
- quote-repetition (report-only):
  - 2 exact duplicate prompts;
  - 138 distinct quotes, of which 77 appear 3× and one 6×.
  - Cluster it by quote, and drop the duplicates.

### S5. Ceiling and floor limit what C1 can detect
- Known DIRECT levels (different formats): repetitive-algebra 98.1% (watch_public), quote 86%, NeQA 61.5%, hindsight 46% (public/).
- Algebra can contribute at most about +2 to the equal-weight mean, so the mean effectively rests on 4 families.
- The +5 threshold on a family-equal mean can be passed by one family alone (e.g. memo-trap +25, others 0).
- **Suggest:** also pre-register "gain with low > 0 in ≥ 3 of 5 families", or a leave-one-family-out check.

### S6. Power of C1/C2
- With 200 per family and a 20–40% change rate, the 95% half-width of the 5-family mean is about ±2.7 to ±3.9. The cluster scheme widens it a little.
- So "low > 0" is not binding. The binding condition is point ≥ +5: power is 50% if the true effect is exactly +5, and about 95% at +8.
- That is acceptable if +5 is meant as a minimum interesting effect. Say so in PREREG.

### S7. Disclose the prior exposure of the main tasks
- "Never used before" is true only for memo-trap and into-the-unknown.
- DIRECT accuracy on NeQA, hindsight, algebra and quote has already been seen (public/, watch_public/, general-lamp/). hindsight was kept in watch_public because it moved favourably (audit §3d).
- No THINK/INSTR wording was tuned on them, so the independence claim for the *wording* holds. Still, family selection was informed by knowing DIRECT is weak there. State this in PREREG.
- redefine-math is correctly marked "near".

### S8. Data hygiene to fix in make_ti.py
- quote-repetition prompts all use "\r\n" (300/300), and 4 contain a BOM (U+FEFF) inside the quote. Normalise "\r\n" → "\n" the same way for every arm (public/ did this), and either keep or strip the BOM consistently.
- memo-trap has about 20 items built on segregation laws or a violent verse. The *right* answer is benign ("cats and dogs", "teach"), but THINK may refuse. Report per-family refusal/unparsed rates per arm, and either pre-register exclusion of this cluster or keep it with that report.
- sig-figs was fetched (20,897 rows, 3 classes) but is absent from the plan. Write down why it is excluded, before the run, to avoid a forking path.
- MMLU: `data/mmlu.jsonl` is present. A "fresh seed" does not mean disjoint items. Nothing was tuned on MMLU in thinkaloud, so disjointness hardly matters. Either drop the claim, or exclude the IDs used before (watch_public seed 185 and the first 10 per subject).

### S9. Truncation and "Answer:" inside the reasoning
- hindsight prompts contain many "Answer: Y/N" lines. The scorer uses `rindex` on the *generated* text only, which is correct; keep it that way, never on the prompt + generation.
- Report how often THINK outputs contain more than one "Answer:".
- Report MMLU truncation separately. Long math items may hit 512 tokens, and their forced choice after an appended "\nAnswer:" is then a guess.

---

## Evidence summary (python3 -I, full files)

- Class balance by `answer_index`:
  - memo 484/452;
  - itu 896/928;
  - NeQA 137/163;
  - hindsight 142/173;
  - algebra 503/497;
  - modus-tollens 0/1236;
  - quote 0/300;
  - pms 0/1428;
  - redefine 400/500.
- The trivial rules in B1 were computed over the whole file:
  - memo instruction-word match 833 right-only, 0 wrong-only;
  - hindsight opposite-of-outcome 315/315;
  - algebra majority-not-last 1000/1000;
  - itu low-overlap 69.3%.
- into-the-unknown:
  - prefixes "We know" 912 / "Context" 912;
  - pairs identical after the prefix 912/912, same answer 912/912;
  - stories 570, sizes {2: 456, 8: 114}.
- quote: duplicate prompts 2; distinct quotes 138; right word longer/shorter/equal in characters 127/40/133.
