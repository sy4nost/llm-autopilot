# Review of thinkaloud/PLAN.md v1, before code or money (2026-10-18)

Read: thinkaloud/PLAN.md; disguise/PLAN.md, PREREG.md, rezultaty.md, report_main.txt; asktwice/PLAN.md (v3.1), dev-itog.md,
dev_choice2.txt, review/code-review.md, gen_check.py; audit/audit-report.md. Local `python3 -I` reads of sets_dis.json and
out_dev_dev2.jsonl only. Nothing paid was run; PLAN.md was not edited.
Verdict: **do not code yet.** Three blockers: the lamp criterion passes for a surface reason, the accuracy comparison is not
fair (prompt, format and parse all differ), and no arm separates "thinking" from the instruction re-ask that already fixed traps.

## BLOCKERS

B1. **Criterion 3 is a surface detector and cannot fail for an interesting reason.** The "no-conflict" controls (R-same, K/D plain)
  have no counterfactual sentence at all; traps always have one. The rule S = "a redefinition/context sentence is present" scores
  100% on traps, 0% on R-same and K/D plain: a perfect criterion-3 PASS with no model. Any reasoning that restates the premise
  ("in this game a hexagon has 9 sides, instead of the usual 6") fires B1. It can also fail for a boring reason: N-confirm (counted as
  no-conflict) has a "Rule 2 reviewed" note, and rule-priority reasoning says "however"/"but" on almost every item.
  Fix: demote criterion 3 to a sanity check. The lamp claim must rest on (i) conflict-present controls where the habit is right:
  R-agree, N-agree, R-traps T0/T1/T2/T4/T8 (counterfactual present, DIRECT ~100% right). A lamp that fires there as often as on traps
  is a conflict detector, i.e. S; and (ii) criterion 4 (within-cell), with the rivals in R4.
B2. **DIRECT vs THINK is not a like-for-like comparison.** DIRECT = 2-3-option forced choice, prompt ends "Reply with only the
  answer, nothing else." + prefix "The answer is", mass-filtered (R kept 233 of 360 habit-template traps; K 76 of 91). THINK = free
  generation, different tail, no option list, "unparsed" scored wrong, no filter. Worse, N prompts say "Do not explain. Return only
  its label." twice; appending "think step by step" creates a contradictory prompt (the model may still not explain, or explain and
  truncate). Fix: (a) define the exact THINK prompt per family, with the "do not explain"/R_TAIL lines removed, and freeze the strings;
  (b) add **FREE-DIRECT**: the same edited prompt with "Give only the final answer on one line as: Answer: <answer>", same parser.
  THINK vs FREE-DIRECT isolates thinking from format and parsing; (c) make THINK-FORCED (forced choice over the same options after the
  reasoning, with the reasoning cut before its last "Answer:") a co-primary for criterion 1, not report only. A gain must hold in
  both parsed and forced scoring; (d) score all arms on the same item set: the DIRECT mass-filtered subset (primary) and all items
  (secondary, which needs DIRECT re-scored without the filter, as in asktwice v3.1).
B3. **The instruction re-ask baseline is missing, so "think aloud helps" is uninterpretable.** The INSTR line already gave R 38->54
  (T6 2->75) in disguise, and always-I was 100% on asktwice smoke traps. CoT prompts also tell the model to attend to the task text.
  Fix: add **INSTR-DIRECT** (forced choice; existing for R and N in out_main_main.jsonl "instr", new for K/D, cheap) and
  **THINK+INSTR**. Criterion 1 becomes: THINK gain >= +10 over DIRECT **and** THINK >= INSTR-DIRECT + 5 (low > 0), else the verdict
  reads "thinking helps, but no more than asking to follow the facts". THINK+INSTR is report only (is it additive?).

## REQUIRED

R1. **Reused and selected items: regression to the mean favours THINK.** Every trap was selected *because* DIRECT failed on it, on
  the same data: R habit templates T3/T5/T6/T7 = those < 85% right on this run; K v1/D v1 chosen on dev2 (K 83%, right at the gate);
  H cells and M threshold fitted on dev2; the idea itself came from the K smoke items (37/88 overlap dev). T3 (80%) and T5 (70%) are
  near the gate, so a fresh DIRECT pass on new items would itself "gain". Fix: build fresh trap sets and judge on them: new N seeds
  (make_native, skip all used seeds), new R instances of T3/T5/T6/T7 with new numbers (dedupe vs the 810), K/D only after the
  asktwice B1 generator widening. Re-run DIRECT on them in the same job. Reused items are report only. Exclude the asktwice smoke set.
R2. **Freeze the family set now, with no data gate.** "Families with a DIRECT habit" is decided on data already seen. List them now
  (R: T3, T5, T6, T7; N: D1, D3; K v1; D v1). If fresh DIRECT lands >= 85% in a family, report it and keep it in.
R3. **Separate arithmetic help from habit release.** T3 ("Redefine π as 715+245") and T5 ("963 mod 50") need computation; CoT
  improves arithmetic regardless of any habit. Add a compute-only control per T3/T5 item ("Q: What is the first digit of 715+245?"):
  if DIRECT already fails there, the T3/T5 "gain" is arithmetic. Report gains on T6/T7/K/D (pure recall habit) separately.
R4. **Criterion 4 needs stronger rivals than margin.** For DIRECT-error detection the lamp must beat, within cell:
  (a) **THINK != DIRECT** (free, same generation: if thinking changed the answer, the quick one was likely wrong); (b) INSTR != DIRECT;
  (c) disguise != DIRECT on R (out_main_main.jsonl; T3 0.94, T6 0.96 within template per posthoc); (d) S2 truth check. Require
  AUC(lamp) >= max(rivals) + 0.05, low > 0; else "the lamp is a weaker proxy for the answer flip". L-use (THINK accuracy by lamp)
  is near-circular, because the lamp and the answer come from the same text: report only, within cell.
R5. **Keyword list (B1 rule) has open degrees of freedom and copy hits.** "instead of" is in every R T1/T2/T8 prompt (checked: 90/90
  each); "standard", "typically", "usually", "however", "actually" are generic reasoning words; restating the prompt fires them.
  Fix: (a) match only text not copied from the prompt (drop any sentence with a >= 5-word overlap with the prompt) and freeze the
  tokenizer/regex (word boundaries, curly quotes); (b) split STRONG ("in reality", "real world", "in fact", "contradict", "premise",
  "conventional", "normally", "usually") vs WEAK (the rest); STRONG is primary, ALL is a sensitivity analysis; (c) report the firing
  rate per keyword x population; (d) hand-label 40 smoke reasonings blind (lamp yes/no) for precision/recall, report only. No list
  change after main data.
R6. **Parsing rules must be complete and frozen before smoke.** Specify: strip markdown (**, `), quotes, trailing period,
  "The answer is"; K letters ("'K'", "K."); N labels (case, plural); numbers by value ("9.0", "nine"? decide); multiple options in
  the final line (-> unparsed, or first?); hedged answers; truncation at 512 tokens (report the rate; N prompts are long). Gate: if
  unparsed + truncated > 5% in any family, that family's parsed verdict is INCONCLUSIVE and only THINK-FORCED counts.
R7. **Within-cell AUC sample sizes are too small as planned.** Define "cell" (R: template; N: D1/D3; K/D: hook word) and the minimum
  (>= 5 errors and >= 5 right). On the current DIRECT data: T6 has 1 right of 56, T7 0 of 90 (unmeasurable); T3 ~8 errors/33 right;
  T5 ~14/32; N D1 4 right of 90 (unmeasurable), D3 16 right; K ~4-6 cells with 2-3 errors each; D cube 10/10, ~4-6 cells measurable.
  That leaves ~60 minority cases in ~8-10 clusters; a binary lamp's AUC CI is about +-0.12, so "low > 0.55" needs a point ~0.68+.
  Fix: size the fresh sets so >= 6 cells meet the minimum (e.g. 180 instances each for T3/T5, 180 N sets for D3), state the
  expected count of measurable cells in PREREG, count NaN bootstrap replicates as failures, and tag resampled cells (asktwice R3 bug).
R8. **Control sizes cannot be reached without duplicates.** D plain has 11 distinct texts, K plain 27, D agree 33 in the whole
  template space (asktwice code-review B1). "~100 per family" would repeat identical greedy prompts. Use distinct texts only and report n.
  Bootstrap K/D by hook word, then item (the plan says "by item": near-duplicates make that too narrow).
R9. **Harm controls are missing the most relevant population.** CoT risks "correcting" a premise the model already followed. Add
  R-traps T0/T1/T2/T4/T8 (counterfactual present, DIRECT right) and K/D traps where DIRECT was right as reported harm cells; the
  criterion-2 list must be named exactly: R-same, R-agree, R-nohabit, N-agree, N-confirm, K plain, D plain, K agree, D agree.
R10. **Seeds and process (audit 3d).** The plan uses seed 12345 for control sampling and also "Alex's number" for it. 12345 is the
  smoke/dev seed everywhere. Use Alex's number for sampling and fresh generation; the smoke uses 12345 items that are kept out of the
  scored set. Write PREREG (prompts, parser, keyword regex, cells, scorer sha256) **before** the smoke, keep the smoke outputs, and
  log any post-smoke change as an amendment.
R11. **Cost.** With R1/B2/B3 the job is ~1500 x (THINK, THINK+INSTR) generations + FREE-DIRECT (short) + forced passes ~ 3500
  generations; still ~$5-8. Re-estimate in PREREG.

## OPTIONAL

O1. B2 judge: same model judging itself is biased toward "yes" after premise restatement; also run the judge on a shuffled
  reasoning (from another item of the same cell) as a placebo.
O2. A direct "lamp" arm that tests Alex's idea literally: "First say whether anything in the task differs from real-world knowledge,
  then answer." Report only; it separates "notice" from "think".
O3. Report THINK gains split by whether the reasoning restates the premise in its first sentence (copy vs reasoning).
O4. Report per-template and per-hook results, and the share of THINK answers equal to the habit option (wrong for a habit reason vs
  wrong otherwise / unparsed).
O5. Run DIRECT again in the same job for a 50-item integrity check (max log-prob diff vs the saved jsonl must be ~0).
O6. Written expectation: add "THINK ~ INSTR-DIRECT on K/D/T6; criterion 4 fails, because within a cell the reasoning is nearly the same
  for every item, so the lamp is ~constant" (within-cell texts differ only by lead word and value).

## v2 check (PLAN.md v2, 2026-10-18)
Verdict: **needs small changes (N1-N5), then ready to code.** All blockers resolved; no new blocker.

Resolved: B1 (criterion 3 removed; lamp judged only within cell vs rivals), B2 (same items, no filter, forced choice in all arms,
N "do not explain" lines removed, parsed answer reported), B3 (INSTR arm + criterion 2, THINKI reported). R1/R2 (all items fresh
from Alex's number, family list frozen), R3 (R-compute + adjusted gain), R4 (rivals incl. THINK != DIRECT, DISG, S2), R7 (cell
minimum 5/5, >= 6 cells, else INCONCLUSIVE), R8 (widened K/D generators, hook-then-item bootstrap), R9 (R-cfright, named control
list), R10 (PREREG before smoke, smoke kept). R5/R6 partly: see N3/N4. FREE-DIRECT was not added; acceptable now that forced
choice is primary in every arm.

NEW / remaining
N1. **Cell minimums are probably not reachable: likely INCONCLUSIVE by design.** Expected measurable cells (>= 5 err and >= 5 right),
  using known DIRECT rates: R T3 (~20% err) yes, T5 (~30%) yes, T6 (~2% right) no, T7 (0% right) no; N D1 (4% right of ~60) no,
  D3 (18% right of ~60, ~11) yes; K: 300 traps / 40 countries = 7.5 per hook, errors concentrated in a few countries (dev2 Russia
  3/4, Canada/Germany 2/4) -> 0-1 cells; D: 300 / 22 hooks = ~14 per hook, dev2 hexagon/octopus/square ~55-65% err -> ~3 cells,
  new hooks unknown. Total ~5-7, right at the >= 6 gate. Fix (choose and write it down before the data): put more items per hook
  (K: 20 countries x 15, D: 15 hooks x 20, same totals), and N 120 per probe for D3; print the expected cell count from dev2 rates
  in PREREG. Using dev2 rates to allocate is fine (dev, not test) if it is disclosed.
N2. **The compute adjustment is unpaired and its bootstrap is unspecified.** R-compute has 50 items per style, while traps have 120
  different items, so "trap gain minus compute gain" mixes two independent samples. Fix: give each T3/T5/T6 trap its own compute
  twin with the same numbers (120 each, short prompts, cheap), and bootstrap the per-item difference (trap gain_i - compute gain_i)
  within the template. Also report unadjusted gains. T7 needs no twin (its habit is division, the gold is a string digit): say so.
N3. **The forced-choice prefix is ambiguous and truncation is unhandled.** "(or the whole reasoning + "\nAnswer:" if none) +
  "Answer:"" reads as "Answer:Answer:". Write the exact string for both cases (e.g. cut = reasoning[:last "Answer:"], prefix =
  cut.rstrip() + "\nAnswer:", options " <opt>"). The cap is now 400 tokens (v1 had 512): report the truncation rate per family,
  plus criterion 1 with truncated items excluded (sensitivity). Report agreement between parsed and forced answers, and the share
  of free answers that are not among the options.
N4. **Keyword list: "actually", "standard", "typically", "commonly" are generic reasoning words.** For example "standard order
  of operations", "Actually, let's compute"; "typically" also fires on N rule talk. Counting beyond prompt occurrences fixes literal
  copies only, not paraphrased restatements. Fix: pre-register a sensitivity lamp without these four, and report criterion 4 for it.
  "approximately 3.14" / "2.718" fire when the model computes the habit value: these are R-only by nature, which is fine. The hand
  labels are not blind: hide the DIRECT answer and correctness from the labeller, at least.
N5. **Cost assumes good batching; verify it.** 4600 generations x <= 400 tokens is ~1.4M new tokens: ~$2-4 batched, but plain HF
  generate one prompt at a time at ~25 tok/s would take ~20 h (~$80). Require left-padded batched generation, and add a smoke check
  that batch vs single greedy outputs are identical on 20 prompts (bf16 padding can flip greedy tokens; report the mismatch rate,
  or else run with batch size 1 per length bucket). Set a hard budget cap (e.g. $10) in the run script.
Minor: family weighting is fine (K has only ~17 points of headroom, so the equal weight dilutes gains there; report per family).
R-compute in the harm list is expected to *gain*; harmless. Dedupe the fresh items against the 12345 smoke set and all dev sets,
and assert the counts.
