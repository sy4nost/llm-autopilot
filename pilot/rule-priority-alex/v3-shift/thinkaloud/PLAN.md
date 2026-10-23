# Plan v2: "think aloud first" (2026-10-18). v1 = PLAN_v1.md; review/plan-review.md (3 blockers, 11 required) addressed here.

## Question
Alex: if the model first talks the solution through, (A) does it make fewer habit errors, and (B) does its talk contain a
"lamp" (it says the task differs from normal knowledge) that marks the cases where its quick answer is a habit error?

## Model, arms (Qwen2.5-14B-Instruct bf16, greedy). ALL arms scored on the SAME items, forced choice among the options,
no option-mass filter (mass reported).
- DIRECT: the family's direct prompt; R/K/D: chat(task + "\n\nReply with only the answer, nothing else.") + "The answer is";
  N: the original road-word prompt (asks for the label only) + chat assistant start.
- INSTR: DIRECT with the line "Follow the facts stated in this task exactly, even if they contradict what you know." inserted
  before "Q:" (R/K/D) or before "Classify the TEST case below." (N).
- THINK: think prompt = task with every "answer only / do not explain" line removed (N: "Do not explain. Return only its
  label." and "Return only the label."; R/K/D: no tail) + "\n\nThink it through step by step. Then give your final answer on
  the last line as: Answer: <answer>". Greedy, max 400 new tokens. Then forced choice: prefix = prompt + reasoning up to the
  last "Answer:" (or the whole reasoning + "\nAnswer:" if none) + "Answer:", options " <opt>". Parsed free answer reported.
- THINKI: THINK with the INSTR line. (report + criterion 2 context)
- DISG (R only): disguise/make_dis.disguise_R module text, DIRECT format.  S2 (items with a counterfactual): "Is the first
  sentence of this task true in the real world? Reply yes or no." forced yes/no.

## Items: ALL FRESH, generated with Alex's number (none seen before)
R (new redefine-math instances, templates copied verbatim from the dataset, new random numbers; generator in make_ta.py):
- R-traps: T3 "Redefine π as a+b", T5 "Redefine π as a mod b", T6 "Redefine e as a+b", T7 "View ÷ as a random digit" - 120 each.
- R-cfright (counterfactual present, habit right so far): T0, T1, T2, T4, T8 - 50 each.
- R-agree: T4-style "Redefine π as 3xx" and T6-style "Redefine e as 2xx" (gold = habit digit) - 60 each.
- R-compute (arithmetic confound): "What is the first digit of a+b?" / "of a mod b?" - 50 each for T3, T5, T6 number styles.
N: road words (native/make_native.py change_0) on 2 fresh generator seeds from Alex's number: traps D1/D3 (120),
   N-agree D2/D4 (120), N-confirm (Rule 2 reviewed and unchanged; 60 sampled).
K: capitals "first letter" (asktwice K v1) with a widened generator (40 countries, 4 leads, 3 question forms): traps 300,
   agree (true capital stated) 100, plain 40. D: defined words direct (asktwice D v1, 22 hooks, 2 question forms): traps 300,
   agree 100, plain 40. No data-based gate: the family list is frozen now.

## Verbal lamp (frozen detector on THINK reasoning)
Strong keywords (case-insensitive, counted only beyond the number of times the same phrase occurs in the prompt):
"in reality", "real world", "real-world", "actually", "normally", "usually", "in fact", "conventional", "typically",
"commonly", "well-known", "well known", "premise", "contradict", "differs from", "approximately 3.14", "2.718", "standard".
Weak (report only): "however", "but". Lamp = any strong keyword. Also report: firing rate per population; a hand-labelled
sample of 40 reasonings (labelled by Claude after the run, not blind; report only); judge B2 dropped (cost).

## Criteria (95% CIs; paired bootstrap: R by item within template, N by set, K/D by hook then item; wider of cluster/item)
1. THINK gain over DIRECT on traps >= +10 points, low > 0. Families equal weight; R = mean over templates, where T3/T5/T6
   gain is adjusted: trap gain minus the same-style R-compute gain.
2. THINK gain over INSTR on traps >= +5, low > 0 (same weighting, no compute adjustment).
3. Harm: THINK - DIRECT on each control family (R-cfright, R-agree, R-compute, N-agree, N-confirm, K agree, K plain,
   D agree, D plain) >= -5, low >= -10.
4. Lamp as a detector of DIRECT habit errors on traps: within-cell AUC (cells: R template, N probe, K/D hook; a cell counts
   if >= 5 errors and >= 5 right; need >= 6 cells) >= 0.70, low > 0.55, and >= the best rival + 0.05, low > 0.
   Rivals (within cell): DIRECT margin; THINK answer != DIRECT; INSTR answer != DIRECT; DISG != DIRECT (R); S2 "no".
   Fewer than 6 cells -> "not measurable" (INCONCLUSIVE).
Verdicts: "thinking helps" = 1 & 2 & 3; "verbal lamp" = 4.

## Written expectation
THINK beats DIRECT on K/D and R T6; ties INSTR there; little on T7 and N. The lamp fires on most traps (says "normally..."),
so it barely separates errors from right answers within a cell; criterion 4 likely fails or is not measurable.

## Cost and order
~2300 items, 2 generation arms (~4600 x <=400 tokens, batched) + ~6 forced passes: ~$5-7 on H100.
PREREG with sha256 BEFORE the smoke; smoke on seed 12345 small set kept; Alex's number; run; frozen score; report.

## v2.1 (review "v2 check" N1-N5)
- N1: K uses the 20 dev2 countries x ~15 traps, D 15 hooks (11 dev2 + pentagon, octagon, rectangle, starfish) x ~20 traps;
  more lead/question phrasings so items stay fresh after de-duplication against asktwice dev1/dev2, the smoke set and the
  redefine-math dataset. Allocation was set using dev2 error rates. Expected measurable cells (>= 5 habit errors and >= 5
  right in DIRECT), from earlier rates: R T3, T5; N D3; K 0-2; D 2-4 -> about 5-8; criterion 4 may end INCONCLUSIVE.
- N2: every T3/T5/T6 trap has a paired compute twin with the same numbers ("Q: What is the first digit of 244+230?");
  the adjusted gain is per-item (trap gain - twin gain), twins resampled with their trap. Unadjusted gain reported.
- N3: forced prefix after reasoning = chat(think) + reasoning up to the last "Answer:" + "Answer:"; if no "Answer:":
  chat(think) + reasoning.rstrip() + "\nAnswer:". Options " <opt>". Max 512 new tokens. Truncation rate per family,
  criterion 1 without truncated items, parsed-vs-forced agreement: report only.
- N4: sensitivity lamp without "actually", "standard", "typically", "commonly" (criterion 4 reported for it). Hand-labelling dropped.
- N5: batched left-padded generation (32 per batch); smoke compares batched vs single greedy on 20 prompts; function timeout 3 h
  (cap ~$12).
