# Review of why-think/PLAN.md v1 (independent auditor, 2026-10-24, pre-run)

Measured locally on the saved conftrap-lamp files (out_main_*.jsonl, sets_cl.json). Scripts: scratchpad only, nothing run on GPU.
"fixed" = DIRECT argmax (summed log-prob) is the habit AND argmax of res.think (summed) is the key, as in the plan.

## Population (fixed traps, from saved data)

| model | hindgen | redef | parcels | bonus | D | total |
|---|---|---|---|---|---|---|
| qwen14 | 50 | **11** | 74 | 93 | **21** | 249 |
| phi4 | 83 | 32 | 40 | 59 | 52 | 266 |
| olmo13 | 48 | 59 | 43 | 47 | **27** | 224 |

739 in total, 669 after the cap of 60. 10 of 15 cells are under 60, so the cap only trims 5 cells. qwen-redef (11) and qwen-D (21)
give bootstrap intervals too wide to read; report them pooled or mark them "n<30".
Controls (DIRECT and THINK both right): hindgen/parcels/D 35-40 per cell, bonus 39/39/27. **redef has no control items at all**
(the dataset has none), so "20 controls per family" cannot hold for redef.
No fixed item has a truncated reasoning (trunc_think = 0 everywhere).

## BLOCKERS

1. **The first-token read-out does not work for redef, bonus and D (3 of the 5 families).** In all three models the options are
   " 7", " 11" and so on, and the first token is a bare " " (Qwen splits each digit; Phi-4 and OLMo-2 put the space in its own
   token before a number). In the saved res.think, the first-token log-prob is the same for every option on 100% of redef, bonus
   and D rows (about 0.0). Under the plan's fallback rule, these families would all go to the summed multi-pass path, so the claim
   of "one forward pass per arm" and the cost estimate are both wrong.
   Fix: read the **digit token** at the position after the " " token. To do this, build the input ids exactly as
   ids(prefix + " " + option) and read at index len(ids(prefix + " ")). Do not tokenize "Answer: " with a trailing space as a
   string, and check that the ids match. Checked on saved data, the digit-token argmax matches the summed argmax on 100% of rows
   for redef and bonus (all models), on D for Phi-4 and OLMo, and on D for qwen in 138 of 140 rows. In D, 9 of 140 items have
   options that start with the same digit ('11' vs '1x'), which matters for Qwen only; use the summed path for those.
   For parcels and hindgen the first token is valid: its argmax equals the summed argmax on 140/140 rows in every model, and only
   1 parcels row has two options with tied first-token log-probs.
2. **SWAP is impossible for hindgen: 0 of 181 fixed items qualify.** The reasonings never write "Y" or "N". They end with "...made
   the right decision" or "did not make the right decision". To keep hindgen you would have to edit the meaning (negate the
   sentence), and that is a different manipulation. Drop hindgen from SWAP, or pre-register the negation edit as its own arm.
   For the other families, using a robust last-line rule (item 3), the number of items whose last line names the right label and
   not the habit label is:
   qwen redef 9/11, parcels 50/74, bonus 89/93, D 10/21; phi4 redef 22/32, parcels 39/40, **bonus 23/59**, D 37/45;
   olmo redef 52/59, parcels 34/43, **bonus 19/47**, D 22/27. That is 406 of 739 overall.
   If you use the last line that **names any option** instead of the last line itself, the count rises to 507 (for example phi4
   bonus 59/59 and olmo bonus 46/47). The reason is that many last lines are "Therefore, the final answer is:" and contain no
   label. Pre-register which of the two definitions you use. They test different things.
3. **The "last line" and "Answer:" boundary is broken for OLMo.** OLMo often writes "Answer: Answer: POST". On fixed items this
   happens in 19 redef, 29 parcels and 3 bonus items. after() cuts at the last "Answer:", so the saved FULL prefix ends
   "...\n\nAnswer: Answer:", and the naive last line is the stub "Answer: ". As a result, NOLAST would delete only the stub,
   KO_LAST would block only the stub, and FULL would have a different ending from NONE, FILLER and T*.
   Fix: strip every trailing `(\**Answer:\**\s*)+` from the reasoning, then append one fixed separator ("\n\nAnswer:") in **every**
   arm. The plan currently mixes g[:rindex]+"Answer:" for FULL with "\nAnswer:" for the T* arms. Recheck that FULL reproduces the
   saved answer after this normalization.
   Define the last line as the last line that contains a letter or digit, after the strip. Phi-4 writes
   "Therefore, the answer is:" before the answer line on many items, so on those items NOLAST and KO_LAST remove a line with no
   content, and the arm tests nothing (see the counts in item 2).
4. **The effect measure has a ceiling, and the knockouts have no reference condition.** By construction FULL is 100% right on the
   fixed items. Shares can therefore only fall, and "KO_EX keeps right" cannot be told apart from a ceiling effect. ../inside
   already showed that without reasoning, blocking the examples at the answer *restores* the rule (37% vs 38% "no history"). So
   "KO_EX on FULL stays right" is predicted by both M3 and not-M3.
   Fix: (a) use the per-item margin, logit(right) - logit(habit), as the primary measure, and report shares as secondary. (b) Run
   every knockout on **NONE as well as FULL** (KO_EX×NONE, KO_RULE×NONE). M3's claim that "examples lose their pull" then becomes
   testable: the KO_EX shift in margin toward the right answer should be large in NONE and near 0 in FULL.

## SHOULD FIX

5. **The SWAP target is confounded with the habit prior.** If SWAP produces the habit answer, the cause could be copying the
   conclusion, or the habit prior winning once the right label is gone. Add **SWAP3**: swap to the third option, the one that is
   neither right nor habit. parcels, bonus and D have 3 options; redef and hindgen do not. Copying predicts SWAP3 gives the third
   option; the habit-prior story predicts the habit. Also run SWAP on **control items**, which have no conflict, to measure baseline
   copying. And add **SWAP_ALL** (replace every mention of the right label). Earlier lines name the right label on most items: for
   example qwen bonus 66/93, phi4 59/59 and olmo 43/47 have it before the last line. SWAP_LAST alone therefore pits one line against
   several, and its result depends on how many mentions there are rather than on the mechanism.
6. **The T25/T50/T75 arms do not separate M2 from M3.** A rise across the cuts could just mean the right label has already been
   written partway through. Cut at line or sentence boundaries, not at token fractions, because a mid-word cut followed by
   "Answer:" is off-distribution. Log for each cut whether the prefix already names the right label or the habit label, and analyse
   P(right), or the margin, against "label already written: yes/no" and against fraction of steps. Add a **REDACT** arm: the full
   reasoning with every mention of the right and habit labels replaced by a neutral placeholder (for example "[X]", same token count
   if possible). If REDACT stays right, the steps carry the answer (M2). If it collapses, the written label does (M3). This is the
   cleanest M2-vs-M3 contrast available without generation.
7. **FILLER is not a clean test of M1.** (a) Models are not trained to use " . . ." (dot-by-dot results), so FILLER ≈ NONE is
   expected whether or not M1 is true, and a null result says little. (b) FILLER also moves the examples and the rule further from
   the answer, so any FILLER effect mixes compute with recency/position. Add a length-matched **content-free natural text** filler,
   for example a restatement of the task's rules or a reasoning from a different family's control item with its labels redacted,
   to separate "more tokens" from "off-distribution dots". State in the write-up that M1 is weakly identifiable here.
8. **The knockout rows must include every readout position.** For the digit read-out, the appended " " token is a readout row and
   must be blocked too. For summed multi-token options (D for qwen, multi-token parcels labels), the option-token rows also read
   the prompt. Mask those rows as well, or the knockout leaks. In hindgen the prompt contains 11 "Answer: Y/N" shot lines, so
   locate "the final Answer:" by its token index after the reasoning, never by searching the text.
9. **KO_RULE in redef and D is near-certain to show nothing.** The reasoning restates the definition: D's last lines say "the story
   defines a 'hexagon' as having 7 sides", and the habit label appears in 45/52 phi4 D reasonings. Blocking the original sentence at
   the answer rows is bypassed by that restatement. It only means something next to KO_RULE×NONE, and next to KO_STEPS or REDACT.
10. **Span location.** These anchors work on every item (checked): parcels and bonus have exactly one "solved examples" line, one
    "IMPORTANT UPDATE", one "TEST CASE:" and 60 "Example " lines. Examples span = from the "Below are 60 solved examples" line to
    the end of Example 60. Rule span = the IMPORTANT UPDATE line. hindgen = 11 "Question:" blocks, where shots = everything before
    the last "Question:". redef and D = one sentence followed by " Q:" (always exactly one "Q:"), where rule = text before "Q:".
    Map characters to tokens with offset_mapping on the *chat-templated* string, and assert the decoded span equals the anchor
    text in each model's tokenizer.
11. **NONE is not DIRECT.** NONE puts "Think it through step by step..." in the user turn and then forces an immediate "Answer:".
    That is an off-distribution prefix, and it can differ from DIRECT, which used its own tail or stem. Report NONE against the
    saved DIRECT, and use NONE (not DIRECT) as the zero point for normalised effects: (arm - NONE)/(FULL - NONE) on the margin.
12. **Selection and near-ties.** Items were selected on DIRECT = habit and THINK = right. Many will sit near a tie, so small
    numerical changes can flip them: eager vs sdpa attention, a single pass vs per-option passes, the 4D mask. Report the FULL
    flip rate and mask_check per model, and drop or flag items with |FULL margin| < 1 nat.

## NOTES

- FULL ≈ 100% is true by construction (it is the selection criterion), so it only checks numerics and prefix building, not the
  pipeline's validity. Also print agreement with the conftrap THINK *parsed* answer, which the original study used as THINK.
- Positional confound: NOLAST, SWAP edits and knockouts keep positions (masked tokens still occupy RoPE slots), but NOLAST and the
  T* arms shorten the sequence. That is fine, but keep the readout separator identical across arms.
- The habit label is **never** written in any bonus reasoning (0/199 fixed). In hindgen, Y/N are never written. In the other
  families it is common (D 18/21, 45/52, 19/27; phi4 parcels 40/40). This affects what SWAP and REDACT can mean per family, so
  report per family, not pooled.
- KO_STEPS and KO_LAST block only the direct path. The last-line tokens have themselves attended to the steps, which the plan
  already flags for KO_EX. Word the conclusions as "direct read at the answer position".
- Cost: about 909 items (669 fixed + about 240 controls, redef has none) × 12 arms ≈ 11k passes. With the digit read-out fix it
  stays about 1 pass per arm. The additions above (NONE×KO, SWAP3, SWAP_ALL, REDACT, natural filler) add about 6-7 arms, roughly
  +60%, still under $3 on an L40S.
- The predictions are written before the run. Good. Add the predicted direction on the **margin** for each arm, not only shares.
