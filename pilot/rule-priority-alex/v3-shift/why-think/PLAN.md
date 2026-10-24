# Why does thinking aloud fix "past outweighs the text" errors? — PLAN v1 (2026-10-24), exploratory

Alex 2026-10-24 01:09Z "Ну, проверь его" about Claude's explanation of why reasoning helps:
 (M1) more compute: every written word is one more full pass;
 (M2) small steps: each step is an easy vote that the old examples barely affect;
 (M3) copy-own-conclusion: the final vote copies the conclusion written just before it, so the copying machinery now works for
      the new rule; the old examples lose their pull once the reasoning is there.
No generation. We reuse the saved reasoning texts of conftrap-lamp (seed 57, ../conftrap-lamp/out_main_*.jsonl, gen_think) and
re-read the final answer under edited reasonings / attention blocks. Forced choice exactly as conftrap-lamp after(): chat(think
prompt) + reasoning up to the last "Answer:" + "Answer:", option scores. One forward pass per arm: options compared by the log-prob of
their first token after " " (items whose options share a first token use the full summed log-prob of each option, extra passes).

## Population
Main: "fixed" traps = DIRECT answer is the habit option AND conftrap-lamp THINK forced-choice answer is right. Families hindgen,
redef, parcels, bonus, D; models Qwen2.5-14B, Phi-4, OLMo-2-13B; at most 60 per model x family (random, seed 0).
Also 20 controls per family (habit answer right) where DIRECT and THINK are both right, to see that arms do not just break answers.

## Arms (read-outs; share right / share habit answer)
FULL     saved reasoning (must reproduce THINK right ~100% on fixed items: pipeline check)
NONE     no reasoning: think prompt + "Answer:" (expected ~ habit, like DIRECT)
FILLER   reasoning replaced by the same number of tokens of " ." (M1: compute without content)
T25/T50/T75  reasoning cut at 25/50/75% of its tokens, then "\nAnswer:" (when does the fix appear: gradually (M2) or at the end)
NOLAST   reasoning without its last non-empty line before "Answer:" (the written conclusion)
SWAP     last line with the right option label replaced by the habit label (only items where that line names the right label and
         not the habit label; count reported) (M3: does the final vote follow the written conclusion against the steps above it?)
Attention blocks on FULL (additive 4D mask, eager attention; rows = the final "Answer:" tokens; mask_check vs plain run reported):
KO_EX    answer rows cannot see the solved examples / shots (parcels, bonus, hindgen) (M3: examples lost their pull?)
KO_LAST  answer rows cannot see the last reasoning line
KO_STEPS answer rows cannot see the reasoning except the last line
KO_RULE  answer rows cannot see the new-rule sentence (parcels/bonus) or the definition sentence (D, redef)

## Predictions (Claude, written before any run)
M1: FILLER ~ NONE (no fix from compute alone). M2: T25..T75 rise gradually, not only at the end. M3: SWAP gives the habit/swapped
answer on most qualifying items; NOLAST stays mostly right (steps carry it) or drops (only the conclusion carries it) - open;
KO_LAST drops less than SWAP (other lines also name the answer); KO_EX keeps right (examples no longer pull); KO_RULE keeps right.
Exploratory, report-only: no pass/fail verdict; numbers with item-bootstrap 95% intervals, per model and family.

## Threats
Saved reasonings were produced by the same model; edited texts are off-distribution (SWAP contradicts its own steps).
First-token read-out differs slightly from summed log-prob; FULL vs conftrap THINK forced answer agreement is printed.
Blocking only the answer rows does not undo what the reasoning tokens already absorbed from the examples (KO_EX tests only
the direct path at the answer).
Cost: ~10k forward passes on L40S, ~$1-1.5.

# PLAN v2 (after review/plan-review.md; overrides v1). Written after the smokes (pipeline only, 8-10 items per model), before the main run.
- B1 read-out: summed log-prob of every option " "+c (options batched in one pass; padded rows are never read). First-token read-out dropped
  (for redef/bonus/D the first token is a bare " ").
- B3 endings: every trailing "Answer:" stub stripped from the saved reasoning; all arms end with "\n\nAnswer:" (NONE: "Answer:" right
  after the chat prefix). FULL vs conftrap's THINK answer agreement is implied by "FULL right" on fixed items (smoke 100%).
- Conclusion line = last line naming any option label (else last non-empty line). NOLAST drops it and everything after; KO_LAST blocks it
  and everything after; KO_STEPS blocks everything before it. T25/T50/T75 cut at line ends; we record whether the right label was
  already written before the cut.
- B4 main measure = margin logp(right) - logp(habit) besides share right; knockouts run on NONE too (KO_EX_N, KO_RULE_N).
- New arms: FILLTXT (natural filler " Let me think about this carefully." of the same token length), REDACT (every option label in the
  reasoning replaced by "[?]": steps stay, written answers go = M2 vs M3), SWAP_ALL (right label -> habit label everywhere),
  SWAP3 (conclusion line -> the third option), SWAP on controls (to another option). SWAP impossible for hindgen (labels never written).
- Knockout rows = every position from the final "Answer:" to the end, including option tokens.
- Population cut to <= 20 fixed + 8 controls per model x family (cost; smoke 11 s per item on L40S) -> ~$2 total.
- Smoke mask_check (batched padded masked run vs single plain run): 0-0.6 nats, bf16 batching noise; FULL margins ~12-17 nats.
Exploratory, report-only, as v1. Note: the smoke outputs (~25 fixed items) were seen; the main set re-samples and may include them.
