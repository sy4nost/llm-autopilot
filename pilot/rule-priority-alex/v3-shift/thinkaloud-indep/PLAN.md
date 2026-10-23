# Think-aloud on independent public tasks — PLAN v1 (2026-10-18)

## Question (Alex)
Does "think it through first" (THINK) and "think it through + follow the facts stated in this task" (THINKI) reduce habit
errors on public trap tasks that we did NOT build or tune on, without hurting ordinary questions?

Prior result (thinkaloud/rezultaty.md, our own items): THINK traps +34 vs DIRECT; THINKI ~96-100% on all traps with no harm;
"THINK answer != DIRECT answer" flagged DIRECT errors with within-cell AUC 0.90.

## Model and arms (frozen from thinkaloud/, nothing re-tuned)
Qwen2.5-14B-Instruct, bf16, H100, greedy. Same code paths as thinkaloud/modal_ta.py.
- DIRECT: `chat(P + "\n\nReply with only the answer, nothing else.") + "The answer is"`, options `" " + cls.strip()`.
- INSTR:  same as DIRECT with `INSTR + "\n\n"` prepended to P.
- THINK:  `chat(P' + THINK)`, greedy up to 512 tokens, then forced choice after the last "Answer:" (or appended "\nAnswer:").
- THINKI: same as THINK with `INSTR + "\n\n"` prepended.
- P = task prompt as published; P' = P with a trailing "Answer:" / "A:" removed.
- INSTR = "Follow the facts stated in this task exactly, even if they contradict what you know." (unchanged wording)
- THINK = "\n\nThink it through step by step. Then give your final answer on the last line as: Answer: <answer>"

## Tasks (all published by others; Inverse Scaling Prize rounds 1-2 + MMLU)
Main (both classes present, sampled class-balanced 100 + 100 so "always one class" = 50%):
1. memo-trap (famous phrase vs instruction to end differently) — pretrained-pull habit. Never used before.
2. NeQA (negated common-sense question) — pretrained pull. Used before only as a probe test, never tuned on.
3. hindsight-neglect-10shot (judge the bet, not the outcome; 10 examples) — in-context pattern. All 315 used (173/142).
4. into-the-unknown (pick information that is new, not the one repeating the context) — never used before.
5. repetitive-algebra (many examples with the same answer, then a different one) — in-context copy.
Report-only (only one correct class in the whole dataset, so a gain could be a plain shift toward the odd option):
6. modus-tollens (all "Yes"), 7. quote-repetition (all copy the odd word), 8. pattern-matching-suppression (all "break the pattern"); 150 each.
Near, report-only: 9. redefine-math 150 (our R templates in thinkaloud/ were taken from it, so not independent).
Harm check: 10. MMLU test, 300 questions, zero-shot, fresh seed (different from 185 used in watch_public).
Total ~2,000 items -> 2 generations each; estimated $4-6 (thinkaloud: 2,060 items ~ $4).

Sampling seed = Alex's number. Items drawn before any model call; sets file sha256 recorded.

## Pre-registered criteria (main families 1-5, equal weight; bootstrap 2000, item scheme and cluster scheme, wider interval)
Clusters: memo-trap by the phrase after the colon, into-the-unknown by question+options, others by item.
- C1 THINK − DIRECT ≥ +5 points, low > 0.
- C2 THINKI − DIRECT ≥ +5 points, low > 0.
- C3 harm on MMLU: THINK − DIRECT and THINKI − DIRECT each ≥ −2 points with low ≥ −5.
- Verdict "think-aloud helps on independent tasks" for an arm = its C1/C2 PASS and its C3 PASS.
- Secondary (reported with intervals, no pass mark): THINK − INSTR, THINKI − INSTR, THINKI − THINK; per-family deltas;
  number of main families with a reliable loss (high < 0).
- C4 detector: flag = (THINK answer != DIRECT answer), predicting "DIRECT is wrong"; per-family AUC (= balanced accuracy for a
  0/1 flag), mean over main families ≥ 0.70 and ≥ rival "low DIRECT confidence" (margin, per-family AUC) + 0.05. MMLU false alarms reported.

## Checks against known pitfalls (from audit/ and earlier reviews)
- Class prior: main families class-balanced; single-class families report-only.
- Isomorphism: none of the main tasks were used to tune THINK/INSTR wording (wording frozen from thinkaloud before seeing these).
- Trivial baselines: INSTR alone is a rival; "always one class" = 50% by construction.
- Truncation: share of THINK outputs hitting 512 tokens reported per family; truncated items kept (counted as scored).
- Format: DIRECT keeps the published prompt plus our answer tail; a raw-completion DIRECT (no chat) reported for families 1-5 as a sensitivity
  check, so a gain is not only "chat format hurt DIRECT".

## Steps
plan review → make_ti.py + modal_ti.py + score_ti.py → smoke (seed 12345, 10 per family) → code review → PREREG.md with sums →
Alex's number + OK on cost → run → frozen score → rezultaty.md in plain Russian.

---
# PLAN v2 (after review/plan-review.md; v2 overrides v1 where they differ)
Blocker fixes:
1. Habit-right controls (B1). From items NOT used as traps (other clusters), 80 per family, mechanically changed so the habit
   answer is the right one: memo-trap: instruction word replaced by the famous ending; NeQA: the single negation removed
   ("is not"/"isn't"->"is", "aren't"->"are", ...; key flips); hindsight: the 10th in-context example moved into the test slot
   (9 examples remain; its given answer is the key); repetitive-algebra: first constant of the test equation changed so the answer
   equals the last example's answer (lin.py, solution re-checked). into-the-unknown: no mechanical control (reported as such).
2. Every main family sampled 60 + 60 by answer_index (hindsight too) -> 120 traps each (B2).
3. C4 detector is report-only, needs >= 5 DIRECT errors and >= 5 right per family (B3).
4. MMLU harm check 1000 items (B4).
5. THINK/THINKI answer = option parsed from the final "Answer:" line (numbers: first matching number; letters/Yes/No: first
   matching token; completion tasks: last matching word), forced choice as fallback; parse rate and parse-vs-forced agreement
   reported per family. Completion tasks (memo, quote, pattern): DIRECT's assistant turn is prefilled with the unfinished text
   and the option completes it; THINK's forced fallback likewise "Answer: <unfinished text>" + option (B5).
Suggestions taken: gain must hold against three DIRECT variants (chat summed, chat length-normalised, raw completion without
chat); gain in >= 3 of 5 families; into-the-unknown clustered by story; hindsight by test question; quote-repetition cleaned
(\r\n, BOM, duplicates, items with identical options dropped); memo-trap restricted to the 'Write a quote that ends in the word'
items (drops the 90 "sign" items incl. the segregation ones). sig-figs fetched but excluded: it tests a rounding skill, not a habit.
Prior exposure disclosed: DIRECT on NeQA, hindsight, algebra, quote, redefine was seen in earlier runs (never tuned on).
Not taken: a "rephrase as a question, no reasoning" rival (DIRECT and INSTR already ask for the answer as a question).

Criteria v2 (main = 5 trap families, equal weight; bootstrap 2000, item + cluster schemes, wider interval):
- C1 THINK - DIRECT >= +5, low > 0, against EACH of the 3 DIRECT variants; and THINK > DIRECT in >= 3 of 5 families.
- C2 same for THINKI.
- C3 for each arm: controls (4 families) arm - DIRECT >= -5 with low >= -10; MMLU arm - DIRECT >= -2 with low >= -5.
- Verdict per arm "helps on independent tasks" = its C1/C2 and its C3 all PASS.
Size: 600 traps + 320 controls + 400 report-only + 1000 MMLU = 2320 items, est. $5-7.

# PLAN v2.1 (after review/code-review.md)
- Parser: completion tasks use ONLY the final word of the "Answer:" line (not an option -> forced choice); letter options match a
  standalone capital A-D (first one); numbers = last number on the line; Y/N and Yes/No mapped to each other.
- All PASS thresholds compared after rounding to 4 decimals.
- C3 controls additionally: every control family arm - DIRECT >= -10 (point).
- Controls 40 + 40 by right option. into-the-unknown: only 114 stories -> at most one item per (story, answer side), 50 + 50 traps.
  quote-repetition clustered by quote. Main set excludes every smoke item and every smoke control's source; MMLU deduplicated.
  Not excluded: MMLU items of watch_public (drawn inside Modal earlier; overlap expected ~3%, harmless for a harm check).
- Truncation flag counts <|endoftext|> as a stop. Generations saved only at the end (3 h cap; earlier runs of this size finished).
- The smoke batch-vs-single check (15/20 same last line, mostly free-text memo sentences) is bf16 drift; THINK results carry this noise.
