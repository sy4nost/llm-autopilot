# Plan v1: "think aloud first" as accuracy aid and as a verbal lamp (2026-10-18, Alex's idea)

## Why
asktwice/gen_check.py: when allowed to write freely, 14B often objects to a counterfactual premise ("The answer is incorrect
based on the given premise..."). Alex: let the model talk the solution through before answering; maybe that becomes a lamp.
Earlier related results: restate-rule-first lowered old answers 82->57% (in-context copy); two-pass draft+check broke right
answers (-14..-30%). So harm is measured separately.

## Model and arms (Qwen2.5-14B-Instruct, bf16, greedy)
- DIRECT: existing forced-choice answers where available (disguise/out_main_main.jsonl, asktwice/out_dev_dev2.jsonl); for new
  items the same F2 forced-choice scoring.
- THINK: chat(task + "\n\nThink it through step by step. Then give your final answer on the last line as: Answer: <answer>")
  greedy, max 512 new tokens. Final answer = text after the last "Answer:", matched to the options (exact after stripping
  spaces/quotes/period, case-insensitive; numbers by value). No match -> "unparsed" (counted wrong; rate reported).
- THINK-FORCED (report only): after the reasoning, forced choice among options with prefix "Answer:" (removes parse noise).

## Item families (all already built; nothing tuned on them for this method)
Traps (habit pulls, normal model often wrong):
- R: redefine-math test items of the habit templates T3, T5, T6, T7 (disguise/sets_dis.json, R-traps).
- N: road words with change note, D1/D3 (disguise/sets_dis.json, N-traps).
- K: capitals first letter (asktwice dev2, K v1 traps). D: defined words direct (asktwice dev2, D v1 traps).
Controls (habit is right or no conflict):
- R-same, R-agree; N-agree, N-confirm; K/D plain (dev2 plain v1); new K/D agree items (true context, same question).
Sizes: R traps 270 (T3/T5/T6/T7, sampled to <= 90 each after DIRECT is known: all), N traps 180, K 76+, D 89+; controls
~100 per family (sampled with seed 12345 from the files). ~1500 generations.

## What we measure
A. Accuracy: THINK vs DIRECT on traps (gain) and controls (harm), per family, families equal weight.
B. Verbal lamp = the reasoning mentions a conflict between the task's statement and normal knowledge. Detector frozen now:
   B1 keyword rule on the reasoning (case-insensitive): "in reality", "real world", "actually", "normally", "usually",
      "in fact", "conventional", "standard", "typically", "premise", "contradict", "however", "but in", "instead of".
   B2 judge (report only): the same model reads its reasoning and answers yes/no to "Does this reasoning point out that something
      in the task differs from normal, real-world knowledge?"
   Lamp quality:
   - L-fire: B1 rate on traps vs on controls where nothing conflicts (R-same, K/D plain, N-confirm? -> N-confirm has a
     note but no change; counted as control).
   - L-use: among traps, accuracy of THINK when the lamp fired vs when it did not; within-cell AUC of "lamp off" for THINK
     habit errors (does silence predict the habit error?).
C. Lamp as a detector for the DIRECT answer: within-cell AUC of B1 for DIRECT habit errors (does a fired lamp in the
   thinking run mark the cases where the quick answer was a habit error?), vs confidence margin of the DIRECT answer.

## Criteria (pre-registered; 95% CIs, paired bootstrap by set (N) / item within template (R) / item (K, D), 2000 reps)
1. THINK gain on traps (families with a DIRECT habit, equal weight) >= +10 points, low > 0.
2. THINK harm on every control family >= -5 points, low >= -10.
3. Verbal lamp fires on traps at least 2x as often as on no-conflict controls, and on no-conflict controls <= 20%.
4. Lamp as a detector for DIRECT errors (C): within-cell AUC >= 0.70, low > 0.55, and > margin AUC by >= 0.05.
Methods are judged separately: "think aloud helps" = 1 & 2; "verbal lamp exists" = 3 & 4.

## Written expectation
Gain on K/D (direct restatements) and R T6; little on T7 ("random digit" vagueness) and N (road words, where even disguise
hurt). Lamp: fires on most traps (model says "normally ... but here"), so criterion 3 likely passes; criterion 4 uncertain,
because the lamp may fire on all traps regardless of whether the quick answer erred.

## Cost and order
~1500 generations x ~300 tokens, batched greedy on H100: ~$2-4. Plan -> independent review -> code -> smoke on 20 items ->
PREREG with sha256 -> Alex's number (sampling seed for controls/new agree items) -> run -> frozen score -> report.
