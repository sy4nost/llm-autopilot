# pull — measuring from outside how hard each force pulls (Alex, 2026-10-24: «Ну давай проверим»)

Question (Alex): can we tell which force (instruction / shown examples / truth) pulls hardest in a dialogue, and is
"the model's favourite force is pulling" a warning sign of a habit error? Alex doubts the removal method will tell much.

Data: the 300 direct dialogues of ../judge (5 facts × 20 seeds × 3 kinds; each force has its own number). Errors are
taken from the already saved greedy answers of the judge main run (frozen, scored by ../judge/score_j.py).
Variants per dialogue: FULL; NOEX (3 past Q/A turns removed, final user message unchanged); NOINS (latest number
instruction removed: the correction line in the instruction kind, the first turn + "Okay." otherwise).
Measure: log P that the answer starts with each candidate number (candidate tokens + next token not a digit), 3 models
(Qwen2.5-14B-Instruct, phi-4, OLMo-2-13B-Instruct), bf16, chat template. 900 prompts × 3 candidates per model.
Favourite force fixed from cycle3: qwen14 instruction, phi4 examples, olmo13 examples.

## Predictions (written before the main run; only a 20-dialogue smoke was looked at, to check the pipeline)
- P1 sanity: the candidate with the highest probability equals the greedy answer's source in >= 90% (all models).
- P2 fine lamp (primary): favourite force log-odds (FULL) as an error detector, pooled over kinds (in use we don't know
  the kind): AUC >= 0.75 AND the bootstrap CI of (AUC − AUC of low confidence = −(top1 − top2 log P)) above 0.
  Needed for qwen14 and phi4; olmo13 expected to fail (its errors were not on its favourite force).
  Expectation from smoke (n=20, unreliable): pooled P2 probably FAILS because the favourite pulls hard also where it is
  right; within a kind it may work. Within-kind AUCs are reported (secondary, no threshold).
- P3 "close one eye": for answers that followed instruction or examples, the force whose removal lowers its own
  log-odds more is the force the model followed, in >= 80% (all models). Report: removal-drop lamp error AUC.
- What this cannot answer: who SHOULD win. It only measures pull.

## Cost: < $0.5 (smoke cost ~$0.1).

## v2 after plan review (review/plan-review.md), before the main run
- R1: each candidate is scored as a bare number OR a fact sentence ("A hexagon has N sides."), combined; analysis uses
  dialogues where the 3 candidates hold >= 0.5 of the answer probability (coverage reported per kind).
- R2: P2 must also beat the coarse lamp (answer = favourite): PASS = AUC >= 0.75 and CI of (fine − low confidence)
  and of (fine − coarse) both above 0.
- R3: P2/P3 are NOT TESTABLE for a model with < 15 errors in the analysed dialogues (phi4 expected: 7 errors total).
- R4: old P3 was near-automatic. New P3 "close one eye adds information": AUC of the favourite force's removal drop
  minus AUC of the fine lamp (FULL), CI above 0 -> PASS. Report: share of dialogues whose top candidate changes after
  each removal, per kind. Note: in the examples kind NOEX keeps the praise line with nothing before it to praise; in the
  instruction kind the examples number is also the first-turn instruction.
- Smoke log-probs (n=20) were looked at before these predictions were written; a second smoke checked the v2 pipeline.
