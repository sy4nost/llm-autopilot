# many-instr/mix — instructions vs examples: from which example does the last instruction break? (Alex, 2026-10-24)

Scale between two extremes seen earlier: only successive instructions (models follow the last one ~100%) and instruction +
shown answers (the shown answer wins). Here: 5 successive instructions in separate turns ("Correction: from now on … instead",
assistant "Okay."), the 5th is valid; then N past assistant answers to other questions; then the test question.

## Design (make_mix.py -> sets_mix.json, 1600 items per model)
- N in {0,1,2,3,5,8}; 5 facts; 20 seeds. Values: 5 instruction numbers + 1 new number, from 2..20, excluding M-1..M+1, distinct.
- Arms (N>=1): E1 examples answer with the FIRST instruction's number; E4 with the PENULTIMATE (4th); EX with a NEW number no
  instruction named (Alex asked for this variant). N=0 is shared (instructions only).
- Bank questions: three-forces bank, excluding questions whose true answer equals any drawn number or is near M, and same-topic.
- Same question list per (fact, seed) across arms and N (prefixes), so arms differ only in the shown number.
- Models: Qwen2.5-14B-Instruct, phi-4, OLMo-2-13B-Instruct; chat template; greedy; 40 tokens; repetition_penalty 1.0.

## Scoring (score_mix.py, v2 after review/plan-review.md): committed value. Candidates = 5 instruction numbers + example number.
One distinct candidate in the answer -> it; several -> number after "answer is"/"Answer:"/":" if a candidate, else the candidate
the answer starts with, else multi. No candidate -> memory (true answer present) or other. Label last / ex / old / memory / multi / other.

## Predictions (written before any main-run output; v2 rules after review)
- P1: last-instruction share at N=3 is lower than at N=0 (paired, CI below 0) in each arm; holds if in >= 7 of 9 model×arm cells.
- P2 (one-sided, 2.5% level via the two-sided 95% bootstrap CI): examples that repeat an old instruction (E1, E4) are copied
  more than a new number (EX), N=1..3 pooled, CI above 0. Holds per model if both E1 and E4 show it; overall if in >= 2 of 3
  models. Untestable (not "fails") if ex% is >= 95 or <= 5 in all arms at N=1..3 for that model. N=0 rate of answering that
  old-instruction number is reported next to it.
- Headline (reported): ex% − last% per N with paired CI; first N where ex% > last% (CI above 0) per arm. Also multi/other/memory shares, P1 at N=1, 2.
- Note: same-topic bank exclusion only for week/year (hexagon may see other polygons); batched greedy may differ slightly from unbatched.
- Prior: from three-forces (examples beat a single instruction by N=2), expect a fast drop; P2 genuinely uncertain.

## Cost: 3 × 1600 prompts on L40S, < $0.5.
