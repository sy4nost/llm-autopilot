# three-forces result (2026-10-24; Alex's idea; Camassa-style paradigm)
Said (instruction "answer X to each question") vs shown (N hardcoded assistant answers Y) vs memory (true answer M of the final question).
3 models (Qwen2.5-14B, phi-4, OLMo-2-13B), 8 facts, 5 arms, N 0-50, 10 seeds; 3136 prompts per model. Full numbers: report_main.txt.
Plan reviewed before the run (review/plan-review.md, 6 required changes applied). Descriptive study, no pass/fail. ~$0.5.

1. Shown beats said fast. Arm A (three different answers): said share at N=2 is 19% / 4% / 9% (qwen / phi / olmo), ~0 by N=8.
2. Memory fights back only for pi. Memory share in A over N>=1: qwen 17%, phi 7%, olmo 31%, almost all from pi (qwen pi_formula 74%, pi_words 53%;
   olmo pi ~99-100%; integer facts 0-11%). Hexagon, spider legs, days, months, fingers: memory ~0-4% (olmo hexagon_formula 29%).
3. Instruction agreeing with memory (B) helps only where memory is strong: said B-A +22 / +22 / +30 pooled, but per fact +76...+100 on pi and
   -6...+10 on integer facts. So "said + memory" against "shown" wins only for pi; for hexagon etc. the shown pattern still wins 60-85%.
4. Pattern agreeing with memory (C) wins almost always (shown -> 95-100% by N=4).
5. Without any instruction (E) memory resists the pattern much more: qwen keeps memory 82% over N>=1 (59% at N=50), olmo 51%, phi 27%.
   With an instruction (A) memory drops to 17 / 7 / 31%. Untested interpretation: the "answer X to every question" instruction switches the model
   into a game where truth does not count, and inside that game the shown pattern beats the instruction.
6. Nonce object (D): models follow said somewhat longer at small N (N=2: 46 / 26 / 22% vs 19 / 4 / 9% in A) and at large N often say they do not
   know the word ("other" 24-74%).
7. Formula vs words (Q5, per fact n=70, no CI): memory share in A higher for the formula form - pi qwen 74 vs 53, phi 27 vs 19, olmo 99 vs 100;
   hexagon formula vs words qwen 6 vs 0, olmo 29 vs 0, phi 3 vs 1. Same direction as hex-wrap (formula form pulls memory harder). Tendency only.
Not analysed: the S/P swap (said larger vs smaller number).
