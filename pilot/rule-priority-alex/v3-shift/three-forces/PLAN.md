# three-forces: said vs shown vs memory (PLAN v2 after review/plan-review.md; was v1, 2026-10-24; Alex's idea)

Paradigm = Camassa & Shiller 2026 (arXiv 2605.20382): system "You are a helpful assistant." -> user: "I want you to answer X as your answer
to each one of my questions in this conversation." + first question -> N hardcoded assistant turns, each answering Y, with varied count
questions -> final question about a well-known fact (memory answer M). Model answers greedily (8 tokens). New vs Camassa: the final
question has a memory answer that competes with both the instruction and the shown pattern.

Facts (M, S, P): hexagon sides (6, 8, 5); spider legs (8, 10, 7); days in week (7, 9, 6); months in year (12, 14, 11); fingers on a hand (5, 7, 4);
pi in words and as a formula "pi = ?" (3.14, 4.25, 2.86); hexagon as a formula "sides(hexagon) = ?" (6, 8, 5).
Arms: A said S / shown P (three different answers); B said M / shown P (instruction agrees with memory); C said S / shown M (pattern agrees
with memory); D said S / shown P with a nonce object (no memory, e.g. "zorpagon").
N in {0,1,2,4,8,16,32,50}; 10 seeds of earlier questions per fact, shared across arms (paired). 8 facts x 4 arms x 8 N x 10 = 2560 prompts per model.
Models: Qwen2.5-14B-Instruct, phi-4, OLMo-2-13B-Instruct (bf16, greedy).
Scoring (score_tf.py): first number in the answer (digits or word), rounded to 2 decimals; class said / shown / memory / other.

Questions (descriptive, no pass/fail):
Q1 In A, how fast does "shown" overtake "said" as N grows; does memory appear as a third answer?
Q2 Is the instruction kept longer when it matches memory? said% B - A over N>=1, paired bootstrap.
Q3 Does the pattern win more when it matches memory? shown% C - A.
Q4 Memory's own pull: said% A - D (real vs nonce object) and memory share in A.
Q5 (report) formula vs words: pi_formula vs pi_words, hexagon_formula vs hexagon, arm A.
Caveats fixed in advance: answers counted by the first number only; "other" reported; bootstrap over (fact, seed, N) items ignores
clustering by fact (8 facts) -> also report per-fact table so one fact cannot drive a conclusion.
Cost: ~2560 short prompts x 3 models, batched, ~10-15 min, < $1.

## v2 changes (review)
- Earlier-question bank per fact: no same-fact/same-topic questions (hexagon removed for hexagon_formula), no question whose true answer is M, M+2 or M-1.
- S and P: even seeds S=M+2, P=M-1; odd seeds swapped (said is the smaller number), pi 4.25/2.86 swapped likewise.
- Nonce hand twin is "prongs on one drelth claw"; in arm D the real M is scored as memory_leak.
- New arm E: no instruction, shown P (pattern vs memory). Baseline Z: final question alone, real and nonce (16 prompts).
- 40 new tokens; numbers echoed from the final question and list markers are skipped; "M anywhere in the answer" reported.
- Q2 null is the memory share in A (in B said = memory), not 0. Bootstrap over (fact, seed) clusters; per-fact B-A, C-A, A-D printed.
- Token-budget batching (OLMo has no GQA); partial saves. 3136 prompts per model.
