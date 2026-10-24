# cycle3 — instruction vs examples vs truth on the same questions (main run 2026-10-24, < $0.5)

3 models × 1320 prompts (6 facts × 20 seeds × 11 arms). Plan v2 + sums frozen before main output (PLAN.md, SHA256.txt).

## Verdict
- CYCLE: NOT supported, 0 of 3 models. Each model has a plain ranking, not a circle:
  - qwen14: instruction > truth > examples. (IT X 94; ET N4 truth 96; IE N4 X 78 vs Y 22.)
  - phi4: examples > instruction > truth. (IT X 100; IE N4 Y 100; ET N4 Y 97.)
  - olmo13: examples > instruction > truth. (IT X 100; IE N4 Y 81 vs X 9; ET N4 Y 84.)
- The "circle" I had pieced together from earlier separate runs was an artifact of mixing formats (earlier the instruction sat in
  the same message as the first question; here every instruction is its own turn with "Okay."). Withdrawn.
- C4 (generic "answer with a single number" removes truth vs a neutral "Let's start." preface): formally 2 of 3, but tiny —
  qwen −6.5 [−12.5,−1.0], phi −5.0 [−9.5,−1.0]; olmo opposite +19.5 [+14.5,+25]. What removes truth is a SPECIFIC number
  instruction (IT/IE truth 0–18%), not the "number game" as such.

## Integer facts, shares of 100 (truth / instruction X / example Y)
| arm | qwen14 | phi4 | olmo13 |
|---|---|---|---|
| question alone | 100 truth | 100 truth | 100 truth |
| instruction X | 6 / 94 / – | 0 / 100 / – | 0 / 100 / – |
| 2 examples Y, no instruction | 100 / – / 0 | 35 / – / 64 | 48 / – / 51 |
| 4 examples Y, no instruction | 96 / – / 0 | 1 / – / 97 | 16 / – / 84 |
| instruction X + 2 examples Y | 0 / 98 / 2 | 0 / 0 / 100 | 18 / 45 / 35 |
| instruction X + 4 examples Y | 0 / 78 / 22 | 0 / 0 / 100 | 10 / 9 / 81 |
| correction X → X2 | X2 98 | X2 100 | X2 100 |

## Pi (strong fact, 20 per arm, reported only)
qwen: examples never beat pi (truth 100 with 4 examples); instruction does (X 100). olmo: pi resists examples (truth 75–85) and,
with instruction + 4 examples, truth 80. phi: pi falls to examples (Y 95 with 4).

## Plain reading
Qwen is the "obedient and knowledgeable" type: obeys words, keeps facts, ignores planted answers. Phi-4 and OLMo are "copiers":
planted answers beat both words and facts. Everyone obeys a direct "answer X" over the truth.
