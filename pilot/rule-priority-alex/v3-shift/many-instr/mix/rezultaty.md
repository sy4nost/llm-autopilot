# many-instr/mix — instructions vs examples (main run 2026-10-24, < $0.5)

5 successive instructions in separate turns (the 5th valid), then N past assistant answers showing the FIRST instruction's
number (E1), the PENULTIMATE (E4) or a NEW number (EX), then the test question. 3 models × 1600 prompts. Plan v2 + sums frozen
before output (one later amendment: empty-cell guard in scoring only).

## Verdict by pre-registered rules
- P1 (last-instruction share falls from N=0 to N=3): SUPPORTED, 9 of 9 cells (qwen −32/−53/−45, phi −100 all, olmo −97/−62/−89).
- P2 (example repeating an old instruction is copied more than a new number, both E1 and E4, >= 2 of 3 models): NOT supported,
  1 of 3 models (phi4 +20/+14). qwen14: E1 −2.7 [−7.7,+2.3], E4 +11.3 [+6,+17]. olmo13: E1 +28 [+22,+34], E4 −15.7 [−23,−8].

## Share following the LAST instruction (of 100), by number of shown examples
| model / arm | N=0 | 1 | 2 | 3 | 5 | 8 |
|---|---|---|---|---|---|---|
| qwen14 E1 first | 100 | 99 | 67 | 68 | 78 | 77 |
| qwen14 E4 penult. | 100 | 90 | 55 | 47 | 58 | 71 |
| qwen14 EX new | 100 | 100 | 71 | 55 | 65 | 54 |
| phi4 E1 | 100 | 1 | 0 | 0 | 0 | 0 |
| phi4 E4 | 100 | 21 | 0 | 0 | 0 | 0 |
| phi4 EX | 100 | 20 | 0 | 0 | 0 | 0 |
| olmo13 E1 | 100 | 33 | 2 | 3 | 1 | 0 |
| olmo13 E4 | 100 | 92 | 52 | 38 | 9 | 1 |
| olmo13 EX | 100 | 64 | 15 | 11 | 2 | 0 |
The rest are almost all copies of the shown number (multi 0%, memory <= 2%).

## Headline: first N where the example beats the last instruction (CI above 0)
- phi4: N=1 in all arms (one shown answer is enough: 97/100 copy in E1).
- olmo13: N=1 (E1), 2 (EX), 3 (E4).
- qwen14: never; the example peaks at about half (E4 N=3: 53 vs 47) and Qwen gets MORE resistant at N=5..8.

## Notes (post hoc, untested)
- Qwen holds the last instruction much better here than with a single instruction in three-forces (said 20/100 at N=2 there,
  67/100 here). Possible reason: a history of corrections acknowledged with "Okay." signals that instructions are live.
- OLMo: examples that agree with the FIRST instruction pull hardest (54/100 after one example), penultimate the least.
