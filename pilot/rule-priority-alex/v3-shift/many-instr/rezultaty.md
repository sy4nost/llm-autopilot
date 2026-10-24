# many-instr — results (main run 2026-10-24, < $0.5)

K successive instructions ("Correction: from now on … instead"), no examples, then a question. 3 instruct models, 1605 prompts
each (4 formats × K in {1,2,5,10} × 5 facts × 20 seeds + baseline). Plan v2 and sums in PLAN.md / SHA256.txt (frozen before output).

## Verdict by pre-registered rules
- P1 (last-instruction share drops from K=2 to K=10, CI below 0 in >= 9 of 12 cells): NOT supported — 4 of 12 cells
  (qwen14 F4 −6; olmo13 F1 −41, F3 −65, F4 −18). Qwen and phi-4 follow the last instruction 94–100% everywhere.
- P2 (first instruction sticks): NOT supported. The first instruction was chosen in 0 of 4800 K>=2 answers across all models.
  Testable cells (olmo F1, F3) give the minimum possible value (−12.5): slips go to middle positions, never to the first.

## Last-instruction share at K=10
| model | F1 one message | F2 separate turns | F3 "start with word" | F4 identical wording |
|---|---|---|---|---|
| qwen14 | 97 | 100 | 99 | 94 |
| phi4 | 100 | 97 | 100 | 100 |
| olmo13 | 59 | 100 | 35 | 82 |

- OLMo slips in one-message formats at K=5..10. Where it slips (K=10): F1 positions 2–9 fairly even; F3 mostly positions 3–4
  (22 and 21 of 100); F4 only the last few (7–9). With one instruction per turn (F2) it is perfect.
- All slips in Qwen/phi are to position 9 (the next-to-last).
- F3: true answer present 100% in all cells — word rule and truth do not compete.
- No memory answers anywhere: as long as an instruction is present, models follow some instruction.

## Plain reading
Successive instructions confuse these models far less than shown examples did. The newest instruction wins; nothing like
"stick to the first one" appears. Only OLMo gets lost when many corrections are packed into one message, and then it picks a
random middle one, not the first. Splitting instructions into separate turns removes the problem entirely.
