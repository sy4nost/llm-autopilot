# Parcels and bonus: automatism or misreading? (amb-check, 2026-10-24)

Same 280 items from conftrap-lamp (100 traps + 40 controls per family), 3 models, fast forced-choice answer. One thing changes:
- BASE: as before (rule, 60 solved examples, "IMPORTANT UPDATE", test case);
- REMIND: "Apply the rules as they are NOW, after the IMPORTANT UPDATE." before the test case;
- REPEAT: "Remember: <new rule text>" before the test case;
- NOHIST: same items without the 60 examples (rule + update kept).
Check: BASE picks match the earlier run on 274-280 of 280.

## Traps: right / old-rule answer (of 100)
| model | family | BASE | REMIND | REPEAT | NOHIST |
|---|---|---|---|---|---|
| qwen14 | parcels | 25 / 75 | 21 / 79 | 39 / 61 | 68 / 32 |
| qwen14 | bonus | 4 / 93 | 7 / 86 | 10 / 84 | 8 / 80 |
| phi4 | parcels | 57 / 42 | 51 / 49 | 11 / 89 | 100 / 0 |
| phi4 | bonus | 33 / 61 | 41 / 53 | 8 / 87 | 72 / 12 |
| olmo13 | parcels | 0 / 100 | 0 / 100 | 0 / 100 | 27 / 73 |
| olmo13 | bonus | 0 / 95 | 0 / 94 | 0 / 96 | 6 / 77 |

Controls (rule unchanged for that case): BASE 82-100; NOHIST drops phi4 bonus 97.5->75, olmo 100->72.5 and 82.5->35 (examples help understand the rule). CIs: report_main.txt.

## Conclusion
1. Reminders do NOT fix traps (-6...+8), unlike the hexagon case where one repeated phrase removed almost all errors. Not misreading.
2. Removing the history helps a lot: parcels qwen14 +43, phi4 +43; bonus phi4 +39. The past examples pull the error = true automatism from experience.
3. Bonus on qwen14 and olmo13: without history the old answer stays at 77-80 of 100, so the problem there is applying the rule amendment, not the examples.
4. REPEAT hurts phi4 sharply (parcels 57->11). Reason untested.
5. olmo13 barely handles amendments in any arm.
