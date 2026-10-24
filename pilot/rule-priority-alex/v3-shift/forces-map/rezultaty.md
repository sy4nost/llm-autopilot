# forces-map — does a "map of forces" step act as a lamp? (main run 2026-10-24, ~$1)

Alex's question: if the model answers the "which force says what" questions itself, will it repeat the same error?
Set: 240 self-built dialogues (NUM "answer each question with the number X", PREF "my coffee drink is latte"), half calm,
half traps (the assistant's earlier replies contradict the user's latest instruction). Qwen2.5-14B, phi-4, OLMo-2-13B.
Plan, predictions and all pre-main fixes: PLAN.md (v1.1–v1.3); review: review/plan-review.md; full numbers: report_main.txt.

## Results (measured)
1. **The map does not repeat the error.** Map line 1 ("what the user's instructions call for, latest one") is right
   almost always, including where the quick answer was wrong:

   | model | DIRECT right | always THINK | map line 1 used as the answer | line 1 right where DIRECT wrong |
   |---|---|---|---|---|
   | Qwen | 95.0% | 88.8% | 97.1% | 10 of 12 |
   | phi | 62.1% | 86.2% | **100%** | 91 of 91 |
   | OLMo | 77.1% | 57.1% | **98.3%** | 54 of 55 |

2. **Pre-registered lamp (line 1 != line 2 "assistant's recent replies"): OVERALL FAIL (P1+P2 in 1 of 2 testable
   models; Qwen P2 n/a, too few DIRECT errors).** It detects the trap in the dialogue, not the model's error: Qwen fires
   on 88% of traps but gets almost all of them right (false alarms on 42% of DIRECT-correct). OLMo cannot fill line 2
   ("Okay." plus a list), P1 FAIL (+10). phi: P1 PASS (+100), P2 PASS (+12.1 over random re-check). P3 PASS where
   testable (phi 100%, OLMo 98%).
3. **Post-hoc lamp "map line 1 != DIRECT answer" (not pre-registered) is close to the oracle:** catches 83 / 100 / 98% of
   DIRECT errors with 2 / 0 / 2% false alarms (Qwen / phi / OLMo); fires on calm dialogues 0 / 0 / 1%.
4. THINK is worse than DIRECT for Qwen and OLMo (NUM: pulled to the true fact), so "re-check by thinking" can hurt;
   taking the answer from map line 1 is better than THINK in all three models.

## Caveats
- Self-built, easy set: the instruction is stated explicitly. On CICM the goal sentence itself was stale.
- The map prompt contains the hint "if the user changed an instruction, use the latest one" and a third-person view of
  the transcript; no control arm without the three-line map (review S2), so "map" vs "outside view + hint" is not separated.
- Result 3 is post-hoc; needs confirmation on new dialogues.
