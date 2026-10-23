HOW TO RUN

1. Start a NEW CHAT for each trial. Use the same model/version/settings.
2. Randomly choose one prompt file (control, experimental, or mixed) and paste its COMPLETE contents as a single user message.
3. Record the model's first response without correcting it or giving feedback.
4. Repeat each condition across many independent new chats, preferably at least 20-30 runs per condition as an initial pilot; ideally use varied seeds and multiple task templates for generalization.
5. Do not show the model ANSWER_KEY_keep_private.txt. Compare outputs using the key.
6. Randomize arm order; avoid model switching, persistent memory, or carryover chats.

LIMITATION: the control uses unrelated solved examples and has slightly different character length. This is an initial pilot, not a fully length-matched controlled study. A rigorous follow-up should match token counts and add many independently generated problem sets plus a fresh-probes design. All training examples have displayed answers, so this tests in-context priming rather than sequential feedback-based habit formation.

Arm: CONTROL — 20 unrelated solved demonstrations
Arm: EXPERIMENTAL — 20 PASS-only demonstrations
Arm: MIXED — 10 PASS + 5 HOLD + 5 REVIEW demonstrations
