# Run 2: unfamiliar tasks with a compound amendment (2026-10-17, Alex's number = 32)

Plan: PREREG.md (checksums verified). Model Qwen2.5-14B, 1120 tasks, sanity checks w1/up1 = 0.0. Full output: score_new2_out.txt.

**Verdict by the pre-registered rules: FAIL**, but only on one criterion out of four.

| | parcels | bonus (bonus points) |
|---|---|---|
| Gain on traps A60+E_nocopy | **+57 [+46, +69] PASS** | **+28 [+18, +39] PASS** |
| Harm where experience is right | **0 PASS** | **−7 [−10, −5] FAIL** (threshold −5) |

This time the traps worked: without help the model held on to the old answer in 80–90 cases out of 100.

Correct answers on traps, normal → with the lamp:
- **parcels:** A60 20→78, E_nocopy 12→70. Ceiling without history (F0): 92.
- **bonus:** A60 5→30, E_nocopy 8→38. Ceiling (F0): 88.

Where the harm is (bonus): the lamp fires too often, also where the rule did not change (35–98%). Losses:
- A60 untouched cases 95→72;
- R60 (note "the rule does not change") 100→78 and 95→80.

On parcels the lamp is selective: 80–88% on traps, 0–15% where there is nothing to change.

Report-only, not part of the verdict:
- **bonus, "just silence the old" (w0):** more correct answers on traps (62/60), but also more harm.
- **×4 instead of ×16:** the gain is clearly smaller.
- **E_copy (a fresh identical case):** already 95–100 without help.

Conclusions:
1. A recipe tuned on the old tasks and frozen in advance transfers to new tasks with a compound amendment. Habit
   errors drop 2–6 times.
2. The weak spot is the lamp. On one task type it separates a trap from a normal case poorly, and the harm slightly
   exceeds the pre-set threshold.
