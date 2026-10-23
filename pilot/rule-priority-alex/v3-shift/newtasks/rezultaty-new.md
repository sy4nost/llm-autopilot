# Result of the honest run on unfamiliar tasks (2026-10-17, Alex's number = 25)

Plan: PREREG.md (checksums verified before the run). Model Qwen2.5-14B, 1120 tasks, sanity checks w1/up1 = 0.0. Output: score_new_out.txt.

**Verdict by the pre-registered rules: FAIL.** No gain (robot +0, points −1), no harm either (robot 0, points 0).

**Reason: on these tasks the model had no automatism at all.** Without any help the traps were solved correctly:
- robot: 100 out of 100, although the history had 7–25 exact copies of the old answer;
- points: 98–100 out of 100.

There was nothing to cure, so there could be no gain.

- **The lamp.** On robot it almost never fires (0–2%). On points it fires in 32–38% of N60/R60 cases but barely spoils answers: R60 traps 100→98, the rest unchanged.
- **"Always ×16".** Without the lamp it hurts slightly on points (A60 98→90, R60 100→88). The lamp removes this harm.

**What it means (inferred from the data, not tested separately).** 14B follows a simple explicit note ("word X now means Y", "a chip is worth 3") even when the history has dozens of old answers. In earlier tasks, the habit from examples appeared where the new rule was an amendment to a compound condition ("if A and at the same time B"). This agrees with the earlier conclusion: the remaining error is a weakness in applying amendments. Simple copying does not beat a clear note.
