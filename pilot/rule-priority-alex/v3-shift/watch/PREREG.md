# Watchman, written in advance (2026-10-17, before Alex's number)

*English translation. The original text, frozen before the run, is in the git history (commit 6d3f172). The checksums below are of the code and data files, not of this text.*

## What the watchman is

It does not solve the task. For each old example the watchman asks the model: "what is the answer here under **today's** conditions?".
- Today's conditions are the task text without the history of examples: the rules and all notes.
- If the fresh answer does not match the one recorded in the example, the example is "from another era", and the lamp fires.

**Brake:** only these stale examples are silenced in the 40 copying heads (b40), from the question onward.
- No thresholds, no boosting of the note.
- If nothing is flagged, the answer is exactly the normal one (checked in code).

This is a direct implementation of Alex's formula "vote strength = sameness of conditions": an example made under different conditions gets a vote of 0.

## Tasks

make_watch.py with one new number from Alex:
- 4 task types from runs 1 and 2 (robot, points, parcels, bonus), 20 sets per type, arms A60, E_copy, E_nocopy, N_same, N60, R60, F0;
- 2240 tasks in total;
- the truth about which examples are actually stale comes from the independent checks check_new*.py.

## Model and settings

Qwen2.5-14B-Instruct.
- **Main:** watch_b40.
- **Report only:** watch_all (silence in all heads) and oracle_b40 (a perfect watchman).

## Success criteria

The same thresholds as before.
- **parcels and bonus:** gain on traps A60 + E_nocopy ≥ +10 points, lower bound of the 95% interval > 0.
- **All 4 types:** harm where experience is right (untouched cases in A60/E_nocopy/E_copy plus all N60, R60, N_same) no worse than −5 points.
- **PASS** only if all conditions are met.

## Trial run

Twice 28 tasks with number 12345, mechanics only: boundaries of example lines, identity with the normal answer when nothing is flagged. Answers were not looked at.

## Checksums (sha256)

- `make_watch.py` 30563a460f6f3c5d3ef7a7f565c8625db6a1669df2497bfa8e8ad563c564ea15
- `modal_watch.py` 923b82a02d2fd44448e1f336bd6c520fa121c4f74a6bed56a943ffbf5db84063
- `score_watch.py` d7fb830b9450968129c2623298889d89e717e99ad2d55d5866d325b6b85996a9
- `../newtasks/make_new.py` a5fc13333c9dedca224427430a851769c0465ea12feb6fba5aa1f0f7a4c0f409
- `../newtasks2/make_new2.py` d61c64eab044d423b86d470ade67d3e9e019a5d622456d27d4c4a96aa3942c40
- `../newtasks/check_new.py` 3719f3a7867f79fe99d8cfe1f818f34884e045329d4beb9c78909e04e9f6bfde
- `../newtasks2/check_new2.py` d029b2ac3f0f2ac9a87afe9cc00a51a377ca2462a384e44a4a70e35eefbbcc40
- `../eras/settings2.json` 5a03d3037745bf96e45abbf877c5cc81a1bb2a2b9a1cbe77bc33e5fec14db6c0
