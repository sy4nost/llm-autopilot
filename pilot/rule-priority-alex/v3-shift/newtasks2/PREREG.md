# Run 2, written in advance: "lamp → louder new rule" on unfamiliar tasks with a compound amendment

*English translation. The original text, frozen before the run, is in the git history (commit 6d3f172). The checksums below are of the code and data files, not of this text.*

Written 2026-10-17, before Alex named the number for this run. After that this file and the scripts do not change; checksums at the end.

## Why a second run

In run 1 (newtasks/rezultaty-new.md) the model followed simple amendments even without help (98–100%), so there was nothing to test.
Here the amendment changes **part of a compound condition** "A and at the same time B". It was on such an amendment that 14B previously held on to the old answer.

## What does not change compared with run 1

- Lamp, threshold, boost (w0 in the b40 heads, threshold "lead below half", up16_all), model Qwen2.5-14B-Instruct.
- Arms A60, E_copy, E_nocopy, N_same, N60, R60, F0, 20 sets per type, 4 test cases each.
- Success criteria, scoring script and run code. modal_new2.py and score_new2.py differ from the run-1 files only in names.

## New task types (make_new2.py)

- **parcels.** A parcel is described in words: weight, address, insurance, urgency. Prioritized rules choose the transport. Amendment: rule 2 "insured AND urgent" becomes "insured AND regular".
  - Traps T1/T2: a light insured urgent parcel (abroad / domestic). Under the old rule 2 it flies, under the new one it does not.
  - Untouched S1/S2: a heavy parcel or an uninsured one.
- **bonus.** Colored chips worth 1–2 points plus a +3 bonus. Amendment: "has X AND has Y" becomes "has X AND no Y".
  - Traps: a pile with X and Y (the bonus disappears).
  - Untouched: without X.

## Success criteria (each type separately, as in run 1)

- **Gain:** traps A60 + E_nocopy, "by lamp" minus normal ≥ +10 points, and the lower bound of the 95% interval > 0.
- **No harm:** untouched cases in A60/E_nocopy/E_copy plus all N60, R60, N_same, change ≥ −5 points.
- **PASS** only if everything is met on both types.
- If without help the model solves traps A60 + E_nocopy at 85% or more, I report that the traps again did not work, and no conclusion about the lamp's usefulness on that type is drawn.

## Technical trial run

One run on 41 tasks with number 12345, only to check the mechanics (w1/up1 = normal, text boundaries, labels distinguishable). Its answers do not go into conclusions.

## Checksums (sha256)

- `make_new2.py` d61c64eab044d423b86d470ade67d3e9e019a5d622456d27d4c4a96aa3942c40
- `check_new2.py` d029b2ac3f0f2ac9a87afe9cc00a51a377ca2462a384e44a4a70e35eefbbcc40
- `modal_new2.py` aa0fab7aea13d4f209fbd02a894c90c367ce2fb293f5def04477e39a3e4d9f28
- `score_new2.py` 0f241409b53ce99169acf1a0b4e86893cfd275a5109c1fb94a3a1f4d91d88d63
- `../eras/settings2.json` 5a03d3037745bf96e45abbf877c5cc81a1bb2a2b9a1cbe77bc33e5fec14db6c0
