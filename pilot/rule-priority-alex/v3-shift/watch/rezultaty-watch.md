# Watchman, result (2026-10-17, Alex's number = 185)

Plan: PREREG.md (checksums verified). Qwen2.5-14B, 2240 tasks. Check "nothing flagged → answer as usual": difference 0.0. Output: score_watch_out.txt.

**Verdict by the pre-registered rules: FAIL**, again only on harm on bonus points (−7 with a threshold of −5). The other 5 conditions are met.

| type | traps A60 / E_nocopy, normal → with watchman | harm where experience is right | watchman wrongly flags correct examples |
|---|---|---|---|
| parcels | 28→80 / 22→80 (**+55 [+40,+70]**) | **0** | 2.5% |
| bonus | 5→22 / 5→15 (**+14 [+6,+21]**) | **−7 [−11,−3]** | 20.8% |
| robot | 100 / 100 (no habit) | 0 | 0% |
| points | 98→100 / 95→98 | 0 | 3.5% |

Main points:
- **The watchman and "own" tasks.** Where the model itself understands the current rules well, the watchman is almost
  perfect: on parcels and robot it stays silent on all tasks without a change (0%). Where there was a change, it
  silences the right examples. It needs no change note, only the text of the rules.
- **The weak spot is bonus points.** Without history the model computes the bonus imprecisely (F0 ceiling 92%), so
  the watchman also flags some correct examples as stale. The lamp fires almost everywhere, harm −7.
- **A perfect watchman on bonus (oracle).** Traps 48 and 57 out of 100, no harm. So with exact re-checking bonus
  would also be harmless, but the gain is smaller than on parcels.
- **E_copy traps (report only).** parcels 100→85, bonus 95→42. Silencing old examples sometimes hurts even where a
  fresh correct example lies next to it. This is a minus; the cause was not tested.

Comparison with the lamp from run 2 (different numbers, the comparison is approximate):
- **parcels:** both gain about +55, no harm.
- **bonus:** lamp +28/−7, watchman +14/−7.

**Conclusion.** The watchman is exactly as reliable as the model's own understanding of the current rules without
history. Its errors are the model's errors when re-checking.

Note: a later independent audit (`../audit/audit-report.md`) found that simply dropping all history (F0) beats the
watchman on these tasks, so the value of the watchman is not proven here.
