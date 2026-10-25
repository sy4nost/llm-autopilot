# cicm-goal — "goal -> conflict -> slow re-check" on Berkeley CICM, short version (main run 2026-10-24, ~$1.5)

240 CICM dialogues (Berkeley's calibration split), 3 models. SCHEME = if the model's own goal sentence names a different
value than its quick answer -> think step by step and answer again; else keep the quick answer. Plan v2 after plan
review (value-parsing fix, random re-check baseline), sums frozen before main output. Data CC BY-NC, not in the repo.

## Verdict: P1 PASS (2 of 3 models), but always-thinking is much better
| right % | quick | placebo | goal note | SCHEME | always think |
|---|---|---|---|---|---|
| qwen14 | 38.8 | 46.7 | 46.7 | **50.4** | **66.2** |
| phi4 | 68.3 | 70.8 | 70.0 | 69.2 | 76.7 |
| olmo13 | 8.8 | 8.8 | 21.2 | **22.5** | **35.8** |
- P1 (SCHEME beats quick AND a random re-check at the same rate): qwen +11.7 [+7.3, +16.2] / vs random +6.9 [+4.0, +9.9]
  PASS; olmo +13.8 / +4.2 [+0.9, +7.2] PASS; phi +0.8 FAIL.
- P2 FAIL in all: SCHEME is 7–16 points below always-think.
- P3 harm: qwen 1.1%, phi 3.0% PASS; olmo 19% of 21 FAIL.
- P4 lamp FAIL: "goal != quick answer" AUC ~0.6; catches only 26–38% of stale errors.

## What it means
- The conflict flag does point at errors better than chance (beats a random re-check), so the idea transfers a little.
- But on CICM the goal sentence itself is often stale: Qwen's goal names the old value 46 of 100 times, OLMo's 72.
  The model's reading of "what the user wants" makes the same mistake as its quick answer, so there is no conflict to
  notice (like "salt in tea": no rival, no lamp). In goal2 the goal was easy to read (one explicit instruction);
  in CICM finding the latest value IS the hard part.
- Slow step-by-step re-checking on every item is the strongest fix here (+27 for Qwen and OLMo). Its cost is long
  answers on every question.
- phi4 is already good (68%) and gains nothing from the goal step.
