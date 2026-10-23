# Audit of the "watchman" results, Qwen2.5-14B-Instruct

Auditor: independent session, 2026-10-18. I ran nothing on Modal or a GPU. I worked only from the files, using local `python3 -I`.
My own scoring script is `audit/recompute.py` and its full output is `audit/recompute_out.txt`. It does not import any `score_*`, `make_*` or `check_*` file, and it has its own text solver for all four task types.
Paths below are relative to `v3-shift/`.

## Bottom line

- **The numbers are real.** Every headline figure recomputes exactly from the raw jsonl.
- **No answer leakage.** I found no leak of the ground truth into `watch_b40` or `watch_all`, no bug, and no change made after the seed was known.
- **Verdict reported honestly.** The pre-registered verdict (FAIL, on bonus harm) is what the data say. The primary arm was not swapped after the results came in.
- **"Tuned to the last task": yes, in design, not in numbers.** The watchman has no tunable numbers. But it was conceived after run 2 failed on these same four generators, and it was tested on a new seed of those same generators. The task where it shines, parcels, is structurally the same task family on which the b40 heads were found and selected.
- **Missing baseline.** The parcels gain is about what you would get by simply asking the model again with no examples. No arm tests that.
- **Public hindsight gain is a class shift.** Every changed answer moves the same way. Always answering class 1 would score 54.9%, above both normal (45.4%) and watchman (50.2%).

| # | Question | Verdict |
|---|---|---|
| 1 | Do the headline numbers recompute? | **OK** |
| 2 | Leakage of truth into watchman or brake | **OK** |
| 3a | Pre-registration and hashes (synthetic run) | **OK** |
| 3b | b40 head selection vs test data | **CONCERN** (same task family, not the same items) |
| 3c | Watchman designed after seeing results on the same generators | **CONCERN** |
| 3d | Public run: recheck-format change after smoke, choice of benchmarks | **CONCERN** (small risk, no audit trail) |
| 4 | Generator bias, degenerate cases | **CONCERN** (several real issues, none flips the verdict) |
| 5 | Statistics | **OK**, with caveats |
| 6 | Other reasons for a reviewer to distrust | **CONCERN** (missing baseline; hindsight is a class shift) |

---

## 1. Recomputed numbers — OK

`sets_watch.json` regenerates byte for byte from `make_watch.py 185` (sha256 `556f4d88…a2ea` both times).
My own solver re-derived every example's stale/not-stale truth, every key and every old answer from the prompt text. It found 0 mismatches against `sets_watch.json` and against the `truth` field of the raw jsonl. Every example is correct under the rule in force at its own time. All 2240 raw rows line up with the set items (task, set, arm, probe, labels, key, old).

Synthetic run (watch_b40 vs normal; paired cluster bootstrap over the 20 sets, B=10000):

| | reported | recomputed |
|---|---|---|
| parcels traps A60 / E_nocopy | 28→80 / 22→80 | 27.5→80.0 / 22.5→80.0 |
| parcels gain | +55 [+40, +70] | +55.0 [+40.0, +70.0] |
| parcels harm | 0 | 0.0 [0, 0] |
| bonus traps | 5→22 / 5→15 | 5.0→22.5 / 5.0→15.0 |
| bonus gain | +14 [+6, +21] | +13.8 [+6.2, +22.5] |
| bonus harm | −7 [−11, −3] | −7.2 [−11.4, −3.3] |
| false flags parcels / bonus / robot / points | 2.5 / 20.8 / 0 / 3.5 % | 2.49 / 20.77 / 0.00 / 3.49 % |
| oracle bonus traps | 48 / 57 | 47.5 / 57.5 |
| E_copy traps (report only) | parcels 100→85, bonus 95→42 | same |

Public run:

| | reported | recomputed |
|---|---|---|
| hindsight | 45.4→50.2, +4.8 [+2.2, +7.6] | 45.40→50.16, +4.76 [+1.90, +7.62] |
| algebra | 98.1→98.5 | same, +0.40 [0.00, +0.90] |
| MMLU | 71.6→72.2, +0.6 [−0.2, +1.6] | same |

The small differences in CI bounds come from my different bootstrap seed and B. The integrity checks hold: no flag means a score identical to normal (917 synthetic prompts and 585 public prompts, difference 0.0).

## 2. Leakage — OK

- **Where truth is used.** In `watch/modal_watch.py`, `truth` and `stale` are used in only two places: building the oracle spans (line 92) and the output record. `flagged` (line 91) is built only from `flags`. Those come from `fresh(recheck_prompt(...)) != first(recorded answer)` (lines 85–89), which reads nothing but the prompt text and the model.
- **What the recheck sees.** `recheck_prompt` (lines 25–29) removes the `Example` lines and the `Below are…` lines. It keeps the rules and any note. These are all visible to the model in the normal arm anyway. No label, key, `old`, `changed` or generator state reaches it.
- **Empirical check.** In all 394 prompts where the watchman's flags exactly equal the truth, `watch_b40` scores are identical to `oracle_b40` (0 differ). With identical prompt, labels and flags, scores are always identical (0 of 2214 groups differ). The same (set, note, case, recorded answer) always gets the same flag (0 of 3324 differ). So no hidden state or truth path exists.
- **Masking** only sets attention to minimum on the flagged example lines, from "Answer for the TEST" onward. It adds no information.
- **Public run.** `gold` is never used for flags. One oddity: the example recheck chooses among the *test item's* answer classes plus the recorded example answers. That uses the test's options, not its gold, so it is not a leak. It does mean the public recheck differs from the synthetic one (see 3d).

## 3. Fitting

### 3a. Pre-registration — OK

- **Hashes.** All 8 sha256 sums in `watch/PREREG.md` match the current files, as do the 3 in `watch_public/PREREG.md`.
- **Timeline** (file mtimes, UTC 2026-10-17):

  | time | event |
  |---|---|
  | 22:46 | `modal_watch.py` last changed |
  | 22:55:13 | smoke run, seed 12345, a different data set |
  | 22:55:26 | `PREREG.md` |
  | 23:00 | `sets_watch.json` generated with seed 185 |
  | 23:20 | results |

  The `.pyc` source stamp equals the 22:46 file, so the run used the hashed code.
- **No arm switching.** `watch_all` would actually PASS every criterion: bonus gain +28.7 [+20, +37.5], harm −1.4 [−3.6, +0.6]. The report still keeps `watch_b40` as primary, as pre-registered, and states FAIL. That is good practice. (The report does not mention that `watch_all` would pass. Future write-ups must not promote it now; `watch_all` also costs −4.4 on public algebra.)
- **Minor gaps.** The PREREG says the smoke ran twice, but only one smoke log exists. A first smoke, if any, left no trace. Since it used seed 12345, the risk is negligible.

### 3b. How b40 was chosen — CONCERN (task-family overlap, no item overlap)

- **How it was chosen.** `heads/modal_heads.py` ranked all 1920 heads by attention to exact-copy example labels, on discovery sets 0–14 of `match/sets_match.json`. `blockex/rezultaty-blockex2.md` then picked b40 among four head sets (b20, b20+early, b40, late_all), based on sets 15–29 of the same task plus MMLU-570 and GSM8K-250. Sets 15–29 had already been used in `heads/` to compare top10/20/40. So b40 is fitted to that task, and its "held-out" numbers there are optimistic.
- **No item overlap.** The watch test items are generated from different code (`make_new.py`, `make_new2.py`), so they share no items with the selection data.
- **Same task family.** The selection task ("RULE-SWITCHING TEST": four priority rules, where the change rewrites one half of a compound Rule 2, "A and, at the same time, B") has the same structure as **parcels**: four priority rules, Rule 2 "insured and, at the same time, express", and the change flips one half. `make_new2.py` line 2 says it was built to reproduce "the kind of change where 14B kept the old answer before". So parcels is not an unfamiliar task for b40; it is a re-skin of the task b40 was selected on. Bonus is the structurally different type, and that is where the method fails.
- **Public MMLU.** b40 was also chosen partly for being harmless on MMLU, using the first 10 test questions per subject. The public MMLU sample (500 random questions, seed 185) should overlap that set by about 20 questions (570/14042 × 500). That is small, but it is not fully fresh no-harm evidence.

### 3c. Was the watchman tuned to the last task? — CONCERN (design, not numbers)

- **Timeline.** Run 1 (`newtasks`, seed 25) and run 2 (`newtasks2`, seed 32) used the same four generators. Run 2 failed only on bonus harm (−7), because the lamp over-fired. The watchman was then designed and tested on the same four generators with a new seed.
- **What is in its favour.**
  - It has no thresholds or tunable numbers.
  - b40 is frozen.
  - The recheck format was fixed before the smoke.
  - The seed is new.
  - It also failed on exactly the same point as run 2, which argues against hidden tuning.
- **But a new seed is not new tasks.**
  - Parcels' trap test cases are fixed tuples: T1 = (0,1,1,1) and T2 = (0,0,1,1), in `make_new2.py` line 56. All 80 parcels trap items in every seed are the same two logical cases. Only the transport names (4 label sets) and the order of the history change.
  - Robot T1 and T2 are the same test word in 20 of 20 sets.
  - So "new seed 185" re-tests the same task distribution the design was developed against. It does not test generalisation.
- **Conclusion.** This is a replication on instances, not an out-of-distribution test.

### 3d. Public run: recheck change after the smoke, and choice of tests — CONCERN (low risk, no audit trail)

- **The format change.** `watch_public/PREREG.md` line 4 discloses it. The public watchman picks the answer among candidates by summed log-prob, because in chat the model "starts explaining". The synthetic watchman instead takes the free first token. So the public watchman is a different recheck procedure than the one called "frozen" in `modal_pub_watch.py` line 1.
- **Timeline** (UTC):

  | time | event |
  |---|---|
  | 23:28:50 | directory and `__pycache__` created (first version of the script) |
  | 23:31:39 | script last modified |
  | 23:32:24 | the only surviving smoke log |
  | 23:32:50 | `PREREG.md` and `score_pub_watch.py` written |

  So the PREREG and the scoring script were written *after* the smoke(s), not before. The smoke output `pubwatch_…_smoke.jsonl` that the log says was written no longer exists, and the first smoke's log was overwritten. I cannot verify what was seen.
- **Risk assessment.** The smoke covers only 5 items per test, and the change is the obvious generic fix. So the risk that outcomes drove it is low. Those 15 smoke items are part of the final scored set.
- **Test selection.**
  - Earlier, in `public/`, 4 inverse-scaling tests were run with the copy heads off. Hindsight-neglect was the only one that moved in the favourable direction (+1.5); redefine-math −1.5, quote-repetition 0, NeQA −2.5.
  - The public watch run keeps hindsight and drops the other three. Items 0–199 of the 315 hindsight items had already been run in `public/`.
  - The report itself admits the tests were "picked by me". This is selection after seeing related results, mitigated by the pre-written expectation of "no gain".

## 4. Generator bias and degenerate cases — CONCERN

1. **The watchman re-solves the test question itself.** Every A60/E arm puts at least 2 copies of the test case into the history, by construction:
   - parcels: on average 8.0 copies (trap) and 5.5 (untouched);
   - robot: 19.6.

   Each copy's recheck is exactly "rules + note + the test case, no history", so the watchman always computes the model's history-free answer to the real question. The trap is defined by those copies, and the watchman deletes them. That is the most favourable possible setting for this method.
2. **Robot is degenerate.**
   - T1 == T2: the same test word in 20 of 20 sets, so its 80 trap items are effectively 40.
   - Its "E_nocopy" is not copy-free. `robot_p2` (`make_new.py` lines 66–70) always puts the changed word twice into PERIOD 2, and the stand-in is drawn from words that include S probes. All 40 robot trap E_nocopy prompts, and 37 of 40 S prompts, contain the test case in PERIOD 2.
   - Points has the same problem in 38 E_nocopy prompts.
   - `check_new2.py` has a check for this; `check_new.py` does not.
   - Robot and points carry no gain criterion, but their E_nocopy "no-harm" rows are made easier.
3. **Parcels has only two distinct trap cases** (see 3c). Untouched S1 has 7 distinct texts and S2 has 4. The parcels n=80 and n=360 overstate how many distinct problems were tested.
4. **No-harm controls make parcels harm zero structurally.**
   - In N60, R60 and N_same, every example is correct under the stated rules. With parcels false flags at 2.5% (0% in these arms), the lamp never fires, and the code then returns exactly the normal answer.
   - In A60/E, the untouched cases are heavy or uninsured parcels, which the model answers at 100% regardless.
   - So "harm 0" means "the recheck is accurate on parcels". There is no control where correct examples look unusual, or where the rules text is hard to apply without examples (bonus shows what happens then: −7).
5. **E_copy traps are excluded from "harm"** by the (pre-registered) definition. Yet the watchman damages them: parcels 100→85, bonus 95→42. The report states this, but a skeptic would count it as harm where experience was right.
6. **Single-token answers.** Answers are single-token for all four types (sums ≤ 9; labels with distinct first tokens, asserted), so the first-token comparison is valid. I found no bug there.

## 5. Statistics — OK, with caveats

- **Resampling unit.** `score_watch.py` resamples (task, set) clusters with replacement and recomputes the pooled difference on the same items. It is paired and uses the right unit: arms and probes of one set share the 60-example history. My independent cluster bootstrap gives the same intervals, and an item-level bootstrap gives similar or narrower ones.
- **Percentile intervals with 20 clusters** are somewhat anti-conservative. The parcels gain (+55, lower bound +40) is far from any threshold. The bonus harm upper bound (−3.3) is also clearly below 0. The verdict is robust.
- **Diversity.** For parcels, the 20 clusters differ only in label names and order, so the CI describes variation over renamings, not over tasks.
- **Multiple comparisons.**
  - The synthetic verdict requires all criteria to pass, so it is not inflated.
  - The narrative highlights the 5 passing conditions and the parcels number. That is acceptable because FAIL is stated first.
  - Public: 2 tests are judged for "gain". Hindsight's sign test (18 fixed, 3 broken, p = 0.0015) would survive Bonferroni.
- **Public bootstrap** is over items and paired. That is correct, since the items are independent.

## 6. Other reasons for a reviewer to distrust — CONCERN

1. **Missing trivial baseline: "answer without history".**
   - The PREREG has no arm that simply asks the model the test question with the examples removed: `recheck_prompt` applied to the test case.
   - The data bound it. In parcels trap prompts, the test-case copies are flagged in 82% of prompts, which means the history-free answer differs from the old answer in 82%. Watchman accuracy is 80%. In N60/R60/N_same parcels the history-free answer agrees with the examples 100% of the time.
   - So on parcels, the watchman appears no better than dropping all examples. It may simply be a costlier version of that.
   - On bonus, the history-free answer contradicts the correct old answer in 20–40% of no-change prompts, so dropping history would hurt there. The watchman's partial masking hurts less.
   - Either way, the claimed value of the watchman over "just re-ask without examples" is not shown. The report's own conclusion, "the watchman is as reliable as the model's history-free understanding", is consistent with this.
2. **Hindsight "gain" is a one-directional class shift.**
   - All 21 changed answers move from class 0 to class 1: 18 become right, 3 become wrong.
   - The gold split is 142 for class 0 and 173 for class 1, so always answering class 1 scores 54.9%, above both normal (45.4%) and the watchman (50.2%).
   - The improvement is consistent with masking examples that pushed the model toward class 0. It does not show the watchman found "stale" examples: in hindsight-neglect the examples are by design not stale, which the PREREG itself expected.
   - The effect is similar on old and new items (0–199: +5.5; 200–314: +3.5).
3. **Comparison with the lamp (run 2) uses different seeds.** The report says so. Even taken at face value, the watchman is not better than the lamp: parcels about equal, bonus +14 vs +28, harm equal.
4. **One model, one architecture.** All numbers are from a single model. The `multi/` folder (other models) was created after these reports.

## What would need to be re-done

1. **Add a "no-history" baseline arm** (rules + notes + test case, no examples) to the same 2240 prompts. Compare `watch_b40` with it on gain and harm, pre-registered.
2. **Test on genuinely new task families:**
   - not isomorphic to the head-selection task (unlike parcels);
   - with varied trap test cases, not two fixed tuples;
   - including traps where the test case does *not* appear in the history, so the watchman cannot simply re-solve the test question.
3. **Fix the robot and points generators.** Make E_nocopy truly copy-free (add the check from `check_new2.py` to `check_new.py`), and make robot T1 ≠ T2. Otherwise report robot as n=40 trap items.
4. **Public run.**
   - Keep smoke outputs, and write the PREREG and the scoring script before any smoke.
   - Pre-specify the benchmarks before looking at related results; include redefine-math, quote-repetition and NeQA again.
   - For hindsight, report the class-prior baseline and per-class accuracy.
   - Use an MMLU sample disjoint from the 570 questions used to choose b40.
5. **Optional: re-select or check b40 on a task family that shares no structure with parcels,** then rerun parcels to see whether the result depends on that overlap.
6. **Keep `watch_b40` as primary.** Any later claim built on `watch_all` (which would pass here) must be a new pre-registered test on fresh tasks.
