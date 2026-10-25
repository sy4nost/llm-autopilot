# Plan + code review: watchman-public (PLAN v1, make_wp.py, modal_wp.py, score_wp.py)

Independent auditor, 2026-10-23. Written before the main run. I did not run Modal and did not edit any file under review. The only file I wrote in this folder is this one.

What I ran (all on copies in my scratchpad):
- **make_wp.py.** I made a copy that reads data from the real folder and writes to scratch. I built seeds 777001, 4242 and 99, full size. Each gives 380 items: cpkc 100, modar 100+40, base9 100+40.
- **Brute-force checks on those 3 builds.** I checked every base9 key and habit, the position of the right option, reuse against the pilot and smoke (by passage and by operand row), and accents and name containment in cpkc.
- **score_wp.py --smoke** on copies of out_smoke_*.jsonl, which appeared at 23:21-23:22. I also dumped every THINK answer line and every sample the parser cannot read.
- **score_wp.py on synthetic out files** for 3 models. It runs end to end: verdict, LOFO, R2/R3/R6. No crash. sklearn 1.9.1 imports.
- **parse_line on hand-made lines** (results below).

Line numbers refer to the files as of 23:18. make_wp.py lines count from its own line 1.

The carry-overs from the conftrap-lamp review are in place:
- off-list THINK = -1 (S1);
- number words (S3a) and the report-only SAMPLE' (S3b);
- repetition_penalty 1.0 for greedy and sampling. The Qwen generation_config has 1.05, and the smoke checks confirm that both calls override it;
- LOFO gate;
- wall-clock stamp;
- sklearn imported first.

Not carried over:
- **B1 (meaningful clusters).** Every cluster is again one item; see S4.
- **N1.** The main-run batch check still compares raw last lines (modal_wp.py:107), not parsed answers.

## Hand check of base9 items (all correct)
| prompt (base 9) | decimal | base-9 sum = right | habit (digits added in base 10) | other | file |
|---|---|---|---|---|---|
| 65+17 | 59+16=75=8·9+3 | 83 | 82 | 84 | smoke id 10 |
| 51+71 | 46+64=110=81+27+2 | 132 | 122 | 131 | smoke id 11 |
| 48+60 | 44+54=98=81+9+8 | 118 | 108 | 117 | smoke id 12 |
| 48+34 | 44+31=75 | 83 | 82 | 84 | smoke id 13 |
| 43+47 | 39+43=82=81+1 | 101 | 90 | 102 | seed 777001 |
| 81+28 | 73+26=99=81+18 | 120 | 109 | 118 | seed 777001 |
| 61+62 | 55+56=111=81+27+3 | 133 | 123 | 132 | seed 777001 |

Brute force over 300 traps in 3 builds: 0 key or habit mismatches. Every trap differs between base 9 and base 10 (make_wp.py:78 rejects the no-carry cases).
- Operands use digits 0-8 only.
- Controls are base-10, with options s-1, s, s+1.
- No option can collide with the decimal value of the sum, so an answer like "83 (base 9) = 75 in base 10" still parses as 83.
- About 30% of habit options contain the digit 9, which is not a valid base-9 digit (e.g. 109, 90). See N3.

---

## BLOCKERS (fix before PREREG / hashes)

### B1. The numeric parser turns correctly formatted answers into "off-list", which counts as FLAG and as wrong. It also drops minus signs.
`score_wp.py:15` (norm), `score_wp.py:35-40,70-73`.
- **Thousands separators.** `"Answer: 1,381"` gives tokens `1` and `381`, so no option matches and parse returns **-1** (off-list: a flag, and THINK wrong).
  - Every modar trap with a 4-digit answer is exposed: addition up to 1999, multiplication up to about 9800, which is roughly 2/3 of modar.
  - Sampled answers have the same problem: `" 1,381."` is unparsed, so it counts as a disagreement.
  - It never happened in the 36 smoke lines, but 512-token step-by-step arithmetic from Phi-4 and Qwen often writes "1,381".
  - On fast-right items it adds false flags. That raises the flag rate, which gives every rival a larger k. So the bias runs against FLAG, and it also corrupts THINK accuracy.
- **Minus sign dropped.** `[a-z0-9]+` drops "-". About 22-26 modar items per build have negative answers (subtraction).
  - `"Answer: 642"` is read as `-642` (right), and `"643"` is read as the habit `-643`.
  - When the target is 0 (a-b = -1), the options `-1, 0, 1` normalise to `1, 0, 1`. Then `"1"` maps to `-1`, because `cn.index` returns the first match.
- **Fix** (for the modar and base9 branches and the off-list check):
  ```python
  num = lambda s: re.findall(r"(?<![\w.])-?\d+", re.sub(r"(?<=\d),(?=\d{3}\b)", "", s.replace("−", "-")))
  ```
  Then compare against the candidates as written, with their signs. Check that `"805 - 575 + 1 = 231"` still works: `-575` is never a candidate, so it is harmless.
  Put this into the frozen score_wp.py and say it in PLAN. It changes the off-list rule, so it must go in before the freeze.

No other blockers. Nothing makes the run unsafe or the cost unbounded. The estimate is about 10-15 H100-min per model: 380 items, shorter than conftrap's 760 items at about 1100-1450 s. PLAN gives no budget line for Alex; add one.

---

## SHOULD FIX

### S1. cpkc dedupe against the pilot and smoke does not work, so the PLAN claim "cpkc also by passage+question" is false.
`make_wp.py:35` versus `make_wp.py:40,104`.
- Line 104 adds the stripped passages (`q = e["input"].strip()`) to `avoid`.
- Line 35 tests the raw `e["input"]`, and every raw input starts with `"\n"` (checked on all 17,527 examples). So the passage filter never fires.
- The only remaining guard is the full-prompt check at line 108, and the prompt includes the shuffled option order. A reused passage whose options are in the other order gets through.

Expected leakage is about 0.2 items per build (64 used out of 17.5k). None appeared in my 3 builds, so the impact is small. But the claim is pre-registered.

Fix: `e["input"].strip() not in avoid` at line 35.

### S2. In modar, trap and control rows are the same questions. Dedupe by prompt misses this.
I checked all three subtasks: row i of `ma_*_plus_one.json` and row i of `ma_*_control.json` have **identical operands and identical 5 shot lines**. Only the answers differ, by +1.
- Traps and controls are drawn independently (`make_wp.py:47-50`). A build can therefore contain the same question as both trap and control: 1 pair in seeds 4242 and 777001, 0 in seed 99.
- In each build, 2 items share their operand row with a pilot or smoke item. The pilot saw that question with the other answer convention.
- Multiplication has 49 duplicate final queries, and 368 queries reappear as shot lines in other rows.

Fix: draw each subtask's rows once without replacement, and assign them to traps and controls. Add the pilot and smoke operand rows to `avoid`, using the prompt with `-> -?\d+` stripped. Use the operand row as the modar cluster.

### S3. In 2 of 9 cells (or more), error equals trap membership, so the cell tests "find the trap among controls", not "find the error".
`score_wp.py:205-207`; PLAN:31.
- Pilot: OLMo got 0/60 traps right on modar and on base9. Those cells are eligible only because of the 40 easy controls: E = 100 traps, non-E = 40 controls.
- In such a cell, recall(FLAG) − RANDOM is at most about +29: (1 − 100/140) times the flag share.
- Every rival is just ranking trap items against control items. This is legitimate under the frozen rule, but it is a different question from "which wrong answers". The same thing happens inside modar. I computed pilot fast accuracy on modar traps per subtask from out_pilot_*.jsonl:

| model | + | − | × |
|---|---|---|---|
| Qwen | 16/20 | **0/20** | 8/20 |
| Phi-4 | 11/20 | 4/20 | 14/20 |
| OLMo | 0/20 | 0/20 | 0/20 |

So in the Qwen modar cell, "error" is close to "subtraction item".

Fix (pre-register now):
- (a) A report-only H1 on traps only. Use the same eligibility, applied inside traps (E ≥ 15 and non-E ≥ 15 among traps).
- (b) Print per-cell recall split by modar subtask.
- (c) Add one sentence to PLAN saying that in all-wrong cells the comparison is trap detection.

### S4. The cluster bootstrap is again the item bootstrap.
`make_wp.py:41,58,92`; PLAN:32.
- cpkc clusters by question, but there are 17,525 distinct questions among 17,527 examples. So every cluster is one item.
- modar and base9 cluster by prompt.
- The real dependence:
  - cpkc **memorized answers repeat**: Bob Dylan ×52, Dan Stevens ×47, Barry Bonds ×39 in the pool, and substitute names repeat up to ×7;
  - modar trap/control pairs and shared shot lines (S2).

The impact at n=100 is small. But the PLAN text suggests a robustness check that does not exist.

Fix: cluster cpkc by the memorized (habit) name and modar by the operand row. Or state in PLAN that cluster = item, so the second scheme adds nothing.

### S5. cpkc name matching: no word boundaries, and accents are only partly handled.
`score_wp.py:29-34`. Tested:

| line | cands | result |
|---|---|---|
| "Chalino Sanchez" | Chalino Sánchez / Claude Rains | None (falls back to forced choice) |
| "Alice Cooper" | Ali / Alice Cooper | None (both found, `"ali"` inside `"alice"`) |
| "Paul Otlet" | Paul Otlet / Paul | None |
| "Beyonce" | Beyoncé / Rihanna | 0 (right, but only by substring luck: `"beyonc"`) |
| "J.K. Rowling", "**Mason Plumlee**", "The context says Mason Plumlee" | | parse correctly |

- Each build has 2-7 cpkc items with non-ASCII names.
- The pool has 3 items where one name is a prefix of the other (Paul / Paul Otlet, Douglas / Douglas Lima, Steve / Steve Young).
- All of these fall back to forced choice, which is acceptable but avoidable.

Fix:
- Fold accents with `unicodedata.normalize("NFKD", s).encode("ascii", "ignore")` inside `norm`.
- Match on padded strings: `(" " + l + " ").find(" " + c + " ")`.
- Drop a match that lies inside a longer candidate's match.
- Exclude the 3 containment items in make_wp.py:35.

### S6. cpkc answers that name neither option are treated differently from numeric off-list answers.
`score_wp.py:70`. A cpkc final line that names neither option goes to forced choice, while modar and base9 treat the same case as off-list (-1). In smoke, Qwen's sampled answers to item 1 were "not provided in the context..." (4 of 5). A THINK line like "Answer: The context does not say" is plausible.

Fix: pre-register what happens to these lines. Either keep forced choice and print a "cpkc answer names no option" rate per cell, or count lines with a refusal pattern as off-list. Decide now, not after the run.

### S7. PLAN:22-23 overstates how much CONF is "not handicapped by selection".
report_pilot.txt shows conf|wrong of 88-100% in 6 of 9 cells (Qwen in all 3, OLMo in all 3). Only Phi-4 is in the 68-90% range.

Fix: change the sentence to "partly confident; CONF is handicapped in most Qwen and OLMo cells". Carry the conftrap S6 wording into the verdict line (`score_wp.py:261`), so that a pass resting mainly on FLAG − CONF is labelled as such.

### S8. The forking path in test selection is not documented.
data/ also contains crass_ai, misconceptions, presuppositions_as_nli, riddle_sense and unnatural_in_context_learning. They were downloaded but are not mentioned anywhere in PLAN, and I found no trace of them in other v3-shift .py/.md files.

Fix: add a line to PLAN saying which candidates were considered and why they were dropped before any model output (or that they were never piloted). Otherwise "picked before any lamp output" cannot be checked.

---

## NOTES

- **N1. Options shown in some prompts but not others.**
  - **cpkc** shows the options in the prompt (`make_wp.py:41`), which the BIG-bench task does not (`append_choices_to_input: false`). This changes the task from free recall to 2-way choice, and the cpkc result should not be compared with published cpkc numbers.
  - The right name is in the passage in 100/100 items. The habit name is in the passage in only 0-2 per build (157 in the whole pool; those items are ambiguous). So "pick the option quoted in the context" solves cpkc. THINK can get there by string matching, which makes recall(FLAG) on cpkc ≈ P(THINK right | fast wrong) (PLAN:41 says so).
  - **modar and base9** do not show the options. So the fact that the right option is **always the numeric middle in modar** (100/100 traps and controls) and in about 50% of base9 traps cannot cue the model. It only means the forced choice is effectively right vs habit (op+2 is almost never chosen). No change is needed. Optionally say in PLAN that the options are not shown.
- **N2. cpkc length bias in DIRECT.** Summed log-prob over multi-token names favours shorter or more frequent names, which are usually the memorized ones. The margin used by CONF shares this bias. The direct_mean sensitivity readout covers the sign. A cheap report-only addition: agreement of DIRECT argmax with the majority of the 5 samples per cell.
- **N3. base9 habit vs slip.** The habit equals right−1 (as decimal strings) in 11-16% of traps (units-only carry). The tens-carry habits (e.g. 123 vs 133) look like a single-digit slip. PLAN:40 already says base9 tests general error finding. About 30% of habits contain the digit "9", which is invalid in base 9, so a careful THINK can reject them on form alone. That is fine, but worth one line in PLAN.
- **N4. Template.** The prompt matches Wu et al.'s template only up to the question; their "end the response with \boxed{}" tail and the counterfactual comprehension check are absent. Fine, since THINK supplies its own answer format. "\boxed{83}" and "83_9" parse correctly.
- **N5. Smoke pipeline looks healthy.**
  - Parse rate 100% (Qwen, Phi-4) and 75-83% (OLMo). One OLMo cpkc item has no Answer line, and 2 OLMo numeric items are off-list ("150", "123"). Truncation 0%.
  - Batch vs single, same last line: 8/8 for Qwen and Phi-4, 5/8 for OLMo. This is the known noise, and it works against FLAG.
  - Phi-4's sampled answers often run on ("2755. \n\nTo determine..."). Only the first line is read, which is correct.
  - Unparsed base9 samples are almost all real off-list numbers (" 116", " 133"), so "unparsed = differs" is correct there.
  - Wall clock was 120-147 s for 16 items.
- **N6. Pilot hardware.** The pilot ran on L40S (modal_wp_pilot.py), the main run on H100. bf16 forced-choice argmaxes can flip on near-ties. This is irrelevant to the verdict, but the pilot numbers in PLAN are approximate.
- **N7. Silent shortfall.** `make_wp.py:106-110` silently drops duplicate prompts, so a family could come out at 99. Add `assert Counter(...) == expected` after the build. The seed list at line 116 is fine.
- **N8. Contamination.** PLAN:38 covers it. Note that modified_arithmetic and cpkc answers are public verbatim (the canary is present), while base9 items are fresh. Contamination can make cpkc and modar easier for DIRECT, which shrinks E. The base9 numbers are new.
- **N9. Volume paths.** The pilot and main both use `/cache/wp_runs/<tag>`, with different tags, so they do not collide. The hash includes MAXNEW. Rerun and skip logic is unchanged from conftrap and was proven there.
