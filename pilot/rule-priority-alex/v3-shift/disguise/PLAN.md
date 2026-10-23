# Plan v2: "disguising" the task and "shaking" the model

*English translation. The original Russian text (the version referenced by PREREG.md checksums) and the earlier draft PLAN_v1.md are in the git history (commit 6d3f172).*

2026-10-18. Rewritten after review (review/plan-review.md: 4 blockers and 10 required fixes). The first version was PLAN_v1.md. After the technical check I write PREREG.md with checksums, and only then does Alex name the number.

## Question

A habit "from training" fires with confidence, without doubt (general-lamp/rezultaty.md). We test external ways to bring the model back "to reality":
- **disguise**: a module rewrites the task in neutral words;
- **shake**: an unexpected line, a "wake-up", or internal noise ("adrenaline").

Each method is evaluated **on its own**. So that four methods do not produce a chance success, intervals are wider: 98.75% instead of 95% (Bonferroni correction for 4).

**Model:** Qwen2.5-14B-Instruct.

## Task families

**N — "native" road words, rule change without history.** native/make_native.py, arm change_0, 90 sets. The generator runs with numbers S, S+1, S+2; all numbers used before are skipped, including 88–95 from the previous run.
- **N-traps** (D1, D3, with the change note): common sense pulls towards the old answer.
- **N-agree** (D2, D4, with the same note): common sense agrees with the new rule. This is a harm control.
- **N-confirm** (D1–D4): the note "Rule 2 has been reviewed and stays the same". The correct answer is the old rule; the habit is right. Also a harm control.

**R — redefine-math.** The same 810 test questions as in general-lamp. Their behavior in format F3 has already been seen, and Alex's number does not affect R. Templates T0–T8, 90 questions each.
- **R-traps**: all 810, the correct answer per the redefinition.
- **R-same** (harm control): only T0, T1, T2, T7, T8, with everything before "Q:" removed. The correct answer is ordinary math: "first digit of 50+47" = 9 by convention. T3–T6 are not used: after removing the definition only 2 distinct questions remain there.
- **R-agree** (harm control, the definition is present but the correct answer matches the habit). Built from the test questions:
  - T4: "Redefine π as N" → "Redefine π as M", where M is N with first digit 3. Options: "3" (correct) and the former first digit of N.
  - T6: "Redefine e as …" → "Redefine e as 2xy", where xy are the last two digits of the former value. Options: "2" (correct) and the former correct answer. If that is itself 2, the question is skipped.

**Report only:** C-rule (A_change_60 × D1–D4, 90 sets): the "copy from examples" habit, shake only.

## How we ask

- **N and C:** our own chat format, as before.
- **R:** format F2. This is chat, "Reply with only the answer, nothing else.", the answer starts with "The answer is". Chosen in advance by the probability mass on the options in the trial part.
  - Before PREREG the plain model is run in F2 on the **trial** part of redefine (90 questions). Share correct by template and share passing the filter are recorded in PREREG.
- **Answer**: the option with the highest sum of log-probabilities.
  - For the N report separately: by first token only and by the sum divided by the number of tokens. The twins' labels tokenize differently.
- **Filter:** probability mass on the options for the plain version ≥ 0.5. A question is either in all methods or in none.

## Methods

**1. Disguise.** The module sees only the task text.

**N:** a made-up twin from match/make_match.py with the same generator number.
- The twin is built in a **separate process**, so that make_native does not substitute the words. The code checks that the pair's words differ while cases, keys and label positions match.
- The label is translated back by its position in the triple.
- This is a **"perfect" translator**: it knows the generator's word map.
- The expectation was written in advance from existing answers: on D1/D3 about +23 points, on D2/D4 about −18. The twin errs there itself, because common sense no longer gives hints. The harm on N-agree will probably fail the method.

**R:** an algorithm over the text.
1. Find the definition "Redefine X as" or "View X as".
2. If X is a symbol (π, e, +, ÷), replace it everywhere **except in the value of the definition itself**: π → "zorp", e (as a separate word) → "quib", + → "§", ÷ → "¤".
3. If X is not a symbol (T0 "View number as text"), the module changes nothing.
4. Without a definition (R-same) the module also changes nothing.

A rewritten example of each template from the trial part is recorded in PREREG.

**The module's decision.** On disagreement the disguised answer is taken. This is the same as "always disguised", because the plain version changes nothing.

**Report only:** how well the disagreement itself catches habit errors (AUC and accuracy on "agreeing" cases).

**2. Shaking with a line.** One line before the question line: "Classify the TEST case…" for N and C, "Q:" for R. The texts are written in advance; the exact strings with spaces go into PREREG.
- **2a "surprise":** "BANG! A loud alarm suddenly goes off, then stops. Silence."
- **2b "wake-up":** "Wake up. Look at this task as if you are seeing it for the first time."
- **Placebo:** "The next line continues the same task, as written above it." A neutral line of about the same length in tokens, without calm and without a shake.
- **Instruction (comparison):** "Follow the rules stated in this task exactly, even if they contradict what you know." A simple request that must also be beaten for a shake to count as a separate effect.

**3. "Adrenaline".**
- Noise is added to the output of layer model.model.layers[20] (of 48) **at every position, including the option tokens**.
- Noise size at a position = α × the length of the state vector at that position.
- The noise is generated with seed (12345 + question number) over the whole sequence. Each option is scored with a full pass, and the shared part of the noise is the same across options.
- **α is chosen before PREREG** on the trial part of R and the technical N by the rule: the smallest of {0.05, 0.1, 0.2, 0.4} at which ≥ 5% of answers change.
- **Checks:**
  - α = 0 gives exactly the plain answer;
  - the same seed gives the same answer;
  - with α > 0 answers change.
- **Comparison with random flipping.** The expected accuracy if the same share of answers were changed at random: for two options, to the opposite; for three, to a random one of the other two.

## What we measure

Share of correct answers. For understanding, separately: the share of habitual answers on traps.

**R is averaged over templates with equal weights**, taking only templates where the plain model has a habit: correct < 85%.

## Criteria (each method separately; 98.75% intervals, paired bootstrap, 2000 resamples)

A method **passes** if all of the following hold:

1. **N-traps:** gain ≥ +10 points, lower bound > 0.
2. **R-traps:** gain ≥ +10 points, lower bound > 0. Gain > 0 in at least half of the templates with a habit.
3. **Harm.** On N-agree, N-confirm, R-same and R-agree each change ≥ −5 points, lower bound ≥ −10.
4. **Only for 2a, 2b:** better than both placebo and instruction by ≥ +5 points on traps. N and R weigh 0.5 each, bootstrap for each separately, lower bound > 0.
5. **Only for 3:** better than random flipping of the same share by ≥ +5 points on traps, lower bound > 0.

**If there is no habit.** If the plain model has ≥ 85% correct on a family's traps, the gain on that family is not evaluated. Then a method that passes on the remaining family gets the result **"partial"**, not "passes". If neither family has a habit: "nothing to say".

**Bootstrap:**
- N and C: over sets;
- R: over questions within a template;
- R-same: over distinct question texts.

**Report only:**
- other α;
- shaking on C-rule;
- disagreement as a detector;
- first-token and per-token-mean scoring for N;
- results by template;
- N-traps for D1 and D3 separately.

## Cost

About 20 thousand short passes plus 2500 long ones (C-rule) plus trial runs. About $3 on H100.

## Order

1. Code.
2. Technical run on 12345:
   - F2 on the trial part of redefine;
   - choice of α;
   - twin and noise checks;
   - disguise examples.
3. PREREG.md with checksums and the chosen values.
4. A number from Alex.
5. Run, scoring by the frozen script, report as is.
