# "Rule change" test, version 2

The second version of the mini-pilot (step 1 of the roadmap). Two files:

- `run_v2.py` runs the model, writes tables and prints a summary at the end;
- `analyze_v2.py` builds the summary from finished tables. It can be run separately, without the model.

The questions are the same as in v1:

- **H1:** if one rule is reinforced for a long time, is the model pulled to answer by it when the rule has changed?
- **H2:** is there such a pull inside the model, in the middle layers, even when the final answer is correct?

## How the experiment works

There are two colored frames, BLUE and ORANGE. One has the rule "circle → LEFT, square → RIGHT", the other the reverse.
There is also a third, grey frame GRAY: for any shape the answer is SKIP.

The frame trained by the history is called H ("habit"). The other frame is called S ("switch").
Rule S is always the reverse of rule H.

Each task is assembled as follows:

1. **Header.** A description of the game and three rules. The words "usual" and "new" do not appear in it.
2. **Training block.** 16 solved examples, balanced over both frames and both shapes.
3. **History** of N lines. There are two conditions:
   - `habit`: N solved examples of frame H (the habit);
   - `filler`: N grey lines of the form `Frame: GRAY. Shape: star. Answer: SKIP`. Same form, but they teach no rule.
4. **Test.** One task with frame H or S, answer empty.

N is taken from the series 0, 2, 4, 8, 16, 32, 64. At N=0 there is no history, and the task is the same for both
conditions. So N=0 is computed once and is called `shared` in the summary.

The answer is read without randomness: the probabilities of the words LEFT and RIGHT are compared. The same comparison
is made at every layer of the model ("logit lens").

## What changed compared with v1 and why

1. **The model must know both rules before the experience.** In v1 at N=0 the small model confused the frames. Now the
   training block has 16 examples instead of 4. A gate was added: at N=0 at least 85% correct is required on each
   frame (details below).
2. **An honest control.** In v1 the `mixed` control added examples of the new rule, and the model learned from them.
   Now the control is `filler`: grey lines of the same form and almost the same length. Examples of rule S are only in
   the training block. Their number is the same for any N. Only the number of rule-H examples depends on N.
3. **Cross-balancing.** In v1 the rules were the same in all tasks. On the square the old answer was always RIGHT,
   and the "hidden pull" sat almost only on squares. The model was already leaning towards RIGHT, so the skew looked
   like habit. Now six factors are fully crossed, each combination occurring equally often (64 cells):
   - which color is habitual (BLUE or ORANGE);
   - which rule is habitual (circle → LEFT or circle → RIGHT);
   - which rule line comes first in the header (H or S);
   - the test frame (H or S);
   - the test shape (circle or square);
   - the frame of the last training example (H or S).

   Now the habitual answer is LEFT and RIGHT equally often. A skew towards one side can no longer pass for a habit.
4. **The history comes right before the test.** In v1 there was a training block between the history and the test.
   Now the habit comes last, as for humans: you do one thing for a long time, then suddenly another. The training
   block is the same in all conditions.
5. **A side-free measure.** The main measure is now `h`: how much the model leans towards the habitual answer
   (details below). Since the habitual answer is LEFT and RIGHT equally often, a skew towards one side cancels out on average.
6. **The H2 window is chosen in advance.** Layers here are numbered from 0. In v1 the answer formed in a jump at layer
   16 (the 17th of 24), and the "hidden pull" was in layers 12–15. Now the window is set by a formula before the run:
   from round(0.50·L) to round(0.67·L)−1, where L = number of layers. For 0.5B (24 layers) that is layers 12–15, for
   1.5B (28 layers) layers 14–18.
7. **Honest statistics.** The unit is the task. Each task is run at all N and in both conditions (paired comparison).
   Intervals: bootstrap over tasks (2000 resamples, fixed seed), Wilson interval for proportions, exact McNemar test
   for paired proportions. The "supported / not supported" rules are written in advance.
8. **More tasks and denser small N.** 512 tasks instead of 100, series N = 0, 2, 4, 8, 16, 32, 64.
9. **Faster.** Tasks for different N start identically. So the model reads the longest prefix once and caches it
   (KV cache). For each N the script truncates the cache to the needed length and reads only the test line. This is
   about 2 times faster. If on some model tokenization does not allow this, the script runs that task in full by
   itself. The `--check-cache` flag compares both methods.

## Running on Windows (CPU, no GPU)

1. Python 3.10 or newer from python.org is needed (tick "Add Python to PATH").
2. Put `run_v2.py` and `analyze_v2.py` in one folder. Open a command prompt in it.
3. Install the libraries (if not already installed after v1):
   ```
   pip install -U torch transformers
   ```
4. A quick check without the model (seconds). Prints tasks for two examples, computes nothing:
   ```
   python run_v2.py --dry-run
   ```
5. Main run, model 0.5B:
   ```
   python run_v2.py --check-cache 4
   ```
   `--check-cache 4` compares the fast method with the full one on the first 4 tasks. This adds a minute or two.
   If the difference exceeds 0.001, the script stops with the message `check-cache FAILED`.
   Then run with `--no-cache` (twice as slow, but without the cache).
6. Then model 1.5B:
   ```
   python run_v2.py --model Qwen/Qwen2.5-1.5B-Instruct --check-cache 4
   ```

On first run the model is downloaded from Hugging Face: about 1 GB for 0.5B and about 3 GB for 1.5B.
About 3 GB of free memory is needed for 0.5B and about 10 GB for 1.5B.

**How long to wait (estimate, not measured).** About 1–2 hours for 0.5B and 4–6 hours for 1.5B on a 10-core CPU.
The estimate was made on a random-weight model of the same size and checked against the speed of the v1 run on this
computer. While running, the script prints how many seconds each task takes and roughly how much time is left.

**Results** are in the folder `results_v2\<model>-seed1\`, e.g. `results_v2\Qwen2.5-0.5B-Instruct-seed1\`:

| File | Contents |
|---|---|
| `trials.csv` | one row per task, condition and N |
| `layers.csv` | `h` at each layer for each row of `trials.csv` |
| `run_info.txt` | model, library versions, time, cache check |
| `summary.txt` | the summary (the same as printed at the end) |

Rows are written as the run proceeds. If the run was interrupted, start it again: there is no resume.
`analyze_v2.py` can also be run on an unfinished folder (while the run is going or after a crash). The last table row
may then be cut off in the middle. Such rows are skipped, and the summary shows
`WARNING: skipped incomplete rows`: the summary covers only rows already written.

Recompute the summary without the model:
```
python analyze_v2.py results_v2\Qwen2.5-0.5B-Instruct-seed1
```

Other `run_v2.py` flags: `--trials` (number of tasks, a multiple of 64, default 512), `--n` (series of N),
`--learn` (training block size, a multiple of 4, default 16), `--seed` (task layout, default 1),
`--out` (results folder), `--no-cache` (no cache), `--boot` (number of bootstrap resamples, at least 1).
`--dtype` sets weight precision: `float32` (default, for CPU) or `bfloat16` (for GPU and 7B models).
Changing `--learn` or `--n` makes it a different experimental plan, which must be written down before the run.

## Gate

H1 and H2 make sense only if the model knows both rules without any history. So the first line of the summary
checks N=0: the share of correct answers on frame H and on frame S, each with a 95% interval.

- **PASS:** both shares at least 85%. H1 and H2 can then be read.
- **FAIL:** at least one share below 85%. Then the summary says H1 and H2 cannot be read and adds the mark
  `(gate FAILED: not interpretable)` to their results. The tables are still printed and can be viewed as a description.

If 0.5B does not pass the gate, that is expected after v1. The main run is then on 1.5B.

## How to read the summary

Words that appear in the summary:

- **H** and **S:** the habitual frame and the frame with the rule change.
- **h:** log-probability of the habitual answer minus log-probability of the other answer, in nats.
  Above 0: the model leans towards the habitual answer. Below 0: towards the other. On an S task the other answer is
  correct, so h > 0 on S means a pull towards the old rule.
- **Nats:** the unit of log-probability. A difference of 1 nat means one answer is about 2.7 times more likely than
  the other. 2 nats: about 7 times.
- **Layer index** is counted from 0. The last index (23 for 0.5B) is the model output.

The summary starts with the lines of `run_info.txt`: model, versions, time per task, number of runs without cache
(`cache fallbacks`) and the `--check-cache` result. Then a line with the number of layers and the H2 window.

### 1. GATE

Example (made-up numbers): `H frame 92% [88%-95%] (236/256)   S frame 88% [83%-91%] (225/256)   -> PASS`.
Share correct, 95% Wilson interval, number correct out of number of tasks. The PASS and FAIL rule is described above.

### 2. BEHAVIOUR (what the model does under each condition and N)

| Column | Meaning |
|---|---|
| `cond` | `shared` (N=0), `habit` or `filler` |
| `N` | history length |
| `H acc` | share correct on frame H |
| `S old [95% CI]` | share of S tasks solved by the habitual rule (old-rule error), with interval. 50% is guessing |
| `D` | direction index, 2·`S old` − 1. +1: the model answers by the habitual rule. 0: the shape does not affect the answer. −1: the model answers by the correct rule S |
| `S RIGHT` | share of RIGHT answers on S tasks. Far from 50% means a skew to one side |
| `final_h S` | mean h at the model output on S tasks, ± standard error |
| `p_mass` | mean total probability of the words LEFT and RIGHT. About 1: the model answers with game words. About 0: it wants to say something else, and conclusions are shaky |

The main comparison: `habit` vs `filler` at the same N. If the habit works, in `habit` the index D and
`final_h S` are higher than in `filler`, and the difference grows with N.

### 3. H1 (main test, rule written in advance)

For each S task take the difference `final_h(habit) − final_h(filler)` at one N. Above 0: the habit history pulls
towards the old answer more strongly than the same number of grey lines.

- Lines `N / items / mean diff / 95% CI`: the mean difference at each N (2 and above) and its interval.
- Line `slope vs log2(N)`: the slope of the difference over log2(N), i.e. how much the difference grows with each
  doubling of N. **H1 is supported (`SUPPORTED`) if the whole slope interval is above 0.** Otherwise `NOT SUPPORTED`.
- `Extra ... by habit-answer side`: the same separately for tasks where the habitual answer is LEFT and where it is
  RIGHT. Not part of the decision. If the signs differ, that is a skew to a side, not a habit.
- `Secondary: OLD_RULE_ERROR habit vs filler`: shares of old-rule errors in the two conditions. `only habit` and
  `only filler`: how many tasks erred in only one condition. `p`: exact McNemar test.
- `Secondary: within habit`: within `habit`, the smallest and largest N (2 and 64) are compared, also by McNemar.

### 4. H2 (main test, rule written in advance)

The same, but instead of the model output the mean h in the layer window is used (`window_h`, for 0.5B layers 12–15).
Question: is there an internal pull towards the old answer that grows with habit.

- Lines per N and the `slope vs log2(N)` line over all S tasks, as in H1.
- `Secondary`: the same difference by group. For each group: number of tasks, mean difference at each N (`by N`)
  and slope with interval:
  - `habit ans LEFT` and `habit ans RIGHT`: habitual answer LEFT or RIGHT;
  - `shape circle` and `shape square`: test shape;
  - `correct both, LEFT` and `correct both, RIGHT`: only tasks solved correctly in both conditions, separately for
    habitual answer LEFT and RIGHT. This is the "hidden pull with a correct answer". Caution: these tasks are
    selected by the model's answer. If the model leans to one side, almost all of them will fall on one side, and a
    simple side skew will look like a pull. So there is no combined line. Treat it as a hidden pull only if both
    lines show the same thing.
- Result: **H2 is supported if the interval of the overall slope is above 0 and the slope interval is above 0
  separately for LEFT and separately for RIGHT** (lines `habit ans LEFT` and `habit ans RIGHT`). The second condition
  protects against the v1 error, where the effect was only on one side. A simple "slope above 0 on both sides" is not
  enough: if one side has no effect, its slope will still be slightly above 0 about half the time due to noise. The
  price of the rule: a very weak effect is supported less often, because each side is only half the tasks.
  The reason for rejection is given in brackets.

### 5. DECISION LAYER (where the answer forms)

Correctly solved H tasks are taken. For each condition and N, the first layer is found where the mean h exceeds half
of its final value. The mean h is computed separately for tasks with habitual answer LEFT and RIGHT, then the mean of
these two curves is taken. This cancels the logit lens's skew to one side in early layers (which v1 warned about).
If correct H tasks exist on only one side, the skew cannot be removed, and the line is skipped with a note.

| Column | Meaning |
|---|---|
| `trials L/R` | how many tasks were included: with habitual answer LEFT / RIGHT |
| `mean curve` | this layer by the mean curve |
| `jump` | what share of the final value the curve gains at this layer. Close to 1: the answer forms in a jump |
| `per trial` | the same per task, after subtracting the side skew at each layer: median [quartiles] and the most frequent layer with its share |

Below are two distribution lines: by condition × N cell and over all tasks together.
In v1 on 0.5B this was layer 16. If it is again 16 and `jump` is large, the finding "the answer forms in a jump" has replicated.

### 6. Breakdowns of old-rule errors

The share of old-rule errors on S tasks separately by the levels of each factor (first level / second):
habitual answer LEFT/RIGHT, shape, habitual color, order of lines in the header, frame of the last training example,
whether the last shape of the history matches the test shape (only `habit` with N > 0).
A large gap within a pair means this factor moves the answers, not the habit.

## What was checked before handover

All checks were done in the cloud, on a tiny random-weight model (meaningless, only to check that it works).
The user's computer was not touched.

- A full run from start to summary.
- The fast cached method gives the same numbers as a full run (difference about 0.000001). A broken cache
  stops `--check-cache`.
- All 64 cells occur equally often. Each task is identical across all N and conditions. The answers in the examples
  match the rules in the header.
- The summary was checked on fake "models" with a known answer: "always by habit" (D = +1), "always correct by the
  switch" (D = −1), "always RIGHT" (D = 0, `S RIGHT` 100%, H2 not supported).
- A hidden pull on only one side (only LEFT or only RIGHT), with random noise: H2 was supported in 4 of 80 repeats,
  about 5% (with the old rule "slope above 0" it was 40 of 80). The same pull on both sides was supported in 40 of 40,
  and a model with no effect at all in 0 of 40.
- A row cut off in the middle of the tables does not break the summary. `--boot 0` is rejected immediately.
