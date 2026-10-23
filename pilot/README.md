# Mini-pilot: a "rule switch" test on a small model

Step 1 of the roadmap. The script `pilot_rule_switch.py` tests two things on Qwen 2.5 0.5B:

- **H1:** do "old rule" errors grow when the habitual rule was reinforced longer (N = 0, 10, 30, 60 examples);
- **H2:** is the model pulled towards the old answer internally (across layers), even when the final answer is correct.

## How the experiment works

The design is Experiment 2 from Impulse Lab. Blue frame: circle → LEFT, square → RIGHT. Orange frame: the reverse.
Each task is a separate prompt to the model: the rules, N solved examples, a short "training" block with both rules
and one test task. The answer is read without randomness: the probabilities of the words LEFT and RIGHT are compared.

Two conditions of equal length:
- **habit**: all N examples are blue (the habit is reinforced);
- **mixed**: the same N examples, but blue and orange alternate (control: same text length, no habit).

H1 is supported only if habit has more old-rule errors than mixed and the difference grows with N.
Otherwise the growth in errors is explained simply by text length.

## What was checked before running

The script was run on a tiny random model (meaningless, only to check that it works) and reviewed by three independent
reviewers. Fixed: double normalization in the "logit lens", balancing of tasks by shape and by frame, identical tasks
for all N (paired comparison), text-length control, a "hidden impulse" threshold with a null level, compatibility with
older versions of transformers.

Important for reading the result: the "hidden impulse" threshold is built so that about 10% of correct blue tasks
exceed it. So H2 is judged as the **difference** between orange and blue (columns `hidden ORANGE` and `null BLUE`),
not by the number of orange ones on its own.

## Running on Windows

1. Install Python 3.10 or newer from python.org (tick "Add Python to PATH").
2. In a command prompt in the script folder:
   ```
   pip install -U torch transformers
   python pilot_rule_switch.py
   ```
   On first run the model (about 1 GB) is downloaded from Hugging Face. No GPU is needed.
3. Result: a table on screen and the files `results/trials.csv` and `results/layers.csv`.

## How to read the table

| Column | Meaning |
|---|---|
| `BLUE acc` | Accuracy on blue (habitual) tasks. If it is around 50%, the model did not understand the game and the rest cannot be read |
| `OLD_RULE_ERROR` | Share of old-rule errors on orange tasks, with a 95% interval. Chance level is 50% |
| `pull` | Mean "pull" towards the old answer at the model output (above 0 = pulled towards the old) |
| `hidden ORANGE` / `null BLUE` | Hidden impulses on orange tasks vs the background level on blue |
| `bias` | The model's skew towards LEFT (plus) or RIGHT (minus) |
