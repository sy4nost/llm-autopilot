# Rule-switch test v2: spec

Context: v1 (/mnt/project-files/pilot/pilot_rule_switch.py, results in /mnt/project-files/pilot/rezultaty-pilota-2026-10-15.md and /mnt/project-files/pilot/run-2026-10-15/) found:
- at N=0 the 0.5B model is frame-blind (header + 4 learn examples are not enough);
- the "mixed" control was confounded (it adds N/2 switched-rule demos);
- habit-frame errors on switched trials were shape-independent guessing with a RIGHT lean, so "OLD_RULE_ERROR" was not perseveration;
- the H2 "hidden impulse" effect lived only on one shape (where the old answer was RIGHT), so it is confounded with side bias;
- the answer forms abruptly at logit-lens index 16 of 24 in Qwen2.5-0.5B; the "hidden pull" sat at indices 12-15.

v2 fixes these. Deliverables go in /mnt/project-files/pilot/v2/:
- `run_v2.py` (runs the model, writes CSVs, then prints the summary by calling analyze_v2),
- `analyze_v2.py` (reads the CSVs, prints and writes `summary.txt`; can be re-run alone),
- `README.md` in Russian (plain language, like /mnt/project-files/pilot/README.md): what changed vs v1 and why, how to run on Windows (CPU), the gate, how to read each summary line.

Dependencies: only `torch` and `transformers` plus the Python stdlib (no numpy/scipy/pandas on the user's PC). Must work with transformers 5.x (tested here with 5.20) and should not use deprecated `torch_dtype`; load with `.float()` like v1.

## Roles and counterbalancing

Two rule frames plus a neutral one. Call the practiced frame H ("habit") and the other S ("switch"). S's mapping is the reverse of H's.
Per item, crossed exactly (each combination equally often):
1. habit colour: H is BLUE or ORANGE (S is the other);
2. habit mapping: under H, circle->LEFT/square->RIGHT, or circle->RIGHT/square->LEFT;
3. header order: H rule line first or S rule line first;
4. test frame role: H or S;
5. test shape: circle or square;
6. last acquisition example's frame role: H or S.
That is 64 cells; `--trials` must be a multiple of 64 (default 512). Items are paired: each item's test trial, acquisition block and history/filler shapes are fixed and reused for every N and condition.

Header (identical in every condition): a one-line game description; the two rule lines in the item's header order, e.g. `BLUE frame: circle -> LEFT, square -> RIGHT.`; then `GRAY frame: any shape -> SKIP.`; then `Answer each trial with one word.` No "usual"/"switched" labels.

Prompt = header -> acquisition block -> history (N lines) -> test line.
- Acquisition: `--learn` examples (default 16, multiple of 4), balanced over H/S x circle/square, shuffled per item, with the last example's frame role set by factor 6 (swap to satisfy it).
- History, condition `habit`: N solved H-frame trials with per-item random shapes.
- History, condition `filler`: N lines `Frame: GRAY. Shape: <triangle|star>. Answer: SKIP` (random per item). Same line format as trials, practices neither mapping.
- N grid default: 0 2 4 8 16 32 64. N=0 is the same prompt for both conditions: run it once and report it as shared.
- Test line: `Frame: <COLOUR>. Shape: <shape>. Answer:`.
- Line format of solved trials as in v1: `Frame: X. Shape: Y. Answer: Z\n`.
- The history for N is the FIRST N lines of one per-item sequence of length max(N), so prompts for increasing N are prefixes of each other (this enables KV-cache reuse).

## Readout

As v1: logprobs of the first tokens of " LEFT" and " RIGHT" at the last position; logit lens over every layer (final norm + unembedding on `hidden_states[1:-1]`, final layer taken from `logits`, no double norm). Assert the two first tokens differ. Also record `p_mass` = p(LEFT)+p(RIGHT) at the output.

Speed: use KV-cache prefix reuse. For each item and condition, run the longest prefix (header+acquisition+history[:Nmax]) once with use_cache, then for N in descending order crop the cache to the token length of header+acquisition+history[:N] and run only the test-line tokens with `output_hidden_states=True`. Verify the tokenization prefix property (ids of the full prompt for N == prefix ids + test ids) and fall back to a full forward for that N if it fails. Provide `--no-cache` (always full forward) and `--check-cache K` (for the first K items compare cached vs full-forward final logprobs and per-layer values, report max abs diff, fail loudly if > 1e-3). Verify the cache cropping API on transformers 5.20 in this container.

## Per-trial measures (trials.csv, one row per item x condition x N)

seed, model, condition, n, item, habit_colour, habit_map (e.g. circle_left), header_order, frame_role (H/S), colour, shape, last_acq_role, last_hist_shape_same (blank for N=0 / filler), correct, habit_answer (the answer H's mapping gives for this shape), choice, error_type (H: CORRECT/OTHER_ERROR; S: CORRECT_NEW_RULE/OLD_RULE_ERROR), final_d = lpL - lpR, final_h = logp(habit_answer) - logp(other answer), p_mass, window_h (mean h over the pre-registered window, below).

layers.csv: seed, condition, n, item, frame_role, shape, error_type, layer, h (logp(habit_answer) - logp(other) at that layer). Store h, not "pull", because h is side-free once habit answers are balanced over LEFT/RIGHT.

Window (pre-registered): logit-lens indices from round(0.50*n_layers) to round(0.67*n_layers)-1, where n_layers = number of transformer layers (for 24 layers: 12..15). Print it.

## Summary (analyze_v2.py)

All inference is item-level: items are the unit, and every CI is a bootstrap over items (default 2000 resamples, seeded) or exact (Wilson, exact McNemar). Use pure Python.

1. Gate: at N=0, accuracy on H trials and on S trials, each with Wilson CI; print PASS if both >= 85%, else FAIL plus a line that H1/H2 below are not interpretable.
2. Per condition x N table: H accuracy; S old-rule rate (Wilson CI); direction index D = 2*old_rate - 1 (+1 habit rule, 0 shape ignored, -1 switch rule); share of RIGHT on S trials (side bias); mean final_h on S (+SE); mean p_mass.
3. H1 (primary, stated in advance): on S trials, paired per item difference (habit - filler) of final_h at each N>=2, with bootstrap CI; plus slope of that difference against log2(N) over N>=2, bootstrap CI over items. H1 supported if the slope CI is above 0. Secondary: exact McNemar on OLD_RULE_ERROR habit vs filler per N; within-habit trend N=2 vs N=64 (exact McNemar).
4. H2 (primary, stated in advance): on S trials, paired difference (habit - filler) of window_h at each N>=2 and its slope vs log2(N), bootstrap CI, over all S items; secondary: the same restricted to items correct in both conditions, and broken down by habit_answer side (LEFT vs RIGHT) and by shape, so a side-specific effect cannot pose as a habit effect. H2 supported if the primary slope CI is above 0 and the effect has the same sign for both habit_answer sides.
5. Decision layer: for each condition x N, the first layer where mean h on correct H trials exceeds half its final value; print the distribution, so we can see if the "answer forms abruptly" finding replicates.
6. Breakdowns that v1 needed after the fact: S old-rule rate by habit_answer side, by shape, by habit_colour, by header_order, by last_acq_role, by last_hist_shape_same.

## Tests before delivery (in the cloud container, never touching the user's PC)

- Build a tiny random Qwen2 model with a byte-level BPE tokenizer (see /tmp/claude-0/-home-claude/6a481795-c8a8-5163-9ff8-7c01dd522236/scratchpad/smoke/make_tiny.py) and run the full pipeline with small settings (e.g. --trials 64 --learn 8 --n 0 2 4) end to end.
- `--check-cache` must pass on the tiny model.
- Item generator: assert every one of the 64 cells has equal counts; assert pairing (same test/acquisition across N and conditions); print a few prompts.
- Analysis: feed analyze_v2 synthetic trials.csv/layers.csv for three fake models (always habit mapping; always switch mapping; always RIGHT) and check D = +1 / -1 / 0, side share, and that H1/H2 verdicts come out as expected (an always-RIGHT model must not show H2).
- Measure seconds per item-condition on the tiny model and report; estimate for Qwen2.5-0.5B and 1.5B on a 10-core CPU only as a rough range, labelled as an estimate.
