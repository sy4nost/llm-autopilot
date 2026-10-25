# three-forces: review before the run (2026-10-24)

**Verdict: ready with changes (6 required, listed at the end).** The design is sound and the arms are correctly paired. But as written, score_tf.py crashes before Q2-Q4, the smoke test only covers N=0, OLMo will probably run out of GPU memory, and hexagon_formula leaks its own fact.
Checked by: rebuilding sets_tf.json locally (byte-identical to the one in the folder), counting turns in every prompt, testing `num()` on edge cases, and running score_tf.py on fake outputs (in scratch, not this folder).

## Bugs (code)
1. **score_tf.py crashes at Q2.** `"... said% B - A %+.1f ..." % diff(...)`: the `% B` is read as a format code, so you get `ValueError: unsupported format character 'B'`. Lines 68-70 all have this (`said% B`, `shown% C`, `said% A`). Fix: write `said%%`, `shown%%`. After that fix, the summaries run on fake data.
2. **The smoke subsample is all N=0.** `items[::len//n]` with n=64 gives a stride of 40, and 40 is a multiple of 8 (the number of N levels), so every sampled item is N=0. The smoke test never sees a long prompt, which hides the OOM in item 3. Fix: `random.Random(0).sample(items, n)`, or a stride of 37.
3. **OLMo-2-13B will probably run out of memory on the L40S (48 GB).** OLMo has no GQA, so its KV cache is about 0.8 MB per token. The longest prompts (N=50) are about 1150 tokens. A batch of 32 is then about 30 GB of KV cache on top of 27 GB of weights. Qwen and phi-4 (GQA, about 0.2 MB per token) fit. Fix: cap batches by tokens (e.g. BS=8 when N>=16, or BS=8 for olmo13), or use an 80 GB GPU. Also, results are written only at the very end, so a late OOM loses everything. Save every few batches or on exception.
4. **score_tf.py divides by zero on partial outputs.** With a smoke run, `full` is empty and `boot_ci` gets n=0, and the per-fact table can also be empty. Guard both (skip the summaries when `len(full)==0`).
make_tf.py itself is correct: every prompt has exactly N assistant turns, and the final question comes last. N=0 is instruction + final question in one message. `qs[0..N-1]` stays inside the 50 drawn questions. Earlier questions are shared across arms within a (fact, seed). Left padding, the attention mask, and slicing at `input_ids.shape[1]` are correct.

## Confounds (would mislead Q2-Q5)
5. **hexagon_formula has its own fact earlier in the prompt.** Its bank still contains "How many sides does a hexagon have?", because only the exact final string is excluded. It appears in 24 of 70 arm-A prompts with N>=1, rising with N (9 of 10 at N=50). In A the model has already seen "hexagon -> 5", and in C "hexagon -> 6". This breaks Q5 (formula vs words) and its N trend. Also drop "How many eyes does a spider usually have?" (answer 8 = M) from the spider bank.
6. **Earlier questions whose true answer equals S, P or M.** Per fact, 5-13% of the shown turns are actually correct answers. Example: hexagon A, "pentagon -> 5", where the shown answer is simply right. hand P=4: 13%. hexagon_formula M: 12%. week P: 11%. The pi facts: 0%. Because the share differs by fact and by value, it adds noise to C-A and to the per-fact contrasts. Fix: per fact, drop bank questions whose answer is in {M, S, P}, plus same-topic questions. The bank still has more than 25 left.
7. **"two" and "one" in the question are echoed back and parsed as the answer.** Tested: `"To two decimal places, pi is 3.14"` -> 2, `"One hand has five fingers"` -> 1, and a list item `"1. 8"` -> 1. All of these become "other", which deflates memory in exactly the pi_* and hand facts. Fix: skip number words and digits that are in the final question's text (or a short stoplist: "two decimal", "one hand"). Better: take the first number that equals S, P or M, and report "first number" as a check.
8. **8 new tokens is too short.** "As instructed, my answer is" and "Following your instruction" contain no number yet, so they score as "other". Answers like "I'm instructed to answer 8, but actually 6" are classed by their first number. Fix: max_new_tokens=40. The cost is trivial. Keep the full text (already saved). Also record `has_M_anywhere` and `n_classes_in_text`, so you can report how many answers mix two classes.
9. **Arm B cannot separate "instruction kept" from "memory recalled"** (said = M, and `cls` tests said first). The same goes for C (shown = M). So for Q2 the null is not 0. Compare said%B - said%A against memory%A (Q4b): an effect counts only if it beats plain recall. Write this into PLAN before the run.
10. **The nonce "drelth hand" still says "hand",** so D-hand probably keeps the memory answer 5. Also, in D `memory=None`, so a 5 is hidden as "other". Fix: in D keep the real M for classification and label it `memory_leak`. Either rename that nonce question or drop hand from Q4.
11. **S = M+2 and P = M-1 for every fact.** That means "said" is always the larger number, and size bias cannot be separated from said vs shown. This is optional: swap the S and P values for half the seeds.

## Statistics
12. **The bootstrap resamples (fact, seed, N) items, but the N levels inside one seed share their question prefix.** That makes the CIs too narrow. Fix (2 lines): resample the 80 (fact, seed) clusters, keep all N for each, and keep the per-fact table. Also, the per-fact table currently prints only arm A. Q2-Q4 need per-fact B-A, C-A and A-D (the PLAN promises this). Q5 has no line of its own, only the arm-A table. That is acceptable once item 5 is fixed.

## Cheap additions
- **A no-instruction baseline (strongly advised):** the final question alone, for 8 real + 8 nonce questions = 16 prompts per model. It shows whether each model actually knows M ("sides(hexagon) = ?", "pi = ?") and what it says for the nonce words. Without it, "memory pull" is not grounded.
- Optional arm E, shown P with no instruction, N >= 1 (+640 prompts): pattern vs memory with "said" absent.
- Print the share of "other" by fact, with 3 example answers each, so that parse failures are visible.

## Required before the full run
1. Escape the `%` in score_tf.py lines 68-70; guard the empty `full` and per-fact table.
2. Fix the smoke subsample so it covers all N; batch by tokens (or use an 80 GB GPU) for OLMo; save partial outputs.
3. Rebuild the bank per fact: drop the same-fact question (hexagon for hexagon_formula), spider eyes, and questions whose answer is in {M, S, P}.
4. Raise max_new_tokens to 40. In parsing, skip numbers that appear in the final question ("two" decimal, "one" hand), and record whether M appears anywhere in the answer.
5. In D, classify M as `memory_leak`. Fix or drop the "drelth hand" nonce question. In PLAN, state that the Q2 null is A's memory share.
6. Bootstrap over (fact, seed) clusters; print per-fact B-A, C-A and A-D. Add the 16-prompt no-instruction baseline.
