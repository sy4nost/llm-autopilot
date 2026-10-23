# General lamp: pre-registered plan (frozen before tasks with number 88 are generated)

*English translation. The original text, frozen before the run, is in the git history (commit 6d3f172). The `PLAN.md` checksum below refers to the original Russian PLAN.md in that commit; the current PLAN.md is its English translation.*

Written 2026-10-18. Full plan: PLAN.md v2.1, approved by the reviewer (review/plan-review.md, review/plan-review-v2.md).
This file and the listed scripts do not change after writing. Checksums are at the end.

## What was fixed by the technical check (number 12345)

- **Answer format for public tasks and MMLU: F3** (raw prompt, no chat). Chosen automatically by the rule from the plan on a trial part that is not scored.
  - Median probability mass on the answer options: F1 — redefine 1.00, NeQA 1.00, quote 0.000; F2 — 1.00, 1.00, 0.000; F3 — 0.39, 1.00, 1.00.
  - F3 has the highest worst-case result. File devformat_out.json.
- **Scoring of options.** First one pass over the prompt, then each option is continued from a copy of the cache. A check against a full pass without cache showed a maximum difference of 0.10 log-probability for the best option. This is bf16 noise; the method is the same for all prompts.
- **Technical run** (meta_smoke.jsonl, report_smoke.txt): all families go through the whole pipeline; filtering and "non-informative" marks work.
  - The run took the first 30 test questions of each public family. They will also be in the main run.
  - This run shows: on P-quote in format F3 the model made no errors out of 30. So P-quote will most likely be non-informative, and P → C is then non-informative by the rule. The plan **does not change** because of this.
  - The report will include results without these 30 questions (report only).
- **Order of events.** Alex named the number 88 before the code was frozen. No task with number 88 was created before this file was written.

## Run

1. `python3 -I make_gl.py 88 sets_gl.json` — generators with numbers 88…95 (none of them used before).
2. `modal run modal_gl.py --mode main --fmt F3 --sets sets_gl.json --seed 88 --tag main --n-boot 2000`.
3. The verdict is printed by analyze_gl.py (report_main.txt). Criteria and the final table are taken from PLAN.md v2.1.

## Checksums (sha256)
- `PLAN.md` 155410b9be79d71b5af1c0350e46f86394aee2815de8cfcc2adbf69297b78314
- `make_gl.py` 14a7327001bd82c3a120889ba1e27750cbd06833cc169713a8f2765219a6a25f
- `modal_gl.py` 9abd598569388ea6ddfcdcb274e24f0b0100fe69c3c754d696d2bb4cd783d224
- `analyze_gl.py` b1a0e6043ca8983914671e9c5cbb18d254fc98bc3e73541bc5fde992ded607fc
- `../match/make_match.py` 7654bb8d8ed7303dc6c66bc5450e74e4c5377ec3f23c0f04d8aae5036f4def30
- `../make_shift.py` da84158a0bd9897e6a3aa930a92c243e8635a8a860284dfbe06d0e250b5502fd
- `../newtasks2/make_new2.py` d61c64eab044d423b86d470ade67d3e9e019a5d622456d27d4c4a96aa3942c40
- `../native/make_native.py` 88dbe7f0fad2315ee15c5338860940c4f53d9dc9354ead260e5ff77a703e78e4
- `devformat_out.json` 48f19981ac350a385c4729751b2b8500803b748ffcebaf695420bba46094c539
