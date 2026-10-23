# Watchman on independent public tests, result (2026-10-18)

The plan and the prediction were written before the run: PREREG.md. Qwen2.5-14B, settings unchanged. Check "nothing flagged → answer as usual": difference 0.0. Output: score_pub_watch_out.txt.

| test | normal | with watchman (b40) | difference [95%] | verdict per plan |
|---|---|---|---|---|
| hindsight-neglect-10shot (315) | 45.4% | 50.2% | **+4.8 [+2.2, +7.6]** | benefit shown |
| repetitive-algebra (1000) | 98.1% | 98.5% | +0.4 [0.0, +0.9] | benefit not shown (the model almost never errs anyway) |
| MMLU (500) | 71.6% | 72.2% | +0.6 [−0.2, +1.6] | no harm (threshold −2) |

- **The prediction did not come true.** Before the run I expected no gain. On hindsight the benefit is small but reliable.
- **How often the watchman fired.** It flagged 16–26% of examples as "not matching what the model would answer itself now" and fired on 58–94% of tasks. Still, it did no harm on any test.
- **watch_all (silencing in all heads).** Worse: algebra −4.4, MMLU −1.4. Silencing must be limited to the copying heads.
- **Why it helped on hindsight is not tested.** A possible reason: it silenced examples the model leaned on as a template ("lost → wrong"). This is a guess.
- **Limitations.** One model. I picked the tests for copy traps, although others built them. On hindsight the gain is small: 50 out of 100 is still coin level.

Note: the later independent audit (`../audit/audit-report.md`) found that this hindsight gain is a pure class shift (always answering class 1 gives 54.9%).
