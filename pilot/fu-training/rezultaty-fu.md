# "Fu / good" training: teaching the model not to trust past experience blindly

Qwen2.5-7B-Instruct, Modal L40S, 2026-10-17. LoRA fine-tuning (r=16, all linear layers) with preference learning (DPO):
the model is shown two answers to the same task, one "good", the other "fu".
- **"Fu"**: a blindly copied past answer when it no longer fits: the rule changed, a record was updated, or past decisions were wrong.
- **"Good"**: relying on past experience where it fits (same case, nothing changed). This is needed so that the model does not unlearn using experience.

Training: 1600 pairs of two task kinds, rules and a log of records. Training time ~14 minutes.

The test used words and names that were not in training. Two task kinds (math and tariffs) did not appear in training at all.
Model answer = the option with the highest probability.

## Main points
- 7B knows the rules (98% correct without history), but with a history of past decisions it answers the old way 86% of the time. This is pure automatism.
- After training old answers almost disappeared: 86% → 0%, correct 5% → 85%. Where experience fits, the model still uses it (98–100%).
- **Transfer to an unseen task kind.** Tariffs where a new price is announced above a list of old orders: before training 88% old answers, after 6%. Correct 12% → 94%.
- Attitude to wrong experience: if past decisions contradict the rules, before training the model copied the error in 85% of cases, after in 12%.
- Benchmarks: MMLU 75.1% → 75.6%; GSM8K 90.4% → 88.4% (−2.0; 13 broken, 8 fixed, within chance, but worth watching).

## A trick check: did it learn the attitude or just the word "NOTICE"?
| | before | after |
|---|---|---|
| rule changed, said in plain words, without "NOTICE" | 3% correct | 42% |
| price changed, said in plain words | 12% | 73% |
| "NOTICE" present but about something else (the warehouse moved); rule unchanged | 97% | 98% |
| "NOTICE" about the price of **another** service; this price unchanged | 97% | **62%** |

The honest conclusion: the model learned the attitude "new information matters more than old experience" and partly transferred it to new task kinds.
But in part it learned the word "NOTICE" itself: without it the effect is weaker (42–73% instead of 85–94%).
And it became over-cautious: in 38% of cases it applies an announcement that does not concern this task.
So it learned "do not trust the past when something has changed", but has not yet learned **to check whether the change applies to the current case**.

## Next
Add two kinds of pairs to training:
- changes said in different words, without "NOTICE";
- decoy announcements about something else, where "good" = rely on experience and "fu" = apply someone else's change.

Then "fu" will be given for both misses: for blind copying and for blindly following the new. This is closest to your wording "check whether it fits now". A new training run costs about the same, ~15 minutes on L40S.

Files: make_fu.py, make_check.py (data), train.json, test.json, check.json, modal_fu.py (training and testing), modal_check.py,
score_fu.py, score_out.txt, score_check_out.txt, fu_Qwen2.5-7B-Instruct*.jsonl. Adapter: Modal volume silo-hf-cache, /fu/adapter.
