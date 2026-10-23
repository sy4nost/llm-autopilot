# Decisive experiment v3: threshold written before the run (2026-10-17, ~03:05 UTC)

*English translation. The original text, frozen before the run, is in the git history (commit 6d3f172).*

Model: Qwen2.5-7B-Instruct, DPO + LoRA, the same scheme as v1/v2. Training on 8 kinds of our own traps.
Public tests (not in training; base results already obtained in external/ext_Qwen2.5-7B-Instruct.jsonl):
- near in meaning (training has its own analogue): redefine-math, quote-repetition, GSM-Symbolic p1, GSM-Symbolic p2
- far (no analogue in training): NeQA, hindsight-neglect-10shot
- control: GSM-Symbolic main, MMLU 570, GSM8K 250

**Success:** a gain of ≥ +5 points over the untrained model on at least 2 public tests, with the 95% interval excluding 0.
In addition, no public test drops so that its 95% interval lies entirely below −3 points. MMLU and GSM8K do not drop by more than 3 points.
**General skill:** success on at least one far test.
**Failure:** threshold not met → the "fu" line is closed and a negative result is recorded.
