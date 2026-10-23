# Lamp "the prompt conflicts with memory" + brake (14B, 2026-10-17)
A replication of the lamp from Zhao et al. 2024 (a linear detector on the last-token state) and a "take it from the prompt" brake (adding a direction at the lamp's layer). The brake switches on only when the lamp fires.
Files: make_facts.py → facts.json (100 facts × "agrees/conflicts"), modal_conflict.py, conflict_Qwen2.5-14B-Instruct.jsonl, score_conflict.py → score_conflict_out.txt. Cost ≈ $2.

## The lamp
- On our own facts (held-out third): chance up to layer 8, 90% by layer 19, 100% from layer 27. This replicates their result: a conflict signal exists in the middle layers.
- **On other tasks the lamp does not work:** it fires on 100% of everything: redefine, quote, hindsight and even MMLU, where there is no conflict (minimum 0.92). The detector learned "does not look like my agreeing template", not the conflict itself. Transfer to other formats is zero.

## The brake (direction "use only the text" minus "answer from knowledge")
| task | normal | brake ×1 | ×2 | ×4 |
|---|---|---|---|---|
| own facts, agrees | 100 | 100 | 100 | 100 |
| own facts, conflicts (correct = per text) | 32.4 | 32.4 | 32.4 | 32.4 |
| redefine-math | 69.0 | 72.5 | 73.0 | 70.0 |
| quote-repetition | 86.0 | 88.0 | 88.0 | 82.0 |
| hindsight-neglect | 46.5 | 44.0 | 45.5 | 50.5 |
| MMLU | 80.5 | 80.0 | 80.0 | 77.0 |
×1 was chosen in advance on our own facts: redefine +3.5 [+0.5, +7.0], quote +2.0 [0, +4.5], hindsight −2.5 [−5, −0.5], MMLU −0.5 [−2, +1].
- Since the lamp fires everywhere, "brake by lamp" = "brake always".
- On our own facts the brake moves the answer in the right direction (gap −30 → −26…−10) but does not override it: 14B holds on to what it memorized (it does not accept "the capital of France is Hanoi") in 68% of cases.

## Conclusion
1. A conflict signal exists inside, but our detector, trained on one template, catches the template, not the conflict. We need facts in different formats (dialogue, tasks, lists, different languages) and a check on other formats before any brake.
2. The brake gives a small plus where the prompt is right (redefine +3.5) and a small minus elsewhere. It does not override strong memorized knowledge.
3. "Who is right in a conflict" is still unsolved: on our own facts the model essentially behaves sensibly by rejecting absurd text.
