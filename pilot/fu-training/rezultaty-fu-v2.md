# "Fu" training v2: notice a change in any words and check whether it applies to me

Qwen2.5-7B-Instruct, Modal L40S, 2026-10-17. The same scheme as v1 (DPO + LoRA), 2100 pairs. Added to v1:
- **a change in plain words**, without "NOTICE" (500 pairs): "good" = the new, "fu" = the old;
- **decoy announcements** about something else (550 pairs): another system, another person, a rejected proposal. Here "good" = rely on experience, "fu" = apply someone else's change.

Test: 1260 tasks. The wordings of changes and decoys in the test differ from those in training. Math and tariffs never appeared in training.

## Summary (share correct)
| | before training | v1 | **v2** |
|---|---|---|---|
| familiar task kinds, traps (experience misleads) | 24% | 72% | **88%** |
| familiar kinds, experience fits (incl. decoys) | 98% | 98% | **100%** |
| unseen kinds, traps | 67% | 95% | 88% |
| unseen kinds, experience fits (incl. decoys) | 98% | 79% | **87%** |
| MMLU 570 | 75.1% | 75.6% | 74.6% |
| GSM8K 250 | 90.4% | 88.4% | 89.6% (9 broken, 8 fixed) |

## Details for the important rows
| task | before | v1 | v2 |
|---|---|---|---|
| rule changed, said in plain words (new wordings) | 0% | 32% | **75%** |
| rule changed, plain words (second check) | 3% | 42% | **80%** |
| record changed, plain words | 45% | 98% | 92% |
| decoy about rules / records | 97–100% | 97–100% | 100% |
| tariff: new price announced above the list (unseen kind) | 12% | **94%** | 68% |
| tariff: new price in plain words | 12% | 73% | 73% |
| tariff: decoy "NOTICE" about another service | 97% | 62% | **75%** |
| tariff: decoy in plain words about another service | 95% | 68% | **82%** |
| math with an added condition | 94% | 100% | 88% |

## Conclusion
1. **On familiar task kinds v2 does what we wanted.** It notices a change in any words (+75…80 points) and does not confuse other people's announcements with its own (100%). This is "notice → check whether it applies to me → answer".
2. **Over-caution on unseen tasks decreased but did not go away.** On tariffs, decoys 62% → 75% and 68% → 82%, while before training it was 95–97%.
3. **The price: the model became more cautious and on unseen tasks changes its answer less often when it should.** Tariff with the price above the list: 94% → 68%. Over all unseen tasks v1 and v2 are almost equal (90% and 88%). v2 is more honest: it follows the new less blindly.
4. Benchmarks within chance.

Bottom line: the model learned "check whether it applies to me" on the task kinds it was trained on. It transfers to new kinds only partly. For the skill to become general, what is most likely needed is **breadth of training**: more different kinds of tasks, not more pairs of one kind.

Files: make_fu2.py (v2 data), train2.json, check2.json, test_all.json, compare_fu.py, compare_out.txt, fu_Qwen2.5-7B-Instruct_v2.jsonl,
fu_Qwen2.5-7B-Instruct_check2_v1.jsonl. v2 adapter: Modal volume silo-hf-cache, /fu/adapter_v2.
