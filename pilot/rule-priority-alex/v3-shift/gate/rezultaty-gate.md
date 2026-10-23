# Lamp → brake + "louder new rule" (free, on saved data), 2026-10-17

Model Qwen2.5-14B, sets 15–29, tasks from eras/boost. No new runs. Script gate_offline.py, output gate_offline_out.txt.

**The lamp** (rule set in advance, nothing trained): compare the answer "as is" with the answer when old experience (period 1) is muted in the 40 copying heads.
The lamp fires if the answer changed or the confidence in it dropped by more than half.

**If the lamp fires**, take the answer with "new rule louder" in the middle layers (×4 or ×16). If it does not fire, keep the normal answer.

Correct answers on traps, lamp by full muting of old experience (w0) + ×16:

| | normal | always ×16 | lamp → ×16 | lamp fires |
|---|---|---|---|---|
| A60 (no fresh examples) | 10% | 37% | 40% | 87% |
| E_nocopy | 7% | 30% | 37% | 60% |
| E_copy (a fresh identical case exists) | 70% | 60% | 80% | 30% |
| rule unchanged (N_same) | 100% | — | 100% on 57/60, 3/60 unknown | 5% |
| D2/D4 (old and new answer coincide) | 87–100% | 100% | 93–100% | 0–13% |

With ×4 the picture is the same but weaker: A60 30%, E_nocopy 23%, E_copy 83%.

Conclusions:
- The lamp separates traps (fires on 87 of 100) from an unchanged rule (fires on 5 of 100).
- Switching on "by the lamp" keeps the whole gain and removes the harm where loud boosting got in the way (E_copy 60→80).

Caveats:
- One model, one task type. The ×16 strength and the "half" threshold were chosen after looking at the data.
- For the 3 of 60 no-change cases where the lamp fired, there is no boosted answer (not run). Worst case: harm up to 5 out of 100.
- In N_same the old examples are followed by 10 fresh ones under the same rule, which helps the lamp stay silent.
- The lamp needs a second model pass (about ×2 compute).
