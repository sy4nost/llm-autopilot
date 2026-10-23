# The model saw the rule change and answered the old way anyway: a pre-registered study of in-context habits in Qwen2.5-14B

*Oleksandr Rylov, independent researcher (Spartanburg, SC, USA). October 2026. Draft v2 (updated October 23 with the confident-trap tests and their audit).*

*Code and data: https://github.com/sy4nost/llm-autopilot. All experiments were run with Claude (Anthropic) as a research assistant: Claude wrote the code, ran the jobs and drafted the reports. I designed the questions, chose the directions and supplied the random seeds for every frozen test.*

---

## TL;DR

- **The habit is real and large.** Qwen2.5-14B-Instruct learns a two-condition rule from solved examples. Then it is told explicitly, in one line, that the rule changed. After 60 examples it still gives the old-rule answer in **82 of 100** changed cases. With no examples the figure is 26 of 100 (difference +56, 95% CI [+42, +70], 30 randomized task sets).
- **Most of the habit is copying, not strategy.** If the exact same case sits in the history with its old answer, the model repeats it **93 of 100** times. Without an identical case, the rate is 38, the same as with no history at all (37).
- **The copying runs through a small set of attention heads.** Ranking was done on half the task sets and testing on the other, held-out half. Zero-ablating 20 of 1,920 heads (layers 29 to 38) halves old-rule answers (43 → 21 of 100). 20 random heads do nothing (43 → 42). Masking the best 40 heads from the past-examples block only raises overall task accuracy from 63 to 73 of 100, with **no cost** on MMLU (+0.5) or GSM8K (0.0).
- **It transfers only partly.** A frozen detector-plus-boost recipe was tested on unfamiliar tasks with a fresh seed. It fixed many habit errors (+57 and +28 points on two task types) but missed a pre-registered harm threshold on one of them (−7 vs −5). An independent audit found that a trivial baseline, simply dropping the history, beats it on our own tasks.
- **On independent public trap benchmarks, one thing worked, pre-registered:** asking the model to reason before answering gave **+15.8 points** [+12.5, +19.0] across five public trap families, with no harm on control items (−0.3) or MMLU (+1.8). Think-aloud on traps is known in the literature.
- **Confident habit errors are common, and confidence does not see them.** A pre-registered 3-model test (Qwen2.5-14B, Phi-4, OLMo-2-13B, 5 trap families) found fast answers that are confidently wrong by habit in all three models; low-confidence gating misses them, and reasoning fixes about **75 of 100**. Comparing the fast and the reasoned answer flags these errors, but only *after* reasoning, so an independent audit concluded it adds nothing over always reasoning. **We do not yet have a signal that predicts habit errors before reasoning; that is the open problem.**
- **We report more than ten negative results**, including: no general internal "habit signal" across habit types, preference training that does not transfer, and "wake-up" prompts that are placebos or harmful.

---

## 1. The question

People act on autopilot. A driver takes the usual turn after the road was rerouted; a trader keeps a strategy after the market regime changed. The knowledge is fine. The failure is that a well-practiced response fires without a check of whether it still applies.

I ran a human study on this before (Impulse Lab: a game that measures when people notice that a rule changed). This project asks the same question of language models:

> When the condition has changed and the model has been told, does it still answer from experience? If so, where in the model does the pull come from, and can we detect the moment it should re-check?

This matters for reliability and safety. Agents increasingly act on long histories of their own past actions and examples. An agent that "knows" the rule changed but still acts on the old pattern is a quiet and confident failure.

## 2. Set-up and discipline

- **Model:** Qwen2.5-14B-Instruct (bf16), plus Qwen 0.5B to 7B, Phi-4 and OLMo-2-13B in some tests. Runs were on Modal (H100 / L40S). The total compute spent across all experiments is roughly **$60** (estimate).
- **Tasks:** synthetic rule tasks with random names (HOLD / PASS / REVIEW labels from field values), one case per prompt. Every answer key was verified by an independent script.
- **Statistics:** paired cluster bootstrap over task sets, 95% intervals.
- **Pre-registration:** from the middle of the project on, every decisive test had its plan, code hashes (sha256) and pass/fail thresholds frozen in a PREREG file before data generation. The random seed came from me after freezing. Independent reviewer sessions read the plan and code beforehand. One full independent audit recomputed the numbers from raw outputs.
- **What we count as a result:** only a pass against thresholds set in advance. Everything else is labeled "report only" or "post hoc".

## 3. The phenomenon: an announced rule change does not beat a habit

**Old rule:** HOLD if seal=CLOSED; HOLD if waiver=YES **and echo=ON**; PASS if tag=BLUE; otherwise REVIEW. The model sees 0, 20 or 60 solved examples. Then one line says: "IMPORTANT UPDATE: Rule 2 has been changed… waiver=YES and echo=OFF". One word changes, and the change is stated openly.

| | 0 examples | 20 | 60 |
|---|---|---|---|
| Old-rule answers where the new rule **removes** a HOLD | 26 / 100 | 74 | **82** |
| Old-rule answers where the new rule **adds** a HOLD | 14 | 8 | 11 |

The model **adds** the new rule fine, but it does not **drop** the old one. Without the change line (old rule still valid) it is correct 97 to 100 of 100 times. So this is not a failure to learn.

**Is it strategy inertia or copying?** We kept the history identical except for the few examples that used the old Rule 2:

| history (60 examples) | old-rule answers |
|---|---|
| A: the exact test case appears in history with its old answer | **93 / 100** |
| B: same rule path, different case | 38 |
| C: no example of the old rule path | 12 |
| no history | 37 |

A − B = +55 [+42, +68]. B − no history = 0 [−20, +20]. The effect is almost entirely **copying a ready answer** from a matching past case.

**Boundary.** In a later frozen test on new task types, simple explicit changes (a word now means something else; points are now worth more) produced **no habit at all**: 98 to 100 of 100 correct despite 7 to 25 exact old copies in context. The habit appears when the change **amends a compound condition**. A second frozen run with compound amendments confirmed it, with old answers at 80 to 90 of 100.

## 4. Where it lives: a small set of copy heads

For each of 1,920 attention heads we measured attention from the answer position to the label of the identical past example. Heads were ranked on task sets 1 to 15 only and tested on held-out sets 16 to 30.

| intervention (held-out) | old answers after change | correct, changed cases | correct, no-change cases |
|---|---|---|---|
| none | 43 / 100 | 51 | 57 |
| **20 top heads zeroed** (layers 29–38) | **21** | 61 | 57 |
| 20 random heads zeroed | 42 | 47 | 49 |

Paired difference for old answers is −22 [−30, −14]. The top heads sit exactly where a logit-lens analysis shows the old answer overtaking the new one (after layer 32).

**Cost and a free version.** Zeroing these heads costs GSM8K −5.6 points, because the heads also read the model's own reasoning steps. Masking them **only from the past-examples block** removes that cost:

| | overall task accuracy | MMLU (570) | GSM8K (250) |
|---|---|---|---|
| none | 63 | 77.5 | 93.6 |
| 40 heads masked from past examples | **73** (+10 [+6, +14]) | 78.1 | 93.6 |

**Limits we measured.** About 31 of 100 old answers remain even when all 760 late-layer heads are masked. Masking early layers (20–28) removes the copy but also removes useful learning from examples. Roughly half of the removed old answers move to a third, wrong label rather than to the correct one. A brake alone is not enough; the right answer also needs a push.

## 5. Detecting the moment to re-check

**Internal "lamp" on our tasks.** A small probe on internal states detects habit errors (held-out AUC 0.91). Its main feature is whether the answer survives when the copy heads are quieted. A frozen lamp-then-boost recipe was tested on **unfamiliar task types with a new seed**:

| task | correct on traps, normal → with lamp | harm where experience is right |
|---|---|---|
| parcels | 20 → 78 (A60); gain +57 [+46, +69] | 0 |
| bonus | 5 → 30; gain +28 [+18, +39] | **−7** [−10, −5] (threshold −5) → **FAIL** |

**Independent audit.** A separate session recomputed every number from raw outputs and found no leakage. It raised two problems we accept: (1) a trivial baseline, re-asking with no history at all, does as well or better on these tasks, because our tasks state the rules fully, so history adds nothing; (2) the heads were found on a task family structurally close to "parcels". **Our internal interventions are therefore evidence about mechanism, not yet a useful method.**

**Is there one general habit signal?** No. A probe trained on context-copy habits detected habits from pretraining (redefined π, negated questions) at **AUC 0.55**, no better than output-only baselines. It fired on 55 of 100 ordinary MMLU questions. Even across two context-copy families it reached only 0.64. Habit errors from pretraining are **confident**: plain confidence reached 0.89 on one twin task where the probe reached 0.59.

## 6. Independent public benchmarks: think before answering

**Pre-registered test** (seed 78, Qwen2.5-14B). Five public trap families: memo-trap, NeQA, hindsight-neglect, into-the-unknown and repetitive-algebra from the Inverse Scaling Prize. There were also control items where the habitual answer is right, plus 1,000 MMLU questions.

| | direct answer | "follow the stated facts" | **reason first, then answer** |
|---|---|---|---|
| memo-trap (120) | 83 | 84 | **98** |
| NeQA (120) | 73 | 76 | 78 |
| hindsight-neglect (120) | 36 | 20 | **85** |
| into-the-unknown (100) | 53 | 52 | **63** |
| repetitive-algebra (120) | 100 | 100 | 100 |
| controls (habit is right) | 98 | 98 | 98 |
| MMLU (1000) | 78 | 75 | 80 |

- Reason-first vs direct: **+15.8 [+12.5, +19.0]** → PASS. Gains appear in 4 of 5 families; harm on controls −0.3 → PASS; MMLU +1.8 → PASS.
- Hindsight carries much of the gain. Post hoc, without hindsight, the gain is still +7.5 [+4.6, +10.5]. On hindsight it is not a one-direction shift: both answer classes improve (20 → 98 and 52 → 72).
- An instruction alone ("follow the stated facts") **hurts** (−2.5 overall, hindsight −16): the model takes the outcome as "the fact".

**The detector: fast ≠ reasoned.** Where the fast answer disagrees with the reasoned one, the fast answer is likely a habit error: AUC **0.97** on memo-trap, **0.90** on hindsight, 0.65 on NeQA and 0.61 on into-the-unknown (report only).

**Second frozen test, 3 models (Qwen 14B, Phi-4, OLMo-2-13B), new benchmarks: FAIL.** The verdict benchmarks were TruthfulQA MC1, HANS and an epistemic-reasoning task. They turned out to have little confident autopilot: fast answers were right 59 to 98 of 100. There the disagreement flag beat random and 5-sample self-consistency, but not plain low confidence (+2 [−2.7, +6.6]). Reasoning gave +3.2 against a +5 threshold and hurt HANS controls for two models. On tasks **with** confident habit errors (fresh memo-trap and hindsight items; replication, not counted in the verdict), it worked on all three models: reasoning +38, and the flag beat low confidence by +36 in errors caught.

**What we concluded at that point:** the fast-vs-reasoned disagreement looked valuable where habit errors are **confident**, which is exactly where confidence-based gates fail. Section 6b tests this and shows its limit.

## 6b. Confident habit errors in three models, and an audit of the "fast ≠ reasoned" flag (October 23)

**Pilot (not frozen, ~$0.5).** 517 fresh items in 9 trap families, fast answers only, Qwen2.5-14B, Phi-4, OLMo-2-13B. In 4 families (hindsight, redefine-math, our redefined π/e, parcels) all three models make habit errors with high confidence (typically 80–100 of 100), while control items where the habit is right are solved at 95–100 of 100.

**Pre-registered test (seed 57, ~$5).** 760 fresh items, 5 families, the same 3 models. Reasoning fixes most of the confident errors (traps correct, fast → reasoned): hindsight 43 → 89, 1 → 84, 7 → 53; parcels 26 → 100, 58 → 98, 0 → 43; bonus 4 → 100, 36 → 99, 0 → 49 (Qwen, Phi-4, OLMo). At equal flag rate, "fast ≠ reasoned" caught more fast-answer errors than random (+37 [+31, +44]), low confidence (+26 [+20, +31]) and 5-sample disagreement (+30 [+24, +36]): PASS by the frozen rules.

**Independent audit (`pilot/rule-priority-alex/v3-shift/conftrap-lamp/audit/`).** The numbers and the pre-registration checked out. But the pass was close to certain by design: on these families the flag is almost the same as "reasoning fixed the error" (74.5 of its 75.1 recall points), it ties a rival that also sees the reasoned answer, it adds nothing over simply always reasoning, and it catches only 2.7% of the errors that reasoning cannot fix (low confidence catches 34.8% of those).

**What we conclude now:** (1) confident habit errors are real and widespread across model families; (2) confidence-based gating is blind to them; (3) reasoning fixes about 75 of 100. A flag that needs the reasoned answer is not a detector. **The open problem is a cheap signal available before reasoning.** Our internal probes reach high AUC within one task kind (e.g. 0.98 on definitions, 0.87 on π traps, 0.91 on the rule-change task) but do not transfer across kinds (section 5).

## 7. Negative results (all reported in full in the repository)

| idea | outcome |
|---|---|
| DPO "don't blindly copy" training, 7B, 8 trap types | Strong in our formats (old answers 86 → 0). Failed its pre-registered threshold on public tests: hindsight −29, GSM-Symbolic −11. The skill stays in our formats. |
| Copy heads off on public traps | No gain (−2.5 to +1.5). Public traps pull from pretrained memory, not from in-context copying. |
| Context-memory conflict probe (replicating Zhao et al. 2024) | Perfect on our facts, but fired on 100% of foreign prompts including MMLU: it learned the format. |
| General habit lamp across habit kinds | AUC 0.55, not better than baselines (section 5). |
| "Shake" the model (alarm token, wake-up line, residual noise) | Placebo; plain instruction beat it. |
| "You will die after this cycle" framing | Harmful: traps −8.6 vs a length-matched placebo. |
| Telling the model "your internal sensor detects a conflict" | Hurt traps (63 → 55); the same hint without the sensor framing helped (+19). |
| Linear probe predicting where reasoning will change the answer | AUC 0.67 vs 0.65 for low confidence; failed its threshold. |
| "Fast ≠ reasoned" flag as a habit detector (3 models, confident traps) | Passed its frozen test, but the audit showed it only restates that reasoning fixed the error; no gain over always reasoning (section 6b). |

## 8. What we think is going on (interpretation, partly untested)

1. There are at least two habits: **copying from context** (in-context, through identifiable heads, removable at the mechanism level) and **pull from pretraining** (confident, not carried by those heads). Different tools are needed for each.
2. The model can tell that "something changed" (it applies new additions correctly). What it lacks is a step that **re-checks an already available answer**. The disagreement between a fast and a reasoned pass acts as an external version of that step.
3. In humans, conflict monitoring (anterior cingulate) and a prefrontal brake handle this. The engineering goal is a cheap monitor that triggers the expensive check only when needed. It should not be an always-on "think more" rule, which costs compute and sometimes hurts (HANS controls).

## 9. Next steps (proposal)

1. **A benchmark of confident autopilot.** Collect public and new items where fast answers are confidently wrong because of habit: changed rules in agent histories, modified famous texts, outcome-vs-decision. Verify the habit exists on a small pilot before any paid run. Release it openly.
2. **A cheap pre-reasoning signal**, tested pre-registered on that benchmark across 4 to 5 open models and 2 API models (Claude, GPT): internal probes, short self-checks, or other signals available before reasoning. Baselines: always reason, low confidence, self-consistency, and the after-the-fact fast-vs-reasoned flag. Measure errors caught per unit of extra compute.
3. **An agent setting.** A simple trading or operations agent whose environment changes regime mid-episode. Measure habit errors with and without the gate.
4. **Mechanism follow-up on open models.** Check whether the copy-head picture (sections 3–4) replicates in Llama and Gemma families.

---

*Correspondence: 3075549@gmail.com. All numbers above are from the reports in the repository (`pilot/rule-priority-alex/…`), each with its raw outputs and scoring script.*
