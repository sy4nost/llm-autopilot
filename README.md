# llm-autopilot

**When a language model is told the rule changed but still answers the old way.** This repository holds experiments on in-context habits ("autopilot") in open-weight language models (mainly Qwen2.5-14B-Instruct): how strong they are, where they live inside the model, and how to detect the moment a habitual answer should be re-checked.

Author: Oleksandr Rylov (independent researcher). The code, runs and draft reports were produced with Claude (Anthropic) as a research assistant. The research questions, directions and random seeds for the frozen tests came from the author.

**Start here: [REPORT.md](REPORT.md)**, a summary in English with all key numbers, limits and negative results.

## Map
| folder | what |
|---|---|
| `pilot/rule-priority-alex/v3-shift/` | main line: rule-shift habit, exact-copy test (`match/`), copy heads (`heads/`, `blockex/`), lamp and gate (`lamp/`, `gate/`, `newtasks*/`, `watch/`), independent audit (`audit/`), public benchmarks (`public/`, `thinkaloud-indep/`, `fastthink/`), confident traps in 3 models and their audit (`confident-traps/`, `conftrap-lamp/`), negative lines (`general-lamp/`, `disguise/`, `cycle/`, `gurdjieff/`, …) |
| `pilot/rule-priority-alex/`, `v2/` | earlier rule-priority tests on Qwen 0.5B–14B |
| `pilot/fu-training/` | preference (DPO) training against blind copying, a closed negative line |
| `pilot/autopilot-bench/`, `pilot/odd-phrase/`, `pilot/v2/` | first pilots |
| `notes/` | literature notes |

Per-experiment reports are named `rezultaty*.md` ("results"). All text in the repository is in English; the reports were originally written in Russian and translated on 2026-10-23 (the original versions are in the git history, mainly commit 6d3f172; PREREG checksums of plan files refer to those originals). Two figures in `gurdjieff/viz/` still have Russian axis labels. Each pre-registered test has a `PREREG.md` with sha256 hashes of the code and the data, fixed before the run. Scoring scripts (`score_*.py`) recompute every number from the raw outputs (`*.jsonl`). Runs used Modal GPUs (`modal_*.py`).

Public benchmark files in `data/` folders (MMLU, TruthfulQA, HANS, Inverse Scaling Prize tasks) belong to their original authors under their own licenses.
