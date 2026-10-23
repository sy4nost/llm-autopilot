# llm-autopilot

**When a language model is told the rule changed but still answers the old way.** This repository holds experiments on in-context habits ("autopilot") in open-weight language models (mainly Qwen2.5-14B-Instruct): how strong they are, where they live inside the model, and how to detect the moment a habitual answer should be re-checked.

Author: Oleksandr Rylov (independent researcher). The code, runs and draft reports were produced with Claude (Anthropic) as a research assistant. The research questions, directions and random seeds for the frozen tests came from the author.

**Start here: [REPORT.md](REPORT.md)**, a summary in English with all key numbers, limits and negative results.

## Map
| folder | what |
|---|---|
| `pilot/rule-priority-alex/v3-shift/` | main line: rule-shift habit, exact-copy test (`match/`), copy heads (`heads/`, `blockex/`), lamp and gate (`lamp/`, `gate/`, `newtasks*/`, `watch/`), independent audit (`audit/`), public benchmarks (`public/`, `thinkaloud-indep/`, `fastthink/`), negative lines (`general-lamp/`, `disguise/`, `cycle/`, `gurdjieff/`, …) |
| `pilot/rule-priority-alex/`, `v2/` | earlier rule-priority tests on Qwen 0.5B–14B |
| `pilot/fu-training/` | preference (DPO) training against blind copying, a closed negative line |
| `pilot/autopilot-bench/`, `pilot/odd-phrase/`, `pilot/v2/` | first pilots |
| `notes/` | literature notes (in Russian) |

Most per-experiment reports are named `rezultaty*.md` and are written in Russian. Each pre-registered test has a `PREREG.md` with sha256 hashes of the code and the data, fixed before the run. Scoring scripts (`score_*.py`) recompute every number from the raw outputs (`*.jsonl`). Runs used Modal GPUs (`modal_*.py`).

Public benchmark files in `data/` folders (MMLU, TruthfulQA, HANS, Inverse Scaling Prize tasks) belong to their original authors under their own licenses.
