# PREREG — think-aloud on independent tasks (2026-10-18)
Frozen before the main run. Plan: PLAN.md v1 + v2 + v2.1. Reviews: review/plan-review.md, review/code-review.md (blockers fixed).
Model Qwen2.5-14B-Instruct bf16 greedy, H100. Seed = Alex's number (not 12345 = smoke).
Run: python3 -I make_ti.py SEED ; modal run modal_ti.py --sets sets_ti.json --tag main ; python3 -I score_ti.py out_main.jsonl
Criteria: as in score_ti.py (C1 THINK, C2 THINKI vs 3 DIRECT variants >= +5 low > 0 and >= 3 of 5 families; C3 controls >= -5 low >= -10
every family >= -10; MMLU >= -2 low >= -5). Smoke (seed 12345, 72 items, out_smoke.jsonl): pipeline works; parse rate 67-100%.
Prediction (mine, before the run): THINKI helps memo-trap and maybe NeQA, hurts hindsight (smoke: "follow the facts" pushes it to
judge by outcome); no clear verdict expected.

sha256:
001cb8f37a3049c81be66c4fda5d60b38737e8fed10697439edec629549570bd make_ti.py
45c41abfa7ecd77e486294b5305fdc698547d733e9f816c8ac7af09d797695d6 lin.py
de750f1ef1c48c20f794735f8ceb830f95d3d9c1ce583aad665d39a3d79e9366 modal_ti.py
1a4b2ac2607ceb54605114205328d40913c95282b69891f5452ac3699cf4bfed score_ti.py
seed 78 (Alex, 2026-10-18 14:13Z); sets_ti.json sha256 7e0db239fcc32e9567286977c8b99a15dceebf101333c5b46042f7f6a52dddfe

## Amendment 2026-10-18 ~15:00Z (before any result was seen)
First main run died near the end of the THINKI pass: Modal "workspace is disabled" (account limit). Nothing was saved (out_main.jsonl empty).
modal_ti.py changed for robustness only: generations streamed to gen_main.jsonl per batch, scored items appended, a rerun skips what is saved.
Prompts, scoring, items (seed 78, same sets_ti.json) and score_ti.py unchanged. Old version kept as modal_ti_v1.py.
Resumed batches may differ slightly from one full pass (bf16 drift, as in the smoke batch check).
e462cfbeb91e1549d81b0b1502a3958db5382dd7d8b429cb29652dee3e32e019 modal_ti.py

## Amendment 2 2026-10-19 ~01:30Z (no result seen yet)
Second attempt lost too: the cloud container restarted mid-run; locally saved generations were lost (shared-folder writes not yet uploaded)
and the client-less Modal app was stopped by me (its output had nowhere to go; it was at THINKI 1632/2300).
modal_ti.py now keeps its state on the Modal volume (/cache/ti_runs/main/, committed every batch), is launched with --detach and
fetched with --get. Tested end to end on 8 smoke items (tag t8). Prompts, items, scoring unchanged.
37261916feb2a7eb76948ce65aeef5c74561f6aed89c5584bbb63baf2a9c14e5 modal_ti.py
