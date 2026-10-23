# PREREG — fast vs thought answers on public tests, 3 models (2026-10-19)
Frozen before the main run. Plan: PLAN.md (v1 + v2 + code-review fixes). Reviews: review/plan-review.md, review/code-review.md (blockers fixed).
Smoke (seed 12345, 42 items): smoke + smoke2 runs, pipeline checks only (no accuracy looked at); truncation 0% -> MAXNEW 512.
Run: python3 -I make_ft.py SEED ; then per model (launches ~30 s apart):
  modal run --detach modal_ft.py --sets sets_ft.json --tag main --models qwen14 | phi4 | olmo13
  modal run modal_ft.py --tag main --get ; python3 -I score_ft.py main
Verdicts as in score_ft.py (H1 lamp vs RANDOM/CONF/SAMPLE >= +10 low > 0; H2 >= +5 low > 0; harm; readout sensitivities same sign).

sha256:
cadfa201c960f5c84ff374ef8d09fc60a4db9a1d121f983381779a0190e60f26  make_ft.py
71485f524d07b645aec1e5d16d7524567928db9843591a0b8d2d63297f224498  modal_ft.py
42cf47047fb32c9597b96a0432b3b1072973f2990be3c6cf66b315ddfabadb03  score_ft.py
001cb8f37a3049c81be66c4fda5d60b38737e8fed10697439edec629549570bd  thinkaloud-indep/make_ti.py
45c41abfa7ecd77e486294b5305fdc698547d733e9f816c8ac7af09d797695d6  thinkaloud-indep/lin.py
seed 58 (Alex, 2026-10-19 20:56Z); sets_ft.json sha256 4cb480bbc29ac0b2c1163ab968df59dd8cd596a6402e9cb3f822c817a26258a6
