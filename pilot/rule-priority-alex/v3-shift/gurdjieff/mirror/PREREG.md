# Mirror test — PREREG (frozen 2026-10-19, before the main run)
Plan: PLAN.md v1 + v2. Reviews: review/plan-review.md (5 blockers, fixed in v2), review/code-review.md (2 blockers fixed:
TEMPLATE now ranks whole cells by flip rate with the boundary cell as an expected value; sums below). Also: verdict only if all 1399 items scored.
Alex's OK: "yes" (2026-10-19 17:54 UTC). Smoke: out_smoke.jsonl (28 items, sanity 100%).
Command: modal run --detach mirror_g.py --tag main ; modal run mirror_g.py --tag main --fetch ; python3 -I score_mirror.py out_main.jsonl
## sha256
d5b1e88ada58abb70ef14fa58b03005352553bcdcd5cafaec0fa06dddc3317af  PLAN.md
e7c7e42065e7fc6b351190c0dc6c0c2f8784de82764e1e8cb166ca80f8c198d6  lines.json
37b194117854014048b5bfb9f88fe4a41d686566a1787bddaac64a78d9f35804  lamp_scores.json
b766aa55613171109fbbdc6d85efcac5253e06040ce0120a5040165c6e59b5df  mirror_g.py
c9cdf858f1eb7848a73da1d19a5ec3f97ce50b1b42cec4030d4fc673dd199664  score_mirror.py
7e7bd0350b605e12441722291b77d3ad640de1232a4150739e8ef6a81fc59c51  ../items_g.json
