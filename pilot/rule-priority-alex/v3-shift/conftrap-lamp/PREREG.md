# PREREG — lamp on confident traps, 3 models (2026-10-23)
Frozen before the main run. Plan: PLAN.md (v1 + v2 + cost note). Review: review/plan-code-review.md (blocker B1 and S1-S6 fixed).
Smoke: seed 12345 (sets_smoke.json, 32 items), runs smoke + smoke2, pipeline checks only (no accuracy, no lamp numbers looked at).
Run: python3 -I make_cl.py SEED ; then per model (launches ~35 s apart):
  modal run --detach modal_cl.py --sets sets_cl.json --tag main --models qwen14 | phi4 | olmo13
  modal run modal_cl.py --tag main --get ; python3 -I score_cl.py main
Verdict as printed by score_cl.py: H1 (FLAG beats RANDOM, CONF and SAMPLE by >= +10 recall points, low > 0, > 0 in >= 2 of 3 models)
AND readout signs hold AND leave-one-family-out holds.

sha256 (scripts):
97208701ee77ac88943700ceb63bc0a79b309a72d6361c98366e4b23891c1124  make_cl.py
d414912556b3276d4078b8d6dfafdcad71c9c7147ac4ce94648e96fbcf39dfd5  modal_cl.py
9b404453ea41b154bd1d8682ab3601eab7ebee538c75002de245e3ae4daa81f4  score_cl.py
18434b01b1677e1104cbfd3db9e2ffe54ccf77659670e745b306efd3482d2c57  ../confident-traps/make_ct.py
00f21808d86b01dc51828060b2ad5dc95f89e659a5de41c775a8463e589ebd5e  ../thinkaloud/make_ta.py
d61c64eab044d423b86d470ade67d3e9e019a5d622456d27d4c4a96aa3942c40  ../newtasks2/make_new2.py
001cb8f37a3049c81be66c4fda5d60b38737e8fed10697439edec629549570bd  ../thinkaloud-indep/make_ti.py
00e5bc837f95174c8c15eacd5c42ed888892cff8d9cd06512dab2a698582b06f  ../disguise/make_dis.py
b2d7562c0f8b4bb3c45c49e4caf3571d57882deb6828cb6affb49b8e94cba35a  sets_smoke.json
seed 57 (Alex, 2026-10-23 22:38Z); sets_cl.json sha256 bd89b2df496a329a1b71d0080ec0402f527aa020ccfc01dc522a9da51e25d1b6
