# Gurdjieff sensor — PREREG (frozen 2026-10-19, before the main run)

Plan: PLAN.md v1 + v2 + v2.1. Code review: review/code-review.md; all 4 blockers fixed:
1. cluster bootstrap added (clusters within population; own K/D clustered by hook, N by set, R by grp; public by run cluster); P1/P2 use the wider of item/cluster intervals; best rival re-chosen in every resample.
2. P2 accuracy: families weighted equally, items pooled inside a family (PLAN v1). P3 stays per population.
3. sanity excludes exact ties (saved top-2 margin < 1e-3), prints median score difference; >= 98% agreement in every family.
4. any missing item = SANITY FAIL; resume rewrites feats.jsonl with whole lines only.
Also: NaN-safe choose/max; hidden states float32 + finite check + layer-count assert; report and scores saved on the volume; numpy/sklearn pinned.
HARM also includes own N/confirm (stricter than the PLAN list).
Random choices: analysis RNG seed 0 (jitter, bootstraps), B = 1000. No new sample is drawn, so no seed from Alex is needed.
Data: items_g.json (4260 items: own 2060 from thinkaloud seed 2, public 2200 from thinkaloud-indep seed 78, redef excluded).
Smoke: report_smoke2.txt (40+40 items, sanity 100% in every family; tiny, so NaN/FAIL is expected).

Commands: modal run --detach modal_g.py --tag main ; then modal run modal_g.py --tag main --analyze

Verdict per split (S1 own->public, S2 public->own): "internal lamp works (at a calibrated 30% share)" only if SANITY, P1, P2, P3 pass (and N1 for S2).
Prediction: P1 fails on both splits.

## sha256
2167a6e6ac266bc7a4d54adc73cd53adf2143ffbd935f3abae53e9dda2e75b66  PLAN.md
190ef39852ad64fefa5ee922b177106bf7c2f4f9678ae0507ac2e5fabe756c8a  make_g.py
9f24f6230e129817a7d0532c38e267c30074cc20089d5a79ced5e50403629e87  modal_g.py
8dbe97eb45a2fe2fdbc25c00fe93ea3b12ec440d09fc006702d5dcea9c0a3c4c  analyze_g.py
7e7bd0350b605e12441722291b77d3ad640de1232a4150739e8ef6a81fc59c51  items_g.json
