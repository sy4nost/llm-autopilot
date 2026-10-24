"""hex-wrap report. python3 -I score_hex.py TAG"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
items = {x["id"]: x for x in json.load(open(os.path.join(HERE, "sets_pi.json")))["items"]}
tag = sys.argv[1] if len(sys.argv) > 1 else "main"; rng = np.random.default_rng(0)
NAMES = {"B": "story, real word", "F": "formula, real word", "B2": "story, made-up word", "F2": "formula, made-up word"}
for mk in ("qwen14", "phi4", "olmo13"):
    fn = os.path.join(HERE, f"out_{tag}_{mk}.json")
    if not os.path.exists(fn):
        continue
    res, hab = {}, {}
    for r in json.loads(open(fn).read()):
        it = items[r["id"]]; a = int(np.argmax(r["s"]))
        res.setdefault(it["style"], {})[it["n"]] = a == it["key"]; hab.setdefault(it["style"], {})[it["n"]] = a == it["habit"]
    print(f"\n{mk}: right (by the condition) / memory answer, out of 100")
    for k in ("B", "F", "B2", "F2"):
        print(f"  {NAMES[k]:24} right {100*np.mean(list(res[k].values())):5.1f}   memory {100*np.mean(list(hab[k].values())):5.1f}")
    for a, b, lab in (("F", "B", "formula - story (real word)"), ("B2", "B", "pull of the real word in story (made-up - real)"),
                      ("F2", "F", "pull of the real word in formula (made-up - real)")):
        ns = sorted(res[a]); d = np.array([res[a][n] - res[b][n] for n in ns], float)
        objs = sorted({n.split(':')[0] for n in ns})
        bs = []
        for _ in range(2000):   # resample objects (6 items each)
            pick = rng.choice(len(objs), len(objs)); bs.append(np.mean([d[i] for o in pick for i, n in enumerate(ns) if n.split(':')[0] == objs[o]]))
        print(f"  {lab:50} {100*d.mean():+5.1f}  [{100*np.percentile(bs,2.5):+.1f}, {100*np.percentile(bs,97.5):+.1f}]")
