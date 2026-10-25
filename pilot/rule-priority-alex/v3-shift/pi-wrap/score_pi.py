"""pi-wrap report. python3 -I score_pi.py TAG. Share answering 9 (stated value) per style and model; habit pull = control minus pi."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
items = {x["id"]: x for x in json.load(open(os.path.join(HERE, "sets_pi.json")))["items"]}
tag = sys.argv[1] if len(sys.argv) > 1 else "main"
rng = np.random.default_rng(0)
NAMES = {"F": "formula 'Let pi = 974'", "F2": "formula, x instead of pi", "M": "book, pi, digits", "W": "book, pi, words", "W2": "book, zorp, words"}
for mk in ("qwen14", "phi4", "olmo13"):
    fn = os.path.join(HERE, f"out_{tag}_{mk}.json")
    if not os.path.exists(fn):
        continue
    R = json.loads(open(fn).read()); res = {}
    for r in R:
        it = items[r["id"]]; res.setdefault(it["style"], {})[it["n"]] = int(np.argmax(r["s"])) == it["key"]
    print(f"\n{mk}: answered 9 (by the condition) out of 100")
    for k in ("F", "F2", "M", "W", "W2"):
        if k in res:
            print(f"  {NAMES[k]:28} {100*np.mean(list(res[k].values())):5.1f}")
    for a, b, lab in (("F2", "F", "pull of 'pi' in formula style (x - pi)"), ("W2", "W", "pull of 'pi' in words style (zorp - pi)")):
        if a in res and b in res:
            ns = sorted(set(res[a]) & set(res[b])); d = np.array([res[a][n] - res[b][n] for n in ns], float)
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            print(f"  {lab:42} {100*d.mean():+5.1f}  [{100*np.percentile(bs,2.5):+.1f}, {100*np.percentile(bs,97.5):+.1f}]")
