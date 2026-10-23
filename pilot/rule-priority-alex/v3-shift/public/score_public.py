"""python3 -I score_public.py public_<model>.jsonl : accuracy normal vs heads off, paired bootstrap, and what the heads change."""
import json, random, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
rng = random.Random(1)
am = lambda v: max(range(len(v)), key=lambda k: v[k])
def boot(d):
    n = len(d); bs = sorted(sum(rng.choice(d) for _ in range(n)) / n for _ in range(2000))
    return sum(d) / n, bs[50], bs[1949]
print(f"{'test':26} {'n':>4} {'normal':>7} {'off_all':>8} {'off_ans':>8}   off_all-normal [95%]      answers changed by off_all: to right / to wrong")
for t in dict.fromkeys(r["task"] for r in rows if "normal" in r):
    R = [r for r in rows if r["task"] == t]
    ok = {m: [am(r[m]) == r["gold"] for r in R] for m in ("normal", "off_all", "off_ans")}
    m, lo, hi = boot([a - b for a, b in zip(ok["off_all"], ok["normal"])])
    ch = [(am(r["normal"]) != am(r["off_all"]), am(r["off_all"]) == r["gold"]) for r in R]
    print(f"{t:26} {len(R):4} {sum(ok['normal'])/len(R):7.1%} {sum(ok['off_all'])/len(R):8.1%} {sum(ok['off_ans'])/len(R):8.1%}   {m:+.1%} [{lo:+.1%}, {hi:+.1%}]   {sum(c and g for c, g in ch)} / {sum(c and not g for c, g in ch)}")
for t in dict.fromkeys(r["task"] for r in rows if "mode" in r):
    P = {}
    for r in rows:
        if r.get("task") == t:
            P.setdefault(r["i"], {})[r["mode"]] = r["pred"] == r["gold"]
    ids = [i for i in P if len(P[i]) == 2]
    m, lo, hi = boot([P[i]["off_all"] - P[i]["normal"] for i in ids])
    print(f"{t:26} {len(ids):4} {sum(P[i]['normal'] for i in ids)/len(ids):7.1%} {sum(P[i]['off_all'] for i in ids)/len(ids):8.1%} {'':>8}   {m:+.1%} [{lo:+.1%}, {hi:+.1%}]")
