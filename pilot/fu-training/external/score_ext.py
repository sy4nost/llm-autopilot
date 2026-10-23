"""python3 -I score_ext.py ext_<model>.jsonl : accuracy per task for base / v1 / v2, paired bootstrap vs base."""
import json, random, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
rng = random.Random(1)
mean = lambda v: sum(v) / len(v) if v else float("nan")
ok = lambda r: (max(range(len(r["scores"])), key=lambda i: r["scores"][i]) == r["gold"]) if "scores" in r else (r["pred"] == r["gold"])
tasks = list(dict.fromkeys(r["task"] for r in rows))
models = list(dict.fromkeys(r["model"] for r in rows))
R = {(r["model"], r["task"], r["i"]): ok(r) for r in rows}
print(f"{'task':26} {'n':>4} " + " ".join(f"{m:>6}" for m in models) + "   v1-base / v2-base [95%]")
for t in tasks:
    ids = sorted(i for (m, tt, i) in R if m == "base" and tt == t)
    line = f"{t:26} {len(ids):4} " + " ".join(f"{mean([R[(m, t, i)] for i in ids if (m, t, i) in R]):6.1%}" for m in models)
    for m in models[1:]:
        d = [R[(m, t, i)] - R[("base", t, i)] for i in ids if (m, t, i) in R]
        if d:
            b = sorted(mean([rng.choice(d) for _ in d]) for _ in range(4000))
            line += f"   {m} {mean(d):+.1%} [{b[100]:+.1%}, {b[3899]:+.1%}]"
    print(line)
