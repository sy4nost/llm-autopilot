"""python3 -I score_boost.py boost_<model>.jsonl : right / old answers per arm x probe group per setting,
paired bootstrap over sets vs normal (and the _rep prompt arms vs their plain arm, normal setting)."""
import json, random, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
print("check: up1_late vs normal max diff", rows[0]["checks"]["max_up1_diff"], "(must be 0)")
rows = rows[1:]
S = list(rows[0]["scores"])
pick = lambda r, s: r["labels"][max(range(3), key=lambda i: r["scores"][s][i])]
mean = lambda v: sum(v) / len(v) if v else float("nan")
arms = list(dict.fromkeys(r["arm"] for r in rows))
G = {f"{a} {p}": (lambda r, a=a, p=p: r["arm"] == a and r["probe"] in p) for a in arms for p in (("D1", "D3"), ("D2", "D4"))}
G = {k.replace("('D1', 'D3')", "D1+D3 trap").replace("('D2', 'D4')", "D2+D4"): v for k, v in G.items()}
print("\nRIGHT answers")
print(f"{'group':24} {'n':>3} " + " ".join(f"{s:>13}" for s in S))
for g, f in G.items():
    R = [r for r in rows if f(r)]
    print(f"{g:24} {len(R):3} " + " ".join(f"{mean([pick(r, s) == r['key'] for r in R]):13.0%}" for s in S))
print("\nOLD-rule answers (D1+D3)")
for g, f in G.items():
    R = [r for r in rows if f(r) and r["old"] != r["key"] and "D1" in g]
    if R:
        print(f"{g:24} {len(R):3} " + " ".join(f"{mean([pick(r, s) == r['old'] for r in R]):13.0%}" for s in S))
rng = random.Random(1)
def boot(d):
    sets = sorted(d); bs = sorted(mean([x for k in [rng.choice(sets) for _ in sets] for x in d[k]]) for _ in range(2000))
    return mean([x for v in d.values() for x in v]), bs[50], bs[1949]
print("\nvs normal, right answers, paired bootstrap over sets (95%)")
for s in S[2:]:
    for g, f in G.items():
        R = [r for r in rows if f(r)]
        d = {}
        for r in R:
            d.setdefault(r["set"], []).append((pick(r, s) == r["key"]) - (pick(r, "normal") == r["key"]))
        m, lo, hi = boot(d)
        print(f"  {s:13} {g:24} {m:+.0%} [{lo:+.0%}, {hi:+.0%}]")
