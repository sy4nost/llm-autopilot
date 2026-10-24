"""amb-check scoring: share right / share old-rule (habit) answers per model x family x pop x arm, paired diff vs BASE with bootstrap over source items.
  python3 score_amb.py main"""
import json, sys, random, os
HERE = os.path.dirname(os.path.abspath(__file__)); tag = sys.argv[1] if len(sys.argv) > 1 else "main"
items = {i["id"]: i for i in json.load(open(os.path.join(HERE, "sets_amb.json")))["items"]}
ARMS = ["BASE", "REMIND", "REPEAT", "NOHIST"]; NB = 2000


def pct(x):
    return 100 * sum(x) / len(x)


def boot(a, b, seed=0):
    rnd = random.Random(seed); n = len(a); d = []
    for _ in range(NB):
        ix = [rnd.randrange(n) for _ in range(n)]; d.append(sum(a[i] - b[i] for i in ix) * 100 / n)
    d.sort(); return d[int(.025 * NB)], d[int(.975 * NB)]


for mk in ["qwen14", "phi4", "olmo13"]:
    p = os.path.join(HERE, f"out_{tag}_{mk}.json")
    if not os.path.exists(p):
        print(mk, "missing"); continue
    pick = {r["id"]: max(range(len(r["s"])), key=lambda k: r["s"][k]) for r in json.load(open(p))}
    # sanity: BASE picks vs conftrap-lamp DIRECT picks on the same source items
    cl = {}
    for line in open(os.path.join(HERE, "..", "conftrap-lamp", f"out_main_{mk}.jsonl")):
        r = json.loads(line)
        if "res" not in r or "direct" not in r["res"]:
            continue
        s = [sum(x) for x in r["res"]["direct"]]; cl[r["id"]] = s.index(max(s))
    base = [i for i in items.values() if i["arm"] == "BASE" and i["id"] in pick]
    same = sum(pick[i["id"]] == cl.get(i["src"]) for i in base)
    print(f"\n=== {mk}   sanity: BASE pick = conftrap DIRECT pick on {same}/{len(base)}")
    for fam in ["parcels", "bonus"]:
        for pop in ["trap", "control"]:
            by = {a: {} for a in ARMS}
            for i in items.values():
                if i["family"] == fam and i["pop"] == pop and i["id"] in pick:
                    by[i["arm"]][i["src"]] = i
            srcs = sorted(set.intersection(*[set(by[a]) for a in ARMS]))
            R = {a: [int(pick[by[a][s]["id"]] == by[a][s]["key"]) for s in srcs] for a in ARMS}
            H = {a: [int(by[a][s]["habit"] is not None and pick[by[a][s]["id"]] == by[a][s]["habit"]) for s in srcs] for a in ARMS}
            line = f"  {fam:7s} {pop:7s} n={len(srcs):3d} "
            for a in ARMS:
                line += f"| {a}: right {pct(R[a]):5.1f}" + (f" old {pct(H[a]):5.1f} " if pop == "trap" else " ")
            print(line)
            for a in ARMS[1:]:
                lo, hi = boot(R[a], R["BASE"]); print(f"      {a}-BASE right {pct(R[a]) - pct(R['BASE']):+6.1f} [{lo:+.1f},{hi:+.1f}]")
