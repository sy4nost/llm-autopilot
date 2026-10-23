"""python3 -I score_eras.py eras_<model>.jsonl : per arm x probe group, right / old answers for each setting,
paired bootstrap over sets for the formula (w0.1, w0) vs normal."""
import json, random, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
print("checks:", json.dumps(rows[0]["checks"]["max_w1_diff"]), "(w1 vs normal, must be 0)")
rows = rows[1:]
try:  # automated check: A60 with w0 must reproduce blockex2 'b40' (same prompts, same heads, same block)
    import os
    bx = {(x["set"], x["probe"]): x["scores"]["b40"] for x in map(json.loads, open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "..", "blockex", "blockex2_Qwen2.5-14B-Instruct.jsonl")))
          if x.get("bench") == "ours" and x["arm"] == "A_change_60"}
    dd = [max(abs(a - b) for a, b in zip(r["scores"]["w0"], bx[(r["set"], r["probe"])])) for r in rows if r["arm"] == "A60"]
    print(f"A60 w0 vs blockex2 b40: {len(dd)} prompts, max diff {max(dd):.3f}, same pick {sum(1 for r in rows if r['arm']=='A60' and max(range(3),key=lambda i:r['scores']['w0'][i])==max(range(3),key=lambda i:bx[(r['set'],r['probe'])][i]))}/{len(dd)}")
except Exception as ex:
    print("blockex2 comparison skipped:", ex)
S = list(rows[0]["scores"])
pick = lambda r, s: r["labels"][max(range(3), key=lambda i: r["scores"][s][i])]
mean = lambda v: sum(v) / len(v) if v else float("nan")
G = {"A60 D1+D3 (trap)": lambda r: r["arm"] == "A60" and r["probe"] in ("D1", "D3"),
     "A60 D2+D4": lambda r: r["arm"] == "A60" and r["probe"] in ("D2", "D4"),
     "E_copy D1+D3": lambda r: r["arm"] == "E_copy" and r["probe"] in ("D1", "D3"),
     "E_copy D2+D4": lambda r: r["arm"] == "E_copy" and r["probe"] in ("D2", "D4"),
     "E_nocopy D1+D3": lambda r: r["arm"] == "E_nocopy" and r["probe"] in ("D1", "D3"),
     "E_nocopy D2+D4": lambda r: r["arm"] == "E_nocopy" and r["probe"] in ("D2", "D4"),
     "N_same (no change)": lambda r: r["arm"] == "N_same"}
print("\nRIGHT answers")
print(f"{'group':22} {'n':>3} " + " ".join(f"{s:>7}" for s in S))
for g, f in G.items():
    R = [r for r in rows if f(r)]
    print(f"{g:22} {len(R):3} " + " ".join(f"{mean([pick(r, s) == r['key'] for r in R]):7.0%}" for s in S))
print("\nOLD-rule answers (only where old != key)")
for g, f in G.items():
    R = [r for r in rows if f(r) and r["old"] != r["key"]]
    if R:
        print(f"{g:22} {len(R):3} " + " ".join(f"{mean([pick(r, s) == r['old'] for r in R]):7.0%}" for s in S))
rng = random.Random(1)
print("\nformula vs normal, right answers, paired bootstrap over sets (95%)")
for s in ("w0.1", "w0"):
    for g, f in G.items():
        R = [r for r in rows if f(r)]
        sets = sorted({r["set"] for r in R})
        d = {k: [(pick(r, s) == r["key"]) - (pick(r, "normal") == r["key"]) for r in R if r["set"] == k] for k in sets}
        bs = sorted(mean([x for k in [rng.choice(sets) for _ in sets] for x in d[k]]) for _ in range(2000))
        print(f"  {s:5} {g:22} {mean([x for v in d.values() for x in v]):+.0%} [{bs[50]:+.0%}, {bs[1949]:+.0%}]")
