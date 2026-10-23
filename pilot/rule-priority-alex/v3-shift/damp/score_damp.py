"""Score the pull dial.  python3 -I score_damp.py damp_<model>.jsonl
Answer = label with the highest first-token log-prob. For every setting (always on) and its 'lamp' version (applied
only when the layer-32 logit lens and the normal answer disagree = a takeover was caught), per group:
  change D1+D3: old-rule answers / right;  change D2+D4: right;  no-change D1+D3: right (here the examples are right).
Per condition A/B/C for the main settings. Paired bootstrap over sets vs normal for the all-condition numbers."""
import json, random, sys

rows = [json.loads(l) for l in open(sys.argv[1])]
rows = [r for r in rows if len(set(r["label_tok"])) == 3]
print(len(rows), "prompts; max mask_check", max(r["mask_check"] for r in rows))
SET = list(rows[0]["damp"].keys())
arg = lambda v: max(range(3), key=lambda i: v[i])


def ans(r, s, lamp):
    if lamp and arg(r["lens32"]) == arg(r["damp"]["1.0"]):
        s = "1.0"
    return r["labels"][arg(r["damp"][s])]


G = {"chg D1+D3 old": lambda r: "_change" in r["arm"] and r["probe"] in ("D1", "D3"),
     "chg D1+D3 right": lambda r: "_change" in r["arm"] and r["probe"] in ("D1", "D3"),
     "chg D2+D4 right": lambda r: "_change" in r["arm"] and r["probe"] in ("D2", "D4"),
     "nochg D1+D3 right": lambda r: "_nochange" in r["arm"]}
val = lambda g, r, a: a == r["old"] if g.endswith("old") else a == r["key"]
mean = lambda v: sum(v) / len(v) if v else float("nan")
rng = random.Random(1)
for cond in ("all", "A", "B", "C"):
    rs = rows if cond == "all" else [r for r in rows if r["arm"].startswith(cond + "_")]
    print(f"\n=== condition {cond} ===   " + " | ".join(G))
    for lamp in (False, True):
        for s in SET:
            if lamp and s == "1.0":
                continue
            cells = [mean([val(g, r, ans(r, s, lamp)) for r in rs if f(r)]) for g, f in G.items()]
            name = ("lamp " if lamp else "") + s
            print(f"  {name:16} " + " | ".join(f"{c:6.0%}" for c in cells))
print("\nlamp trigger rate (layer-32 lens disagrees with the normal answer):")
for g, f in list(G.items())[1:]:
    rs = [r for r in rows if f(r)]
    print(f"  {g:18} {mean([arg(r['lens32']) != arg(r['damp']['1.0']) for r in rs]):.0%}")
print("\nall conditions, paired over sets vs normal (95% interval):")
for s in ("all_0.25", "all_0.1", "all_0"):
    for lamp in (False, True):
        for g, f in G.items():
            per = {}
            for r in rows:
                if f(r):
                    per.setdefault(r["set"], []).append(val(g, r, ans(r, s, lamp)) - val(g, r, ans(r, "1.0", False)))
            d = [mean(v) for v in per.values()]
            b = sorted(mean([rng.choice(d) for _ in d]) for _ in range(4000))
            print(f"  {('lamp ' if lamp else '') + s:15} {g:18} {mean(d):+.0%} [{b[100]:+.0%}, {b[3899]:+.0%}]")
