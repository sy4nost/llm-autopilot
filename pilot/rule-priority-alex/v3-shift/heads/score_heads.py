"""Score the copy-head run.  python3 -I score_heads.py heads_<model>.jsonl
Heads ranked on the discovery half (sets 0-14); everything below is on the held-out half (sets 15-29).
Answer = label with the highest first-token log-prob. 'lamp' = switch off the top-20 heads only when their attention
to example labels (signal) is above a threshold fixed on the discovery half (midpoint of A vs C change D1/D3 means)."""
import json, random, sys

rows = [json.loads(l) for l in open(sys.argv[1])]
s1 = {(r["set"], r["arm"], r["probe"]): r for r in rows if r["stage"] == 1}
rk = next(r for r in rows if r["stage"] == "rank")
s2 = [r for r in rows if r["stage"] == 2]
print(f"discovery prompts {rk['disc_prompts']}; held-out prompts {len(s2)}")
print("top 20 heads (layer, head, mean attention to the identical example's label):")
print("  " + ", ".join(f"L{l}.{h} {s:.3f}" for l, h, s in rk["top"][:20]))
top20 = [(l, h) for l, h, _ in rk["top"][:20]]
arg = lambda v: max(range(3), key=lambda i: v[i])
sig = lambda r: sum(r["lab"][l][h] for l, h in top20)
mean = lambda v: sum(v) / len(v) if v else float("nan")
disc = [r for r in s1.values() if r["set"] < 15 and "_change" in r["arm"] and r["probe"] in ("D1", "D3")]
thr = (mean([sig(r) for r in disc if r["arm"][0] == "A"]) + mean([sig(r) for r in disc if r["arm"][0] == "C"])) / 2
print(f"lamp threshold {thr:.3f}")


def answer(r, s):
    base = s1[(r["set"], r["arm"], r["probe"])]
    if s == "normal":
        v = base["final"]
    elif s == "lamp top20":
        v = r["ko"]["top20"] if sig(base) > thr else base["final"]
    else:
        v = r["ko"][s]
    return r["labels"][arg(v)]


G = {"chg D1+D3 old": (lambda r: "_change" in r["arm"] and r["probe"] in ("D1", "D3"), "old"),
     "chg D1+D3 right": (lambda r: "_change" in r["arm"] and r["probe"] in ("D1", "D3"), "key"),
     "chg D2+D4 right": (lambda r: "_change" in r["arm"] and r["probe"] in ("D2", "D4"), "key"),
     "nochg D1+D3 right": (lambda r: "_nochange" in r["arm"], "key")}
S = ["normal", "top10", "top20", "top40", "rand20", "lamp top20"]
for cond in ("all", "A", "B", "C"):
    rs = s2 if cond == "all" else [r for r in s2 if r["arm"][0] == cond]
    print(f"\n=== {cond} ===  " + " | ".join(G))
    for s in S:
        print(f"  {s:11} " + " | ".join(f"{mean([answer(r, s) == r[k] for r in rs if f(r)]):6.0%}" for f, k in G.values()))
print("\nlamp signal (top-20 heads' attention to example labels), held-out mean by condition, change D1+D3:")
for c in "ABC":
    v = [sig(s1[(r['set'], r['arm'], r['probe'])]) for r in s2 if r["arm"] == f"{c}_change_60" and r["probe"] in ("D1", "D3")]
    print(f"  {c}: {mean(v):.3f}; lamp on {mean([x > thr for x in v]):.0%}")
v = [sig(s1[(r['set'], r['arm'], r['probe'])]) for r in s2 if r["arm"].startswith("A_nochange")]
print(f"  A no-change: {mean(v):.3f}; lamp on {mean([x > thr for x in v]):.0%}")
rng = random.Random(1)
print("\nheld-out, paired over sets vs normal (95% interval):")
for s in ("top20", "top40", "rand20", "lamp top20"):
    for g, (f, k) in G.items():
        per = {}
        for r in s2:
            if f(r):
                per.setdefault(r["set"], []).append((answer(r, s) == r[k]) - (answer(r, "normal") == r[k]))
        d = [mean(x) for x in per.values()]
        b = sorted(mean([rng.choice(d) for _ in d]) for _ in range(4000))
        print(f"  {s:11} {g:18} {mean(d):+.0%} [{b[100]:+.0%}, {b[3899]:+.0%}]")
