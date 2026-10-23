"""Score run 2 of the selective block.  python3 -I score_blockex2.py <blockex2_*.jsonl>
Settings (settings2.json), each = heads that cannot attend to the past-examples block:
  b20 = the 20 copy heads (as run 1); b20+early = those + all heads of layers 20-28;
  b40 = top 40 heads by the heads/ ranking; late_all = all heads of layers 29-47."""
import json, random, sys

rows = [json.loads(l) for l in open(sys.argv[1])]
mean = lambda v: sum(v) / len(v) if v else float("nan")
arg = lambda v: max(range(len(v)), key=lambda i: v[i])
rng = random.Random(1)
ours = [r for r in rows if r["bench"] == "ours"]
S = list(ours[0]["scores"])
ans = lambda r, s: r["labels"][arg(r["scores"][s])]
G = {"chg D1+D3 old": (lambda r: "_change" in r["arm"] and r["probe"] in ("D1", "D3"), "old"),
     "chg D1+D3 right": (lambda r: "_change" in r["arm"] and r["probe"] in ("D1", "D3"), "key"),
     "chg D2+D4 right": (lambda r: "_change" in r["arm"] and r["probe"] in ("D2", "D4"), "key"),
     "nochg right": (lambda r: "_nochange" in r["arm"], "key"),
     "all right": (lambda r: True, "key")}
print(f"OUR TASK, held-out sets 15-29, {len(ours)} prompts")
for cond in ("all", "A", "B", "C"):
    rs = ours if cond == "all" else [r for r in ours if r["arm"][0] == cond]
    print(f"=== {cond} ===  " + " | ".join(G))
    for s in S:
        print(f"  {s:10} " + " | ".join(f"{mean([ans(r, s) == r[k] for r in rs if f(r)]):6.0%}" for f, k in G.values()))
third = lambda r, s: ans(r, s) not in (r["key"], r["old"])
print("change D1+D3: third label (neither right nor old): " + ", ".join(
    f"{s} {mean([third(r, s) for r in ours if G['chg D1+D3 old'][0](r)]):.0%}" for s in S))
print("paired over sets vs normal (95% interval):")
for s in S[1:]:
    for g, (f, k) in G.items():
        per = {}
        for r in ours:
            if f(r):
                per.setdefault(r["set"], []).append((ans(r, s) == r[k]) - (ans(r, "normal") == r[k]))
        d = [mean(x) for x in per.values()]
        b = sorted(mean([rng.choice(d) for _ in d]) for _ in range(4000))
        print(f"  {s:10} {g:16} {mean(d):+.0%} [{b[100]:+.0%}, {b[3899]:+.0%}]")
for name in ("mmlu", "gsm8k"):
    a = [r for r in rows if r["bench"] == name]
    print(f"\n{name.upper()} {len(a)} questions")
    base = [r["pred"]["normal"] == r["gold"] for r in a]
    for s in S:
        v = [r["pred"][s] == r["gold"] for r in a]
        d = [x - y for x, y in zip(v, base)]
        b = sorted(mean([rng.choice(d) for _ in d]) for _ in range(4000))
        fx = sum(x and not y for x, y in zip(v, base)); br = sum(y and not x for x, y in zip(v, base))
        print(f"  {s:10} {mean(v):6.1%}  diff {mean(d):+.1%} [{b[100]:+.1%}, {b[3899]:+.1%}]  fixed {fx} broken {br}")
