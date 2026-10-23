"""Score the selective block run.  python3 -I score_blockex.py <dir v3-shift>
blockex = the 20 copy heads cannot attend to the past-examples block (they still see rules, update note, test case,
and their own generated text). Compared with normal and with the old 'heads fully off' runs (heads/, bench/)
on the same prompts and questions."""
import json, os, random, sys

D = sys.argv[1]
rd = lambda p: [json.loads(l) for l in open(os.path.join(D, p))]
bx = rd("blockex/blockex_Qwen2.5-14B-Instruct.jsonl")
hd = rd("heads/heads_Qwen2.5-14B-Instruct.jsonl")
bn = rd("bench/bench_Qwen2.5-14B-Instruct.jsonl")
mean = lambda v: sum(v) / len(v) if v else float("nan")
arg = lambda v: max(range(len(v)), key=lambda i: v[i])
rng = random.Random(1)

ours = [r for r in bx if r["bench"] == "ours"]
off = {(r["set"], r["arm"], r["probe"]): r["labels"][arg(r["ko"]["top20"])] for r in hd if r["stage"] == 2}
ans = lambda r, s: off[(r["set"], r["arm"], r["probe"])] if s == "heads off" else r["labels"][arg(r["scores"][s])]
G = {"chg D1+D3 old": (lambda r: "_change" in r["arm"] and r["probe"] in ("D1", "D3"), "old"),
     "chg D1+D3 right": (lambda r: "_change" in r["arm"] and r["probe"] in ("D1", "D3"), "key"),
     "chg D2+D4 right": (lambda r: "_change" in r["arm"] and r["probe"] in ("D2", "D4"), "key"),
     "nochg right": (lambda r: "_nochange" in r["arm"], "key"),
     "all right": (lambda r: True, "key")}
print(f"OUR TASK, held-out sets 15-29, {len(ours)} prompts")
for cond in ("all", "A", "B", "C"):
    rs = ours if cond == "all" else [r for r in ours if r["arm"][0] == cond]
    print(f"=== {cond} ===  " + " | ".join(G))
    for s in ("normal", "heads off", "blockex"):
        print(f"  {s:10} " + " | ".join(f"{mean([ans(r, s) == r[k] for r in rs if f(r)]):6.0%}" for f, k in G.values()))
print("paired over sets vs normal (95% interval):")
for s in ("blockex", "heads off"):
    for g, (f, k) in G.items():
        per = {}
        for r in ours:
            if f(r):
                per.setdefault(r["set"], []).append((ans(r, s) == r[k]) - (ans(r, "normal") == r[k]))
        d = [mean(x) for x in per.values()]
        b = sorted(mean([rng.choice(d) for _ in d]) for _ in range(4000))
        print(f"  {s:10} {g:16} {mean(d):+.0%} [{b[100]:+.0%}, {b[3899]:+.0%}]")


def bench(name, old_key):
    a = [r for r in bx if r["bench"] == name]
    o = [r for r in bn if r["bench"] == name][:len(a)]
    assert all(x["gold"] == y["gold"] for x, y in zip(a, o))
    acc = lambda rs, s: [rr["pred"][s] == rr["gold"] for rr in rs]
    print(f"\n{name.upper()} {len(a)} questions")
    base = acc(a, "normal")
    for lab, v in (("normal", base), ("heads off (old run)", acc(o, old_key)), ("blockex", acc(a, "blockex"))):
        d = [x - y for x, y in zip(v, base)]
        b = sorted(mean([rng.choice(d) for _ in d]) for _ in range(4000))
        fx = sum(x and not y for x, y in zip(v, base)); br = sum(y and not x for x, y in zip(v, base))
        print(f"  {lab:20} {mean(v):6.1%}  diff {mean(d):+.1%} [{b[100]:+.1%}, {b[3899]:+.1%}]  fixed {fx} broken {br}")
    print(f"  (old run normal {mean(acc(o, 'normal')):.1%})")


bench("mmlu", "off_all")
bench("gsm8k", "off_all")
