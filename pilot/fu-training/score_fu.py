"""Score the fu/good run.  python3 -I score_fu.py fu_<model><tag>.jsonl
Answer = candidate with the highest summed log-prob. Per family x kind: right, old (copied) where old != key.
Paired bootstrap over items for tuned - base. Bench: MMLU and GSM8K accuracy."""
import json, random, sys

rows = [json.loads(l) for l in open(sys.argv[1])]
mean = lambda v: sum(v) / len(v) if v else float("nan")
rng = random.Random(1)
T = {(r["model"], r["i"]): r for r in rows if r["stage"] == "test"}
models = [m for m in ("base", "tuned") if any(k[0] == m for k in T)]
pick = lambda r: r["cands"][max(range(len(r["cands"])), key=lambda i: r["scores"][i])]
items = sorted({k[1] for k in T})
groups = []
for fam in ("rule", "fact", "math", "price"):
    for kind in ("change", "same", "noisy", "nohist", "fresh", "subtle", "top", "plain", "decoy", "plain2", "decoy2", "decoy_plain2"):
        ids = [i for i in items if T[("base", i)]["fam"] == fam and T[("base", i)]["kind"] == kind]
        if ids:
            groups.append((fam, kind, ids))
seen = {"rule": "trained family, new words", "fact": "trained family, new names", "math": "NEVER TRAINED", "price": "NEVER TRAINED"}
print(f"{'family':6} {'kind':7} {'n':>4} | " + " | ".join(f"{m} right  old" for m in models) + " | tuned-base right [95%]")
for fam, kind, ids in groups:
    line = f"{fam:6} {kind:7} {len(ids):4} | "
    for m in models:
        rr = [pick(T[(m, i)]) == T[(m, i)]["key"] for i in ids]
        oo = [pick(T[(m, i)]) == T[(m, i)]["old"] and T[(m, i)]["old"] != T[(m, i)]["key"] for i in ids]
        line += f"  {mean(rr):5.0%} {mean(oo):5.0%} | "
    if len(models) == 2:
        d = [(pick(T[("tuned", i)]) == T[("tuned", i)]["key"]) - (pick(T[("base", i)]) == T[("base", i)]["key"]) for i in ids]
        b = sorted(mean([rng.choice(d) for _ in d]) for _ in range(4000))
        line += f"{mean(d):+.0%} [{b[100]:+.0%}, {b[3899]:+.0%}]"
    print(line + f"   ({seen[fam]})")
tr = [r for r in rows if r["stage"] == "train"]
if tr:
    print("\ntraining (per 100 pairs): " + ", ".join(f"{r['step']}: loss {r['loss']:.2f} acc {r['acc']:.2f}" for r in tr))
for b in ("mmlu", "gsm8k"):
    out = []
    for m in models:
        v = [r["pred"] == r["gold"] for r in rows if r["stage"] == b and r["model"] == m]
        out.append(f"{m} {mean(v):.1%} (n={len(v)})")
    print(f"{b.upper()}: " + ", ".join(out))
