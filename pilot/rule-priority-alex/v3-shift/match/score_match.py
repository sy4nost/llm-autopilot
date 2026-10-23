"""Score the exact-match test.  python3 -I score_match.py <match dir>
Per arm: share of answers (greedy + 4 samples) that give the OLD-rule label and the right label, per probe.
Paired bootstrap over the 30 sets on D1+D3 old-rule answers: A-B, B-C, C-change_0, A-C (change arms, N=20 and 60)."""
import json, os, random, re, sys

D = sys.argv[1]
items = {(x["set"], x["arm"], x["probes"][0]["id"]): x for x in json.load(open(os.path.join(D, "sets_match.json")))}
res = json.load(open(os.path.join(D, "results", "Qwen2.5-14B-Instruct.json")))


def first_label(t, labels):
    w = [x.upper() for x in re.findall(r"[A-Za-z]+", t) if x.upper() in labels]
    return w[0] if w else None


old, right, nolab = {}, {}, 0
for a in res["answers"]:
    it = items[(a["set"], a["arm"], a["probe_ids"][0])]
    pr = it["probes"][0]
    ans = [first_label(t, it["labels"]) for t in [a["greedy"]] + a["samples"]]
    nolab += ans.count(None)
    old[(a["arm"], pr["id"], a["set"])] = sum(x == pr["old"] for x in ans) / 5
    right[(a["arm"], pr["id"], a["set"])] = sum(x == pr["key"] for x in ans) / 5
mean = lambda v: sum(v) / len(v)
rng = random.Random(1)
print("answers without a label:", nolab)
arms = ["fresh", "change_0"] + [f"{c}_{a}_{n}" for a in ("change", "nochange") for n in (20, 60) for c in "ABC"]
print("arm              right: D1   D2   D3   D4  | old-rule label on D1+D3 | on D2+D4")
for arm in arms:
    r = [mean([right[(arm, p, s)] for s in range(30)]) for p in ("D1", "D2", "D3", "D4")]
    o13 = mean([old[(arm, p, s)] for p in ("D1", "D3") for s in range(30)])
    o24 = mean([old[(arm, p, s)] for p in ("D2", "D4") for s in range(30)])
    print(f"{arm:16} " + " ".join(f"{x:4.0%}" for x in r) + f" | {o13:4.0%} | {o24:4.0%}")
d13 = lambda arm, s: mean([old[(arm, p, s)] for p in ("D1", "D3")])
print("\nD1+D3 old-rule answers, paired differences over sets (95% interval):")
for n in (20, 60):
    for a1, a2 in (("A", "B"), ("B", "C"), ("A", "C"), ("C", None), ("B", None)):
        x = f"{a1}_change_{n}"
        y = f"{a2}_change_{n}" if a2 else "change_0"
        d = [d13(x, s) - d13(y, s) for s in range(30)]
        b = sorted(mean([rng.choice(d) for _ in d]) for _ in range(4000))
        print(f"  N={n} {x} - {y}: {mean(d):+.0%} [{b[100]:+.0%}, {b[3899]:+.0%}]")
