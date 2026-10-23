"""Native vs fictional words, paired by set (same structure and history).  python3 -I score_native.py <native dir>
Per arm: right per probe and old-rule answers on D1+D3, native vs fictional (match/results);
paired bootstrap over 30 sets of native - fictional on D1+D3 old-rule answers."""
import json, os, random, re, sys

D = sys.argv[1]
M = os.path.join(D, "..", "match")
fl = lambda t, L: ([w.upper() for w in re.findall(r"[A-Za-z]+", t) if w.upper() in L] or [None])[0]


def tab(sets, res):
    items = {(x["set"], x["arm"], x["probes"][0]["id"]): x for x in json.load(open(sets))}
    old, right = {}, {}
    for a in json.load(open(res))["answers"]:
        k = (a["set"], a["arm"], a["probe_ids"][0])
        if k not in items:
            continue
        it = items[k]
        pr = it["probes"][0]
        labs = [fl(t, it["labels"]) for t in [a["greedy"]] + a["samples"]]
        old[(a["arm"], pr["id"], a["set"])] = sum(x == pr["old"] for x in labs) / 5
        right[(a["arm"], pr["id"], a["set"])] = sum(x == pr["key"] for x in labs) / 5
    return old, right


on, rn = tab(os.path.join(D, "sets_native.json"), os.path.join(D, "results", "Qwen2.5-14B-Instruct.json"))
of, rf = tab(os.path.join(M, "sets_match.json"), os.path.join(M, "results", "Qwen2.5-14B-Instruct.json"))
mean = lambda v: sum(v) / len(v)
rng = random.Random(1)
print("arm          | right fictional D1 D2 D3 D4 | right native D1 D2 D3 D4 | D1+D3 old: fict native | native-fict (95%)")
for arm in ("fresh", "change_0", "A_change_60", "B_change_60", "C_change_60"):
    ps = ("D1", "D3") if arm == "fresh" else ("D1", "D2", "D3", "D4")
    g = lambda d, p: mean([d[(arm, p, s)] for s in range(30)])
    d = [mean([on[(arm, p, s)] for p in ("D1", "D3")]) - mean([of[(arm, p, s)] for p in ("D1", "D3")]) for s in range(30)]
    b = sorted(mean([rng.choice(d) for _ in d]) for _ in range(4000))
    o = lambda dd: mean([dd[(arm, p, s)] for p in ("D1", "D3") for s in range(30)])
    print(f"{arm:12} | " + " ".join(f"{g(rf, p):4.0%}" for p in ps) + " | " + " ".join(f"{g(rn, p):4.0%}" for p in ps)
          + f" | {o(of):4.0%} {o(on):4.0%} | {mean(d):+.0%} [{b[100]:+.0%}, {b[3899]:+.0%}]")
