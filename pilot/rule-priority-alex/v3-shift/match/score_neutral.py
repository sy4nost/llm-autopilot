"""Neutral caption ('past cases and the labels they were given at the time') vs original ('correct classifications').
python3 -I score_neutral.py <match dir>.  One pass, N=60, A/B/C. Paired bootstrap over 30 sets (neutral - original)."""
import json, os, random, re, sys

D = sys.argv[1]
items = {(x["set"], x["arm"], x["probes"][0]["id"]): x for x in json.load(open(os.path.join(D, "sets_match.json")))}
orig = json.load(open(os.path.join(D, "results", "Qwen2.5-14B-Instruct.json")))["answers"]
neu = json.load(open(os.path.join(D, "results_neutral", "Qwen2.5-14B-Instruct.json")))["answers"]
fl = lambda t, L: ([w.upper() for w in re.findall(r"[A-Za-z]+", t) if w.upper() in L] or [None])[0]


def tab(ans, pre):
    old, right = {}, {}
    for a in ans:
        arm = a["arm"][len(pre):]
        it = items[(a["set"], arm, a["probe_ids"][0])]
        pr = it["probes"][0]
        labs = [fl(t, it["labels"]) for t in [a["greedy"]] + a["samples"]]
        old[(arm, pr["id"], a["set"])] = sum(x == pr["old"] for x in labs) / 5
        right[(arm, pr["id"], a["set"])] = sum(x == pr["key"] for x in labs) / 5
    return old, right


o0, r0 = tab(orig, "")
o1, r1 = tab(neu, "nt1_")
o2, r2 = tab(json.load(open(os.path.join(D, "results_outcome", "Qwen2.5-14B-Instruct.json")))["answers"], "nt3_")
mean = lambda v: sum(v) / len(v)
rng = random.Random(1)
print("captions: 0 = 'correct classifications', 1 = 'labels given at the time', 2 = 1 + '(correct under the rules in force at that time)'")
print("arm            | D1+D3 old-rule: 0 1 2 | D1-D4 right: 0 1 2 | diff vs 0 on D1+D3 old (95% interval)")
for arm in [f"{c}_change_60" for c in "ABC"] + [f"{c}_nochange_60" for c in "ABC"]:
    ps = ("D1", "D2", "D3", "D4") if "_change" in arm else ("D1", "D3")
    g = lambda d, q: mean([d[(arm, p, s)] for p in q for s in range(30)])
    ci = []
    for ox in (o1, o2):
        d = [mean([ox[(arm, p, s)] for p in ("D1", "D3")]) - mean([o0[(arm, p, s)] for p in ("D1", "D3")]) for s in range(30)]
        b = sorted(mean([rng.choice(d) for _ in d]) for _ in range(4000))
        ci.append(f"{mean(d):+.0%} [{b[100]:+.0%}, {b[3899]:+.0%}]")
    print(f"{arm:14} | " + " ".join(f"{g(x, ('D1', 'D3')):4.0%}" for x in (o0, o1, o2)) + " | "
          + " ".join(f"{g(x, ps):4.0%}" for x in (r0, r1, r2)) + " | " + " ; ".join(ci)
          + ("" if "_change" in arm else "  (right on D1+D3 only)"))
