"""Score the two-pass run vs one-pass.  python3 -I score_twopass.py <match dir>
Per condition: D1+D3 old-rule answers and D1-D4 right, one-pass (results/) vs two-pass (results_twopass/);
draft accuracy; how often the check flips a right draft to wrong and a wrong draft to right.
Paired bootstrap over 30 sets: two-pass minus one-pass on D1+D3 old-rule answers and on D1-D4 right."""
import json, os, random, re, sys

D = sys.argv[1]
items = {(x["set"], x["arm"], x["probes"][0]["id"]): x for x in json.load(open(os.path.join(D, "sets_match.json")))}
one = json.load(open(os.path.join(D, "results", "Qwen2.5-14B-Instruct.json")))["answers"]
two = json.load(open(os.path.join(D, "results_twopass", "Qwen2.5-14B-Instruct.json")))["answers"]


def first_label(t, labels):
    w = [x.upper() for x in re.findall(r"[A-Za-z]+", t) if x.upper() in labels]
    return w[0] if w else None


def table(ans, prefix):
    old, right, flips, nolab = {}, {}, dict(rw=0, wr=0, r=0, w=0), 0
    for a in ans:
        arm = a["arm"][len(prefix):]
        it = items[(a["set"], arm, a["probe_ids"][0])]
        pr = it["probes"][0]
        labs = [first_label(t, it["labels"]) for t in [a["greedy"]] + a["samples"]]
        nolab += labs.count(None)
        k = (arm, pr["id"], a["set"])
        old[k] = sum(x == pr["old"] for x in labs) / 5
        right[k] = sum(x == pr["key"] for x in labs) / 5
        if "draft" in a:
            dr = a["draft"] == pr["key"]
            flips["r" if dr else "w"] += 5
            flips["rw" if dr else "wr"] += sum((x != pr["key"]) if dr else (x == pr["key"]) for x in labs)
    return old, right, flips, nolab


o1, r1, _, n1 = table(one, "")
o2, r2, fl, n2 = table(two, "tp_")
mean = lambda v: sum(v) / len(v)
rng = random.Random(1)
print(f"no label: one-pass {n1}, two-pass {n2}")
print(f"draft right {fl['r'] / (fl['r'] + fl['w']):.0%}; check turned right draft wrong {fl['rw'] / max(fl['r'], 1):.0%}, "
      f"wrong draft right {fl['wr'] / max(fl['w'], 1):.0%}")
DD = ("D1", "D2", "D3", "D4")
print("\narm                 | D1+D3 old-rule: 1-pass 2-pass | D1-D4 right: 1-pass 2-pass | per probe right 2-pass")
for ch in ("change", "nochange"):
    for n in (20, 60):
        for c in "ABC":
            arm = f"{c}_{ch}_{n}"
            g = lambda d, ps: mean([d[(arm, p, s)] for p in ps for s in range(30)])
            print(f"{arm:19} | {g(o1, ('D1', 'D3')):6.0%} {g(o2, ('D1', 'D3')):6.0%} | {g(r1, DD):6.0%} {g(r2, DD):6.0%} | "
                  + " ".join(f"{p} {g(r2, (p,)):.0%}" for p in DD))
print("\ntwo-pass minus one-pass, paired over sets (95% interval):")
for ch in ("change", "nochange"):
    for n in (20, 60):
        for c in "ABC":
            arm = f"{c}_{ch}_{n}"
            for name, d1, d2, ps in (("D1+D3 old", o1, o2, ("D1", "D3")), ("D1-D4 right", r1, r2, DD)):
                d = [mean([d2[(arm, p, s)] for p in ps]) - mean([d1[(arm, p, s)] for p in ps]) for s in range(30)]
                b = sorted(mean([rng.choice(d) for _ in d]) for _ in range(4000))
                print(f"  {arm:19} {name:12} {mean(d):+.0%} [{b[100]:+.0%}, {b[3899]:+.0%}]")
