"""Per model x family: fast accuracy on traps and controls, habit share among trap errors, confidence of wrong answers.
Confidence = probability of the chosen option after normalising summed log-probs over the options.
"Confident habit error" = trap answered with the habit option at confidence >= 0.8.
python3 -I score_ct.py pilot"""
import json, math, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1] if len(sys.argv) > 1 else "pilot"
MODELS = ["qwen14", "phi4", "olmo13"]
FAM = ["memo", "hindgen", "redef", "R", "N", "K", "D", "parcels", "bonus"]
GOOD = dict(acc=60, conf_share=50, ctrl=80)   # a family "has a confident habit" if trap acc <= 60, >= 50% of errors confident-habit, control >= 80


def probs(res):
    s = [sum(r) for r in res]; m = max(s); e = [math.exp(v - m) for v in s]; z = sum(e)
    return [v / z for v in e]


rows = defaultdict(list)
for mk in MODELS:
    fn = os.path.join(HERE, f"out_{tag}_{mk}.jsonl")
    if not os.path.exists(fn):
        print("missing", fn); continue
    for l in open(fn):
        x = json.loads(l)
        if "res" not in x:
            continue
        p = probs(x["res"]); a = max(range(len(p)), key=p.__getitem__)
        rows[(mk, x["family"], x["pop"])].append(dict(right=a == x["key"], habit=x["habit"] is not None and a == x["habit"], conf=p[a],
                                                      ph=p[x["habit"]] if x["habit"] is not None else None))

out = []
P = lambda v, n: f"{round(100 * v / n) if n else '-':>3}"
hdr = f"{'model':7} {'family':8} {'n':>3} {'trap right':>10} {'habit':>6} {'conf.habit':>10} {'conf|wrong':>10} {'ctrl right':>10} {'n':>3}  verdict"
out.append(hdr)
summary = defaultdict(list)
for f in FAM:
    for mk in MODELS:
        T = rows.get((mk, f, "trap"), []); C = rows.get((mk, f, "control"), [])
        if not T:
            continue
        n = len(T); r = sum(t["right"] for t in T); h = sum(t["habit"] for t in T); ch = sum(t["habit"] and t["conf"] >= 0.8 for t in T)
        wr = [t["conf"] for t in T if not t["right"]]
        cr = sum(c["right"] for c in C)
        acc = 100 * r / n; share = 100 * ch / max(1, n - r); ctrl = 100 * cr / len(C) if C else None
        ok = acc <= GOOD["acc"] and share >= GOOD["conf_share"] and (ctrl is None or ctrl >= GOOD["ctrl"])
        summary[f].append(ok)
        out.append(f"{mk:7} {f:8} {n:>3} {P(r, n):>10} {P(h, n):>6} {P(ch, n):>10} "
                   f"{(str(round(100 * sum(wr) / len(wr))) if wr else '-'):>10} {(P(cr, len(C)) if C else '  -'):>10} {len(C):>3}  "
                   f"{'CONFIDENT HABIT' if ok else ''}")
    out.append("")
out.append("Columns are per 100 trap items (conf|wrong = mean confidence of wrong fast answers, in %).")
out.append("Families with a confident habit in all 3 models: " + ", ".join(f for f in FAM if summary[f] and all(summary[f]) and len(summary[f]) == 3))
out.append("... in at least 2 of 3 models: " + ", ".join(f for f in FAM if sum(summary[f]) >= 2))
txt = "\n".join(out); print(txt)
open(os.path.join(HERE, f"report_{tag}.txt"), "w").write(txt + "\n")
