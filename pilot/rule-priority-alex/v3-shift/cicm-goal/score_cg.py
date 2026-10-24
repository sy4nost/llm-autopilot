"""cicm-goal scoring.  python3 score_cg.py main
Answer value = slot value named in the answer (DIRECT/PLACEBO/GOAL: the earliest-mentioned slot value; THINK: the value
after the last "Answer:", else the last-mentioned slot value). Label: correct (= current_value), stale (in stale_values),
other (another value or none). Goal sentence value = the single distinct slot value it names (else none/mixed).
conflict = goal names exactly one value and it differs from the DIRECT answer's value.
SCHEME = THINK answer if conflict, else DIRECT answer.  Bootstrap over CICM clusters (slot, current value, k, cell)."""
import json, os, re, sys, random
from collections import defaultdict, Counter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv += [] if len(sys.argv) > 1 else ["main"]
tag = sys.argv[1]
sys.path.insert(0, HERE); import make_cg
R = {r["id"]: r for r in make_cg.CAL}
MODELS = ["qwen14", "phi4", "olmo13"]


def mentions(text, r):
    t = text.lower(); hits = []
    for v in r["slot_values"]:
        pat = r"hip[\s\-]?hop" if v == "hiphop" else r"[\s\-]+".join(re.escape(w) for w in v.lower().split())   # v2: "gluten-free", "hip hop"
        for m in re.finditer(r"(?<![a-z])" + pat + r"(?![a-z])", t):
            hits.append((m.start(), v))
    return sorted(hits)


def value(text, r, think=False):
    if think:
        i = text.lower().rfind("answer:")
        if i >= 0:
            h = mentions(text[i:], r)
            if h:
                return h[0][1]
        h = mentions(text, r); return h[-1][1] if h else None
    h = mentions(text, r); return h[0][1] if h else None


def label(v, r):
    return "correct" if v == r["current_value"] else "stale" if v in r["stale_values"] else "other"


def goal_value(g, r):
    vs = {v for _, v in mentions(g, r)}
    return vs.pop() if len(vs) == 1 else None


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def boot(cl, f, nb=2000, seed=0):
    rnd = random.Random(seed); keys = sorted(cl); res = []
    for _ in range(nb):
        v = f([x for k in (rnd.choice(keys) for _ in keys) for x in cl[k]])
        if v == v:
            res.append(v)
    res.sort(); return (res[int(.025 * len(res))], res[int(.975 * len(res)) - 1]) if res else (float("nan"),) * 2


def pct(xs):
    return 100 * sum(xs) / len(xs) if xs else float("nan")


def main():
    passes = 0
    for mk in MODELS:
        p2, pt = os.path.join(HERE, f"out_{tag}_g2_{mk}.json"), os.path.join(HERE, f"out_{tag}_th_{mk}.json")
        if not (os.path.exists(p2) and os.path.exists(pt)):
            print(mk, "missing"); continue
        A = json.load(open(p2)); T = json.load(open(pt))
        G = {i["src"]: i["goal"] for i in json.load(open(os.path.join(HERE, f"sets_g2_{tag}_{mk}.json")))["items"]}
        if len(A) != 3 * len(G) or any(k not in T for k in G):
            print(mk, "PARTIAL - not scored"); continue
        rows = []
        for k, g in G.items():
            r = R[k]; vd = value(A[f"{k}|DIRECT"], r); gv = goal_value(g, r); conflict = gv is not None and vd is not None and gv != vd   # v2: DIRECT must name a value
            vt = value(T[k], r, think=True)
            lab = {"DIRECT": label(vd, r), "PLACEBO": label(value(A[f"{k}|PLACEBO"], r), r), "GOAL": label(value(A[f"{k}|GOAL"], r), r),
                   "THINK": label(vt, r)}
            lab["SCHEME"] = lab["THINK"] if conflict else lab["DIRECT"]
            rows.append(dict(lab=lab, conflict=conflict, glab=label(gv, r) if gv else "none/mixed", k=r["k_overwrites"],
                             cl=(r["slot"], r["current_value"], r["k_overwrites"], str(r["factorial_cell"]))))
        cl = defaultdict(list)
        for x in rows:
            cl[x["cl"]].append(x)
        acc = lambda rr, a: pct([x["lab"][a] == "correct" for x in rr])
        print(f"\n=== {mk} ({len(rows)} dialogues) ===")
        for a in ("DIRECT", "PLACEBO", "GOAL", "THINK", "SCHEME"):
            c = Counter(x["lab"][a] for x in rows)
            print(f"  {a:8s} correct {c['correct'] / len(rows) * 100:5.1f}  stale {c['stale'] / len(rows) * 100:5.1f}  other {c['other'] / len(rows) * 100:5.1f}")
        c = Counter(x["glab"] for x in rows)
        print("  goal sentence names: " + ", ".join(f"{k} {100 * v / len(rows):.0f}%" for k, v in c.most_common()))
        print(f"  conflict (-> slow re-check) in {pct([x['conflict'] for x in rows]):.1f}% of dialogues")
        res = {}
        for name, a, b in (("P1 SCHEME - DIRECT", "SCHEME", "DIRECT"), ("P2 SCHEME - always THINK", "SCHEME", "THINK"),
                           ("GOAL - PLACEBO", "GOAL", "PLACEBO"), ("GOAL - DIRECT", "GOAL", "DIRECT"), ("THINK - DIRECT", "THINK", "DIRECT")):
            f = lambda rr, a=a, b=b: acc(rr, a) - acc(rr, b); lo, hi = boot(cl, f); res[name] = (f(rows), lo)
            print(f"  {name:26s} {f(rows):+6.1f} [{lo:+6.1f},{hi:+6.1f}]")
        cr = pct([x["conflict"] for x in rows]) / 100
        rnd = lambda rr: (lambda c: acc(rr, "SCHEME") - (acc(rr, "DIRECT") + c * (acc(rr, "THINK") - acc(rr, "DIRECT"))))(sum(x["conflict"] for x in rr) / len(rr))
        lo1b, hi1b = boot(cl, rnd)
        print(f"  P1b SCHEME - random re-check at same rate ({100 * cr:.1f}%): {rnd(rows):+6.1f} [{lo1b:+6.1f},{hi1b:+6.1f}]")
        p1 = res["P1 SCHEME - DIRECT"][1] > 0 and lo1b > 0; passes += p1   # v2: P1 PASS needs both CIs above 0
        print(f"  P1 (vs DIRECT and vs random re-check) -> {'PASS' if p1 else 'FAIL'};  P2 (SCHEME not worse than always-THINK by > 3) -> {'PASS' if res['P2 SCHEME - always THINK'][0] >= -3 else 'FAIL'}")
        ok = [x for x in rows if x["lab"]["DIRECT"] == "correct"]
        br = pct([x["lab"]["SCHEME"] != "correct" for x in ok])
        print(f"  P3 harm: SCHEME breaks {br:.1f}% of {len(ok)} DIRECT-correct -> {'PASS' if br <= 5 else 'FAIL'}")
        st = [x for x in rows if x["lab"]["DIRECT"] == "stale"]
        a4 = auc([float(x["conflict"]) for x in st], [float(x["conflict"]) for x in ok])
        print(f"  P4 lamp 'goal != quick answer': catches {pct([x['conflict'] for x in st]):.1f}% of {len(st)} stale, false alarms "
              f"{pct([x['conflict'] for x in ok]):.1f}% of {len(ok)} correct; AUC {a4:.3f} -> {'PASS' if a4 >= .70 else 'FAIL'}")
    print(f"\nOVERALL: P1 PASS in {passes} of 3 models (need 2)")


if __name__ == "__main__":
    main()
