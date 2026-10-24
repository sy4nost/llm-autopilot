"""forces-map scoring.  python3 score_fm.py main
Value of a text: NUM family = numbers (digits or words) not appearing in the final question; PREF = slot values named.
DIRECT/goal: first value; THINK: first value after the last "Answer:", else the last value.
Label of an answer: right / shown (= the value the assistant's earlier replies showed, trap cells only) / truth (NUM: the
true fact) / other.  MAP lines: USER'S INSTRUCTIONS (I), EARLIER ANSWERS (E), GENERAL KNOWLEDGE (K), first value per line.
Lamps:  MAP  = I and E both give a value and I != E   (primary: instruction vs earlier answers)
        MAP3 = any two of I, E, K give different values (reported only; in NUM general knowledge always differs)
        GOAL = the goal sentence names exactly one value and it differs from the DIRECT answer (DIRECT names a value)
SCHEME(lamp) = THINK answer if the lamp fires, else DIRECT answer.  Bootstrap over dialogues, stratified by family x cell."""
import json, os, re, sys, random
from collections import defaultdict, Counter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import make_fm
tag = sys.argv[1] if len(sys.argv) > 1 else "main"
R = {it["id"]: it for it in make_fm.ITEMS}
MODELS = ["qwen14", "phi4", "olmo13"]
WORDS = {w: k for k, w in enumerate("zero _one_ two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                                    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?|\b(?:" + "|".join(WORDS) + r")\b", re.I)


def vals(text, r):
    if r["fam"] == "NUM":
        echo = {m.group().lower() for m in NUM.finditer(r["final"])}
        out = []
        for m in NUM.finditer(text):
            g = m.group().lower()
            if g in echo:
                continue
            out.append(WORDS[g] if g in WORDS else int(float(g)))
        return out
    t = text.lower(); hits = []
    for v in r["values"]:
        hits += [(m.start(), v) for m in re.finditer(r"(?<![a-z])" + re.escape(v) + r"(?![a-z])", t)]
    return [v for _, v in sorted(hits)]


def value(text, r, think=False):
    if think:
        i = text.lower().rfind("answer:")
        if i >= 0 and vals(text[i:], r):
            return vals(text[i:], r)[0]
        v = vals(text, r); return v[-1] if v else None
    v = vals(text, r); return v[0] if v else None


def label(v, r):
    if v is None:
        return "none"
    if v == r["right"]:
        return "right"
    if v == r["shown"]:
        return "shown"
    return "truth" if v == r["M"] else "other"


LABELS = (("I", r"user\S{0,2}s?\s*instructions?"), ("E", r"assistant\S{0,2}s?\s*recent\s*repl\w*"), ("K", r"general\s*knowledge"))
STATS = Counter()


def map_lines(text, r):
    # v1.1 (review M1): label only at line start (optional markdown), value read to the line end; fallback = line position;
    # E takes the LAST value on its line (review M2)
    lines = [l for l in text.split("\n") if l.strip()]; res = {}
    for n, (key, pat) in enumerate(LABELS):
        m = re.search(r"^[ \t*#\-]*" + pat + r"[ \t*]*:[ \t*]*([^\n]*)", text, re.I | re.M)
        if m:   # v1.3: read the block up to the next label (OLMo lists the replies on separate lines under line 2)
            nxt = re.search(r"^[ \t*#\-]*(?:" + "|".join(p for _, p in LABELS) + r")[ \t*]*:", text[m.end():], re.I | re.M)
            line = text[m.start(1):m.end() + (nxt.start() if nxt else len(text) - m.end())]
        else:
            STATS["label missing"] += 1; line = lines[n].split(":", 1)[-1] if n < len(lines) else ""
        v = vals(line, r)
        STATS["empty"] += not v; STATS["multi-value"] += len(set(v)) > 1
        res[key] = (v[-1] if key == "E" else v[0]) if v else None
    return res


def goal_value(g, r):
    vs = set(vals(g, r)); return vs.pop() if len(vs) == 1 else None


def pct(xs):
    return 100 * sum(xs) / len(xs) if xs else float("nan")


def boot(rows, f, nb=2000, seed=0):
    st = defaultdict(list)
    for x in rows:
        st[(x["fam"], x["cell"])].append(x)
    rnd = random.Random(seed); res = []
    for _ in range(nb):
        v = f([rnd.choice(g) for g in st.values() for _ in g])
        if v == v:
            res.append(v)
    res.sort(); return res[int(.025 * len(res))], res[int(.975 * len(res)) - 1]


def main():
    over = testn = 0
    for mk in MODELS:
        P = {s: os.path.join(HERE, f"out_{tag}_{s}_{mk}.json") for s in ("d", "th", "map", "g1")}
        if not all(os.path.exists(p) for p in P.values()):
            print(mk, "missing"); continue
        O = {s: json.load(open(p)) for s, p in P.items()}
        ids = sorted(O["d"])
        if any(set(O[s]) != set(ids) for s in O):
            print(mk, "PARTIAL - not scored"); continue
        rows = []
        for k in ids:
            r = R[k]; vd = value(O["d"][k], r); vt = value(O["th"][k], r, think=True); ml = map_lines(O["map"][k], r)
            gv = goal_value(O["g1"][k], r)
            lam = {"MAP": ml["I"] is not None and ml["E"] is not None and ml["I"] != ml["E"],
                   "MAP3": len({v for v in ml.values() if v is not None}) > 1,
                   "GOAL": gv is not None and vd is not None and gv != vd,
                   "MAPD": ml["I"] is not None and vd is not None and ml["I"] != vd,    # v1.1 (review S3): map line I != DIRECT
                   "ORACLE": label(vd, r) != "right"}                                    # v1.1 (review M4): ideal lamp
            lab = {"DIRECT": label(vd, r), "THINK": label(vt, r), "MAP-I": label(ml["I"], r)}
            for L, f in lam.items():
                lab["S-" + L] = lab["THINK"] if f else lab["DIRECT"]
            rows.append(dict(id=k, fam=r["fam"], cell=r["cell"], trap=r["cell"].startswith("trap"), lam=lam, lab=lab,
                             ml={a: label(v, r) for a, v in ml.items()}))
        acc = lambda rr, a: pct([x["lab"][a] == "right" for x in rr])
        print(f"\n=== {mk} ({len(rows)} dialogues) ===")
        print("  right %          " + "  ".join(f"{c:>11s}" for c in make_fm.CELLS) + "   all")
        for fam in ("NUM", "PREF"):
            for a in ("DIRECT", "THINK", "MAP-I"):
                print(f"  {fam:4s} {a:7s}     " + "  ".join(f"{acc([x for x in rows if x['fam'] == fam and x['cell'] == c], a):11.0f}" for c in make_fm.CELLS)
                      + f"  {acc([x for x in rows if x['fam'] == fam], a):5.0f}")
        for fam in ("NUM", "PREF"):
            tr = [x for x in rows if x["fam"] == fam and x["trap"]]
            print(f"  {fam} traps, DIRECT answers: " + ", ".join(f"{k} {v}" for k, v in Counter(x['lab']['DIRECT'] for x in tr).most_common())
                  + " | MAP line E: " + ", ".join(f"{k} {v}" for k, v in Counter(x['ml']['E'] for x in tr).most_common()))
        print("  lamp     fires calm  fires trap  catch(DIRECT wrong)  false alarm(DIRECT right)  SCHEME  vs random [CI]")
        res = {}
        for L in ("MAP", "MAP3", "GOAL", "MAPD", "ORACLE"):
            calm = [x for x in rows if not x["trap"]]; trap = [x for x in rows if x["trap"]]
            wr = [x for x in rows if x["lab"]["DIRECT"] != "right"]; ok = [x for x in rows if x["lab"]["DIRECT"] == "right"]
            sel = lambda rr, L=L: pct([x["lam"][L] for x in rr if x["trap"]]) - pct([x["lam"][L] for x in rr if not x["trap"]])
            rnd = lambda rr, L=L: (lambda c: acc(rr, "S-" + L) - (acc(rr, "DIRECT") + c * (acc(rr, "THINK") - acc(rr, "DIRECT"))))(sum(x["lam"][L] for x in rr) / len(rr))
            s_lo, _ = boot(rows, sel); r_lo, r_hi = boot(rows, rnd)
            res[L] = (sel(rows), s_lo, rnd(rows), r_lo)
            print(f"  {L:6s} {pct([x['lam'][L] for x in calm]):9.0f}  {pct([x['lam'][L] for x in trap]):10.0f}  {pct([x['lam'][L] for x in wr]):12.0f} of {len(wr):3d}"
                  f"  {pct([x['lam'][L] for x in ok]):14.0f} of {len(ok):3d}      {acc(rows, 'S-' + L):6.1f}  {rnd(rows):+5.1f} [{r_lo:+5.1f},{r_hi:+5.1f}]")
        multi = pct([len(set(vals(O["d"][x["id"]], R[x["id"]]))) > 1 for x in rows])
        print(f"  DIRECT {acc(rows, 'DIRECT'):.1f}   always THINK {acc(rows, 'THINK'):.1f}   map line I as the answer {acc(rows, 'MAP-I'):.1f}"
              f"   (DIRECT answers naming >1 value: {multi:.0f}%)")
        print("  map parsing: " + ", ".join(f"{k} {v}" for k, v in STATS.items()) + f" (of {3 * len(rows)} lines)"); STATS.clear()
        p1 = res["MAP"][1] >= 20
        wr = [x for x in rows if x["lab"]["DIRECT"] != "right"]
        testable = len(wr) >= max(20, len(rows) // 12) and pct([x["lab"]["THINK"] == "right" for x in wr]) >= 100 / 3   # v1.1 (review M4)
        p2 = res["MAP"][3] > 0 if testable else None
        wt = [x for x in rows if x["trap"] and x["lab"]["DIRECT"] != "right"]
        p3v = pct([x["lab"]["MAP-I"] == "right" for x in wt])
        print(f"  P1 MAP selectivity (fires trap - calm) {res['MAP'][0]:+.0f}, CI low {res['MAP'][1]:+.0f} (need >= +20) -> {'PASS' if p1 else 'FAIL'}")
        print(f"  P2 SCHEME(MAP) beats random re-check at same rate, CI low {res['MAP'][3]:+.1f} -> {'n/a (too few DIRECT errors THINK can fix)' if p2 is None else 'PASS' if p2 else 'FAIL'}")
        print(f"  P3 map line 'user's instructions' right on {p3v:.0f}% of {len(wt)} traps that DIRECT got wrong (need >= 50, n >= 15) -> "
              f"{'n/a' if len(wt) < 15 else 'PASS' if p3v >= 50 else 'FAIL'}")
        over += bool(p1 and p2); testn += p2 is not None
    print(f"\nOVERALL: P1+P2 PASS in {over} of {testn} testable models (need 2; fewer than 2 testable = inconclusive)")


if __name__ == "__main__":
    main()
