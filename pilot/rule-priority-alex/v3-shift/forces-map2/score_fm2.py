"""forces-map2 scoring.  python3 score_fm2.py main
Value of a text: NUM = numbers (digits/words) not in the final question; PREF/CICM = slot values named (hyphen/space
tolerant); HN = Y/N (standalone Y, N, yes, no).  DIRECT and map answers: first value; THINK/SLOW: first value after the
last "Answer:", else the last value.
DISPUTE lamp = the three map answers (request / examples / knowledge; separate calls) give >= 2 distinct values.
SCHEME = SLOW answer if DISPUTE else DIRECT.  Bootstrap over items, stratified by group x sub."""
import json, os, re, sys, random
from collections import defaultdict, Counter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import make_fm2
tag = sys.argv[1] if len(sys.argv) > 1 else "main"
R = {it["id"]: it for it in make_fm2.ITEMS}
MODELS = ["qwen14", "phi4", "olmo13"]; OWN = ["G1", "G2", "G3", "G4"]; OUT = ["CICM", "HN"]
WORDS = {w: k for k, w in enumerate("zero _one_ two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                                    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?|\btwenty[\s\-](?:one|two|three|four|five|six|seven|eight|nine)\b|\b(?:" + "|".join(WORDS) + r")\b", re.I)
UNITS = {w: k for k, w in enumerate("zero one two three four five six seven eight nine".split())}


def numval(g):   # v1.2 (review): "twenty-four" -> 24
    if g.startswith("twenty") and len(g) > 6:
        return 20 + UNITS[g[7:]]
    return WORDS[g] if g in WORDS else int(float(g))


def vals(text, r):
    k = r["kind"]
    if k == "NUM":
        echo = {m.group().lower() for m in NUM.finditer(r["final"])}
        return [numval(g) for g in (m.group().lower() for m in NUM.finditer(text)) if g not in echo]
    if k == "HN":
        return ["Y" if m.group().lower() in ("y", "yes") else "N" for m in re.finditer(r"(?<![A-Za-z])(?:Y|N|[Yy]es|YES|[Nn]o|NO)(?![A-Za-z])", text)]
    t = text.lower(); hits = []
    for v in r["values"]:
        pat = r"hip[\s\-]?hop" if v == "hiphop" else r"[\s\-]+".join(re.escape(w) for w in v.lower().split())
        hits += [(m.start(), v) for m in re.finditer(r"(?<![a-z])" + pat + r"(?![a-z])", t)]
    return [v for _, v in sorted(hits)]


def value(text, r, think=False):
    if think:
        i = text.lower().rfind("answer:")
        if i >= 0 and vals(text[i + 7:], r):
            return vals(text[i + 7:], r)[0]
        v = vals(text, r); return v[-1] if v else None
    v = vals(text, r)
    return (v[-1] if r["kind"] == "HN" else v[0]) if v else None   # v1.2: HN short answers may echo "Y or N" first


def pct(xs):
    return 100 * sum(xs) / len(xs) if xs else float("nan")


def boot(rows, f, nb=2000, seed=0):
    st = defaultdict(list)
    for x in rows:
        st[(x["group"], x["sub"])].append(x)
    rnd = random.Random(seed); res = []
    for _ in range(nb):
        v = f([rnd.choice(g) for g in st.values() for _ in g])
        if v == v:
            res.append(v)
    res.sort(); return res[int(.025 * len(res))], res[int(.975 * len(res)) - 1]


def main():
    over = testn = 0
    for mk in MODELS:
        P = {s: os.path.join(HERE, f"out_{tag}_{s}_{mk}.json") for s in ("d", "th", "slow", "mreq", "mex", "mkn")}
        if not all(os.path.exists(p) for p in P.values()):
            print(mk, "missing"); continue
        O = {s: json.load(open(p)) for s, p in P.items()}; ids = sorted(O["d"])
        if any(set(O[s]) != set(ids) for s in O):
            print(mk, "PARTIAL - not scored"); continue
        rows = []; nonev = Counter()
        for k in ids:
            r = R[k]; mp = {s: value(O[s][k], r) for s in ("mreq", "mex", "mkn")}
            for s, v in mp.items():
                nonev[s] += v is None
            ans = {"DIRECT": value(O["d"][k], r), "THINK": value(O["th"][k], r, True), "SLOW": value(O["slow"][k], r, True)}
            disp = len({v for v in mp.values() if v is not None}) > 1
            ok = {a: v == r["right"] for a, v in ans.items()}
            nov = {a: v is None for a, v in ans.items()}
            noans = {a: "answer:" not in O[s][k].lower() for a, s in (("THINK", "th"), ("SLOW", "slow"))}
            ok["SCHEME"] = ok["SLOW"] if disp else ok["DIRECT"]
            for s in mp:
                ok["M-" + s] = mp[s] == r["right"]
            rows.append(dict(id=k, group=r["group"], sub=r["sub"], disp=disp, ok=ok, nov=nov, noans=noans,
                             nmap=sum(v is not None for v in mp.values())))
        own = [x for x in rows if x["group"] in OWN]
        acc = lambda rr, a: pct([x["ok"][a] for x in rr])
        print(f"\n=== {mk} ({len(rows)} items: own {len(own)}, outside {len(rows) - len(own)}) ===")
        print("  right %          " + "".join(f"{g:>7s}" for g in OWN + OUT) + "   own all")
        for a in ("DIRECT", "THINK", "SLOW", "SCHEME", "M-mreq", "M-mex", "M-mkn"):
            print(f"  {a:14s}  " + "".join(f"{acc([x for x in rows if x['group'] == g], a):7.0f}" for g in OWN + OUT) + f"   {acc(own, a):6.1f}")
        print(f"  DISPUTE fires   " + "".join(f"{pct([x['disp'] for x in rows if x['group'] == g]):7.0f}" for g in OWN + OUT))
        print("  map answers with no value: " + ", ".join(f"{s} {v}" for s, v in nonev.items()) + f" (of {len(rows)} each)")
        print(f"  no value in answer: DIRECT {sum(x['nov']['DIRECT'] for x in rows)}, THINK {sum(x['nov']['THINK'] for x in rows)}, SLOW {sum(x['nov']['SLOW'] for x in rows)};"
              f" no 'Answer:' line: THINK {sum(x['noans']['THINK'] for x in rows)}, SLOW {sum(x['noans']['SLOW'] for x in rows)} (of {len(rows)})")
        sel = lambda rr: pct([x["disp"] for x in rr if x["group"] in ("G1", "G2", "G3")]) - pct([x["disp"] for x in rr if x["group"] == "G4"])
        s_lo, s_hi = boot(own, sel)
        calm = pct([x["disp"] for x in own if x["group"] == "G4"])
        speaks = pct([x["nmap"] >= 2 for x in own])   # v1.2 (review M3): P1 judged only if the map names >= 2 values on >= 70% of own items
        p1 = (s_lo >= 20 and calm <= 30) if speaks >= 70 else None
        print(f"  P1 map notices: fires on dispute groups - calm {sel(own):+.0f} [{s_lo:+.0f},{s_hi:+.0f}] (need CI low >= +20), calm {calm:.0f}% (need <= 30);"
              f" map names >= 2 values on {speaks:.0f}% -> {'n/a (map silent)' if p1 is None else 'PASS' if p1 else 'FAIL'}")
        g = {gg: acc([x for x in own if x["group"] == gg], "SLOW") for gg in ("G1", "G2", "G3")}
        print(f"  P2 slow step picks the right source: G1 {g['G1']:.0f} / G2 {g['G2']:.0f} / G3 {g['G3']:.0f} (need >= 60 each) -> {'PASS' if min(g.values()) >= 60 else 'FAIL'}")
        rnd = lambda rr: (lambda c: acc(rr, "SCHEME") - (acc(rr, "DIRECT") + c * (acc(rr, "SLOW") - acc(rr, "DIRECT"))))(sum(x["disp"] for x in rr) / len(rr))

        def rnd_g(rr):   # v1.2 (review M4, HEADLINE for P3): random re-check at the same rate WITHIN each group, summed
            bg = defaultdict(list)
            for x in rr:
                bg[x["group"]].append(x)
            return sum(len(v) * rnd(v) for v in bg.values()) / len(rr)
        r_lo, r_hi = boot(rows, rnd_g); a_lo, a_hi = boot(rows, rnd)
        outs = [x for x in rows if x["group"] in OUT]; o_lo, o_hi = boot(outs, rnd_g)
        wr = [x for x in rows if not x["ok"]["DIRECT"] and not x["nov"]["DIRECT"]]   # v1.2 (review M1): "no value" not counted
        testable = len(wr) >= 30 and pct([x["ok"]["SLOW"] for x in wr]) >= 100 / 3
        p3 = r_lo > 0 if testable else None
        print(f"  SCHEME {acc(rows, 'SCHEME'):.1f} | DIRECT {acc(rows, 'DIRECT'):.1f} | always SLOW {acc(rows, 'SLOW'):.1f} | always THINK {acc(rows, 'THINK'):.1f}"
              f" | slow-step rate {pct([x['disp'] for x in rows]):.0f}%  (all {len(rows)} items)")
        print(f"  random re-check at same overall rate (not headline): {rnd(rows):+.1f} [{a_lo:+.1f},{a_hi:+.1f}];"
              f" outside only, within group: {rnd_g(outs):+.1f} [{o_lo:+.1f},{o_hi:+.1f}]")
        print(f"  P3 SCHEME - random re-check at same rate within each group {rnd_g(rows):+.1f} [{r_lo:+.1f},{r_hi:+.1f}] -> "
              f"{'n/a (too few DIRECT errors SLOW can fix)' if p3 is None else 'PASS' if p3 else 'FAIL'}")
        for gg in OUT:
            rr = [x for x in rows if x["group"] == gg]
            print(f"  outside {gg}: DIRECT {acc(rr, 'DIRECT'):.0f}, SCHEME {acc(rr, 'SCHEME'):.0f}, always SLOW {acc(rr, 'SLOW'):.0f}, dispute {pct([x['disp'] for x in rr]):.0f}%;"
                  f" lamp on DIRECT errors {pct([x['disp'] for x in rr if not x['ok']['DIRECT']]):.0f}% / on DIRECT-right {pct([x['disp'] for x in rr if x['ok']['DIRECT']]):.0f}%")
        over += bool(p1 and p3); testn += p1 is not None and p3 is not None
    print(f"\nOVERALL: P1+P3 PASS in {over} of {testn} testable models (need 2; fewer than 2 testable = inconclusive)")


if __name__ == "__main__":
    main()
