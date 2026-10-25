"""goal scoring.  python3 score_g.py main
DIRECT = saved greedy answers of the judge main run; GOAL = answers with the model's own goal sentence (step 2).
Answer source via ../judge/score_j.py direct_pick. Goal sentence source = the single candidate number it names
(none / several -> 'none' / 'mixed'). Low confidence (baseline lamp) from ../pull/out_main_<mk>.json (FULL).
Bootstrap over (fact, seed)."""
import json, os, sys, math, random, re, importlib.util
from collections import defaultdict, Counter
HERE = os.path.dirname(os.path.abspath(__file__)); JD = os.path.join(HERE, "..", "judge"); PD = os.path.join(HERE, "..", "pull")
tag = sys.argv[1] if len(sys.argv) > 1 else "main"
sys.argv = [sys.argv[0], "main"]
spec = importlib.util.spec_from_file_location("sj", os.path.join(JD, "score_j.py")); sj = importlib.util.module_from_spec(spec); spec.loader.exec_module(sj)
MODELS = ["qwen14", "phi4", "olmo13"]; SRC = ["instruction", "examples", "truth"]
WEAK = {"qwen14": "examples", "phi4": "truth", "olmo13": "truth"}   # fixed in ../judge PLAN (J3) before any goal data
MINERR = 15


def cands(it):
    k = it["kind"]
    return {(it["X2"] if k == "instruction" else it["X"]): "instruction", (it["X"] if k == "instruction" else it["Y"]): "examples", it["M"]: "truth"}


def goal_src(it, g):
    hit = {cands(it)[v] for v in sj.nums(g) if v in cands(it)}
    return hit.pop() if len(hit) == 1 else ("none" if not hit else "mixed")


def lse(xs):
    m = max(xs); return m + math.log(sum(math.exp(x - m) for x in xs))


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
    for mk in MODELS:
        p = os.path.join(HERE, f"out_{tag}_g2_{mk}.json")
        if not os.path.exists(p):
            print(mk, "missing"); continue
        ga = json.load(open(p)); gs = {str(i["src"]): i["goal"] for i in json.load(open(os.path.join(HERE, f"sets_g2_{tag}_{mk}.json")))["items"]}
        if len(ga) != 3 * len(gs):
            print(mk, f"PARTIAL output ({len(ga)} of {3 * len(gs)}) - not scored"); continue
        saved = json.load(open(os.path.join(JD, f"out_main_{mk}.json")))
        pl = json.load(open(os.path.join(PD, f"out_main_{mk}.json"))) if os.path.exists(os.path.join(PD, f"out_main_{mk}.json")) else {}
        rows = []
        for k_ in gs:
            it = sj.items[int(k_)]; pk = lambda a: sj.direct_pick(it, a)
            d, g, pb, sv = pk(ga[f"{k_}|DIRECT"]), pk(ga[f"{k_}|GOAL"]), pk(ga[f"{k_}|PLACEBO"]), pk(saved[k_])
            lc = None
            if f"{k_}|FULL" in pl:
                F = pl[f"{k_}|FULL"]; v = sorted((lse([F[s + "|bare"], F[s + "|sent"]]) for s in SRC), reverse=True); lc = -(v[0] - v[1])
            gsr = goal_src(it, gs[k_])
            rows.append(dict(kind=it["kind"], cl=(it["fact"], it["seed"]), d=d, g=g, pb=pb, sv=sv, dok=d == it["kind"], gok=g == it["kind"],
                             pok=pb == it["kind"], flag=d != g, pflag=d != pb, lc=lc, gsrc=gsr, gsok=gsr == it["kind"], copy=g == gsr, gtext=gs[k_]))
        cl = defaultdict(list)
        for r in rows:
            cl[r["cl"]].append(r)
        print(f"\n=== {mk} ({len(rows)} dialogues) ===")
        print(f"  DIRECT re-run differs from the saved judge DIRECT in {pct([r['d'] != r['sv'] for r in rows]):.1f}% (baseline below = re-run)")
        print("  accuracy % DIRECT / PLACEBO / GOAL;  goal sentence: right % and copy rate % (answer = what the goal names);")
        print("  goal sentence names instruction / examples / truth / none / mixed %")
        for k in SRC:
            rr = [r for r in rows if r["kind"] == k]; c = Counter(r["gsrc"] for r in rr)
            print(f"    {k:12s} n={len(rr):3d}  {pct([r['dok'] for r in rr]):5.1f} / {pct([r['pok'] for r in rr]):5.1f} / {pct([r['gok'] for r in rr]):5.1f}"
                  f"   goal right {pct([r['gsok'] for r in rr]):5.1f} copy {pct([r['copy'] for r in rr]):5.1f}   names "
                  + " / ".join(f"{100 * c[s] / len(rr):.0f}" for s in SRC + ["none", "mixed"]))
        w = WEAK[mk]
        for base, bk in (("PLACEBO", "pok"), ("DIRECT", "dok")):
            f = lambda rr, bk=bk: pct([r["gok"] for r in rr if r["kind"] == w]) - pct([r[bk] for r in rr if r["kind"] == w])
            lo, hi = boot(cl, f); v = f(rows)
            tagp = "G1 (primary) " if base == "PLACEBO" else "G1b         "
            print(f"  {tagp}fix on weak kind ({w}): GOAL - {base} {v:+.1f} [{lo:+.1f},{hi:+.1f}]  -> " + ("PASS" if lo > 0 else "FAIL")
                  + ("  (low power: little room)" if mk == "phi4" else ""))
        for base, bk in (("PLACEBO", "pok"), ("DIRECT", "dok")):
            f = lambda rr, bk=bk: pct([r["gok"] for r in rr]) - pct([r[bk] for r in rr]); lo, hi = boot(cl, f)
            print(f"  overall GOAL - {base} {f(rows):+.1f} [{lo:+.1f},{hi:+.1f}]")
        ok = [r for r in rows if r["dok"]]; br = pct([not r["gok"] for r in ok])
        lo, hi = boot(cl, lambda rr: pct([not r["gok"] for r in rr if r["dok"]]))
        print(f"  G2 harm: GOAL breaks {br:.1f}% [{lo:.1f},{hi:.1f}] of DIRECT-correct answers (PLACEBO breaks {pct([not r['pok'] for r in ok]):.1f}%)  -> "
              + ("PASS" if br <= 5 else "FAIL"))
        ne = sum(not r["dok"] for r in rows)
        A = lambda rr, key: auc([key(r) for r in rr if not r["dok"]], [key(r) for r in rr if r["dok"]])
        af = A(rows, lambda r: float(r["flag"]))
        print(f"  lamp 'quick != with goal': fires {pct([r['flag'] for r in rows]):.1f}%, catches {pct([r['flag'] for r in rows if not r['dok']]):.1f}% of "
              f"{ne} DIRECT errors, false alarms {pct([r['flag'] for r in rows if r['dok']]):.1f}% of correct; AUC {af:.3f}"
              f"  (placebo lamp AUC {A(rows, lambda r: float(r['pflag'])):.3f})")
        rl = [r for r in rows if r["lc"] is not None]
        if rl:
            al = A(rl, lambda r: r["lc"]); d = lambda rr: A(rr, lambda r: float(r["flag"])) - A(rr, lambda r: r["lc"])
            cl2 = defaultdict(list)
            for r in rl:
                cl2[r["cl"]].append(r)
            lo, hi = boot(cl2, d)
            print(f"  G3 lamp AUC minus low confidence ({al:.3f}): {A(rl, lambda r: float(r['flag'])) - al:+.3f} [{lo:+.3f},{hi:+.3f}]  -> "
                  + ("NOT TESTABLE (< 15 errors)" if ne < MINERR else "PASS" if af >= .75 and lo > 0 else "FAIL"))
        for r in [r for r in rows if r["kind"] == w][:3]:
            print(f"    e.g. {w}: goal \"{r['gtext'][:110]}\" -> {r['g']}")


if __name__ == "__main__":
    main()
