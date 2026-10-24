"""goal2 scoring.  python3 score_g2.py main
Right answer = instruction number X. Answer source via ../judge/score_j.py direct_pick (items carry kind="examples" so
that Y->examples, X->instruction, M->truth). Goal sentence source = the single candidate number it names.
Bootstrap over (fact, seed)."""
import json, os, sys, random, importlib.util
from collections import defaultdict, Counter
HERE = os.path.dirname(os.path.abspath(__file__)); JD = os.path.join(HERE, "..", "judge")
tag = sys.argv[1] if len(sys.argv) > 1 else "main"
sys.argv = [sys.argv[0], "main"]
spec = importlib.util.spec_from_file_location("sj", os.path.join(JD, "score_j.py")); sj = importlib.util.module_from_spec(spec); spec.loader.exec_module(sj)
D = {it["id"]: it for it in json.load(open(os.path.join(HERE, "sets_d.json")))["items"]}
MODELS = ["qwen14", "phi4", "olmo13"]; PRIMARY = {"phi4", "olmo13"}; SRC = ["instruction", "examples", "truth"]; MINERR = 15


def goal_src(it, g):
    c = {it["X"]: "instruction", it["Y"]: "examples", it["M"]: "truth"}
    hit = {c[v] for v in sj.nums(g) if v in c}
    return hit.pop() if len(hit) == 1 else ("none" if not hit else "mixed")


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
        if len(ga) != 4 * len(gs):
            print(mk, f"PARTIAL output ({len(ga)} of {4 * len(gs)}) - not scored"); continue
        rows = []
        for k_ in gs:
            it = D[int(k_)]; pk = lambda a: sj.direct_pick(it, a)
            d, g, pb, rs = pk(ga[f"{k_}|DIRECT"]), pk(ga[f"{k_}|GOAL"]), pk(ga[f"{k_}|PLACEBO"]), pk(ga[f"{k_}|RESTATE"]); gsr = goal_src(it, gs[k_])
            rows.append(dict(N=it["N"], cl=(it["fact"], it["seed"]), d=d, g=g, pb=pb, dok=d == "instruction", gok=g == "instruction",
                             pok=pb == "instruction", rok=rs == "instruction", rs=rs, rflag=d != rs, flag=d != g, gsrc=gsr, copy=g == gsr, gtext=gs[k_]))
        cl = defaultdict(list)
        for r in rows:
            cl[r["cl"]].append(r)
        print(f"\n=== {mk} ({len(rows)} dialogues){'  [primary]' if mk in PRIMARY else '  [report only]'} ===")
        print("  answer source % (instruction=right / examples / truth / other) for DIRECT | PLACEBO | GOAL | RESTATE; goal sentence names; copy rate")
        for N in (1, 3):
            rr = [r for r in rows if r["N"] == N]
            f = lambda key: " / ".join(f"{pct([r[key] == s for r in rr]):3.0f}" for s in SRC) + f" / {pct([r[key] not in SRC for r in rr]):3.0f}"
            c = Counter(r["gsrc"] for r in rr)
            print(f"    N={N} n={len(rr)}  {f('d')} | {f('pb')} | {f('g')} | {f('rs')}   goal names " + " / ".join(f"{100 * c[s] / len(rr):.0f}" for s in SRC + ["none", "mixed"])
                  + f"   copy {pct([r['copy'] for r in rr]):.0f}")
        acc = lambda rr, k: pct([r[k] for r in rr])
        print(f"  H1 habit: DIRECT right {acc(rows, 'dok'):.1f}% (N=1 {acc([r for r in rows if r['N'] == 1], 'dok'):.1f}, N=3 {acc([r for r in rows if r['N'] == 3], 'dok'):.1f})"
              + ("  -> habit present" if acc([r for r in rows if r['N'] == 3], 'dok') <= 50 else "  -> little habit"))
        los = {}
        for base, bk in (("PLACEBO", "pok"), ("DIRECT", "dok"), ("RESTATE", "rok")):
            f = lambda rr, bk=bk: acc(rr, "gok") - acc(rr, bk); lo, hi = boot(cl, f); los[base] = lo
            print(f"  GOAL - {base:8s} {f(rows):+.1f} [{lo:+.1f},{hi:+.1f}]")
        if mk in PRIMARY:
            print("  H2 (primary, GOAL beats PLACEBO and DIRECT): " + ("PASS" if los["PLACEBO"] > 0 and los["DIRECT"] > 0 else "FAIL")
                  + "   | own goal reading vs restated instruction (GOAL - RESTATE): see line above")
        ok = [r for r in rows if r["dok"]]
        if ok:
            br = pct([not r["gok"] for r in ok])
            print(f"  H3 harm: GOAL breaks {br:.1f}% of {len(ok)} DIRECT-right (PLACEBO {pct([not r['pok'] for r in ok]):.1f}%)"
                  + (("  -> PASS" if br <= 5 else "  -> FAIL") if mk in PRIMARY and len(ok) >= 20 else "  (few right answers, report)"))
        ne = sum(not r["dok"] for r in rows)
        Af = lambda rr, key: auc([float(r[key]) for r in rr if not r["dok"]], [float(r[key]) for r in rr if r["dok"]])
        nx = [r for r in rows if r["gsrc"] != "instruction"]
        print(f"  H4 lamp (descriptive) 'quick != with goal': catches {pct([r['flag'] for r in rows if not r['dok']]):.1f}% of {ne} errors, false alarms "
              f"{pct([r['flag'] for r in rows if r['dok']]):.1f}% of {len(ok)} right; AUC {Af(rows, 'flag'):.3f}; RESTATE lamp AUC {Af(rows, 'rflag'):.3f};"
              f" on {len(nx)} items whose goal did not name X: AUC {Af(nx, 'flag'):.3f}")
        for r in [r for r in rows if r["N"] == 3][:3]:
            print(f"    e.g. N=3 goal \"{r['gtext'][:120]}\" -> {r['g']}")


if __name__ == "__main__":
    main()
