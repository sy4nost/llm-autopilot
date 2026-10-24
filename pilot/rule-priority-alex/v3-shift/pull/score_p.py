"""pull scoring.  python3 score_p.py main
Per dialogue and variant: log P of each candidate (instruction / examples / truth number as the whole answer start).
log-odds of source s = logP(s) - logsumexp(logP(other two)).
Errors come from the saved greedy answers of the judge main run (../judge/out_main_<mk>.json, direct mode, judge scorer):
error = chosen source != right source (kind); 'other' counts as an error.
Bootstrap over (fact, seed)."""
import json, os, sys, math, random, importlib.util
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); JD = os.path.join(HERE, "..", "judge")
tag = sys.argv[1] if len(sys.argv) > 1 else "main"
sys.argv = [sys.argv[0], "main"]
spec = importlib.util.spec_from_file_location("sj", os.path.join(JD, "score_j.py")); sj = importlib.util.module_from_spec(spec); spec.loader.exec_module(sj)
items = {it["id"]: it for it in json.load(open(os.path.join(HERE, "sets_p.json")))["items"]}
MODELS = ["qwen14", "phi4", "olmo13"]; SRC = ["instruction", "examples", "truth"]
MASS, MINERR = 0.5, 15   # v2: analysis on dialogues where the 3 candidates hold >= half the answer mass; P2/P3 need >= 15 errors
FAV = {"qwen14": "instruction", "phi4": "examples", "olmo13": "examples"}   # from cycle3 rankings, fixed before data


def lse(xs):
    m = max(xs); return m + math.log(sum(math.exp(x - m) for x in xs))


def lodds(d, s):
    return d[s] - lse([d[o] for o in SRC if o != s])


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


def auc_rows(rows, key):
    return auc([key(r) for r in rows if r["err"]], [key(r) for r in rows if not r["err"]])


def main():
    for mk in MODELS:
        p = os.path.join(HERE, f"out_{tag}_{mk}.json"); g = os.path.join(JD, f"out_main_{mk}.json")
        if not os.path.exists(p):
            print(mk, "missing"); continue
        lp = json.load(open(p)); gen = json.load(open(g)); rows = []
        for sid in sorted({it["src_id"] for it in items.values()}):
            if f"{sid}|FULL" not in lp or str(sid) not in gen:
                continue
            it = items[f"{sid}|FULL"]; jit = sj.items[sid]
            comb = lambda d: {src: lse([d[f"{src}|bare"], d[f"{src}|sent"]]) for src in SRC} if d else None   # v2: bare or sentence
            F, NE, NI = comb(lp[f"{sid}|FULL"]), comb(lp.get(f"{sid}|NOEX")), comb(lp.get(f"{sid}|NOINS"))
            pick = sj.direct_pick(jit, gen[str(sid)])
            srt = sorted(F.values(), reverse=True)
            rows.append(dict(mass=math.exp(lse(list(F.values()))),
                             top_ne=max(SRC, key=lambda s: NE[s]) if NE else None, top_ni=max(SRC, key=lambda s: NI[s]) if NI else None,
                             kind=it["kind"], cl=(it["fact"], it["seed"]), pick=pick, err=pick != it["kind"],
                             argmax=max(SRC, key=lambda s: F[s]), fav=lodds(F, FAV[mk]), lowconf=-(srt[0] - srt[1]),
                             coarse=float(pick == FAV[mk]),
                             expull=lodds(F, "examples") - lodds(NE, "examples") if NE else None,
                             inspull=lodds(F, "instruction") - lodds(NI, "instruction") if NI else None))
        allr = rows; rows = [r for r in allr if r["mass"] >= MASS]
        print(f"\n=== {mk} ({len(allr)} dialogues, favourite = {FAV[mk]}, errors {sum(r['err'] for r in allr)}) ===")
        print(f"  coverage: 3-candidate mass >= {MASS} in {len(rows)} of {len(allr)} (errors kept {sum(r['err'] for r in rows)}); per kind "
              + ", ".join(f"{k} {sum(r['kind'] == k for r in rows)}" for k in SRC) + "  -> analysis below uses these")
        ok = [r for r in rows if r["pick"] in SRC]
        print(f"  P1 sanity: argmax of the 3 probabilities = greedy answer source in {100 * sum(r['argmax'] == r['pick'] for r in ok) / len(ok):.1f}% of {len(ok)}"
              + ("  PASS" if sum(r['argmax'] == r['pick'] for r in ok) >= .9 * len(ok) else "  FAIL"))
        print("  mean log-odds per kind (instruction / examples / truth number), FULL:")
        for k in SRC:
            rr = [r for r in rows if r["kind"] == k]
            print(f"    {k:12s} favourite log-odds mean {sum(r['fav'] for r in rr) / len(rr):+7.2f}   errors {sum(r['err'] for r in rr):3d}/{len(rr)}")
        cl = defaultdict(list)
        for r in rows:
            cl[r["cl"]].append(r)
        res = {}
        for name, key in (("fine lamp (favourite log-odds)", lambda r: r["fav"]), ("low confidence", lambda r: r["lowconf"]),
                          ("coarse lamp (answer = favourite)", lambda r: r["coarse"])):
            a = auc_rows(rows, key); lo, hi = boot(cl, lambda rr, key=key: auc_rows(rr, key)); res[name] = a
            print(f"  error AUC  {name:34s} {a:.3f} [{lo:.3f},{hi:.3f}]")
        a = res["fine lamp (favourite log-odds)"]; los = []
        for bname, bk in (("low confidence", "lowconf"), ("coarse lamp (answer = favourite)", "coarse")):
            d = lambda rr, bk=bk: auc_rows(rr, lambda r: r["fav"]) - auc_rows(rr, lambda r: r[bk])
            lo, hi = boot(cl, d); los.append(lo)
            print(f"  P2 fine lamp - {bname:32s} {a - res[bname]:+.3f} [{lo:+.3f},{hi:+.3f}]")
        ne = sum(r["err"] for r in rows)
        print("  P2 verdict: " + ("NOT TESTABLE (< %d errors)" % MINERR if ne < MINERR else
                                  "PASS" if a >= .75 and min(los) > 0 else "FAIL"))
        for k in SRC:
            rr = [r for r in rows if r["kind"] == k]
            if 0 < sum(r["err"] for r in rr) < len(rr):
                print(f"    within {k:12s} kind: fine {auc_rows(rr, lambda r: r['fav']):.3f}  lowconf {auc_rows(rr, lambda r: r['lowconf']):.3f}  (errors {sum(r['err'] for r in rr)})")
        rk = (lambda r: r["expull"]) if FAV[mk] == "examples" else (lambda r: r["inspull"])
        a3 = auc_rows(rows, rk); d = lambda rr: auc_rows(rr, rk) - auc_rows(rr, lambda r: r["fav"]); lo, hi = boot(cl, d)
        print(f"  P3 'close one eye' adds information: removal-drop lamp AUC {a3:.3f}, minus fine lamp {a3 - a:+.3f} [{lo:+.3f},{hi:+.3f}]  -> "
              + ("NOT TESTABLE" if ne < MINERR else "PASS" if lo > 0 else "FAIL"))
        print("  report: top answer after removal, per kind (share of dialogues where the top candidate changes):")
        for k in SRC:
            rr = [r for r in rows if r["kind"] == k]
            if rr:
                print(f"    {k:12s} n={len(rr):3d}  remove examples: changes {100 * sum(r['top_ne'] != r['argmax'] for r in rr) / len(rr):5.1f}%"
                      f"   remove instruction: changes {100 * sum(r['top_ni'] != r['argmax'] for r in rr) / len(rr):5.1f}%")
    print("\nOVERALL rule: P1 must PASS for all; P2 and P3 judged per model where testable (>= 15 errors).")


if __name__ == "__main__":
    main()
