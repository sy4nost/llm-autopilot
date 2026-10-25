"""lateflip scoring (PLAN.md). python3 -I score_lf.py TAG > report_TAG.txt
Q1 reuses surprise-lamp's frozen within-template AUC / clustered bootstrap code (score_sp.q1) with MAIN = SETTLE."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); CL = os.path.join(HERE, "..", "conftrap-lamp")
sys.path.insert(0, os.path.join(HERE, "..", "surprise-lamp"))
import score_sp as S

MODELS = ["qwen14", "phi4", "olmo13"]; FAMS = ["hindgen", "redef", "parcels", "bonus", "D"]


def leaders(lens, bias=None):
    """leader per layer; bias = per-layer per-option mean to subtract (token-prior debias, review B1); the last layer is the
    real answer and is never debiased"""
    out = []
    for l, r in enumerate(lens):
        r = np.array(r, float)
        if bias is not None and l < len(lens) - 1:
            r = r - bias[l]
        out.append(r)
    return out


def scores(lens, bias=None):
    R = leaders(lens, bias); L = len(R); fin = int(np.argmax(R[-1]))
    d = [l / (L - 1) for l in range(L)]                      # l = 0 is layer 1, l = L-1 is the last layer
    notlead = [R[l][fin] < R[l].max() for l in range(L)]    # a tie with the final answer counts as "final leads" (review B2)
    up = [l for l in range(L) if d[l] >= 0.5]
    late = [d[l] for l in up if notlead[l]]
    m = int(round(2 / 3 * (L - 1))); r = R[m]
    return dict(SETTLE=max(late) if late else 0.0, FRAC=float(np.mean([notlead[l] for l in up])),
                FLIP75=float(any(notlead[l] for l in up if d[l] < 0.75)),
                LEADMID=-float(r[fin] - np.max(np.delete(r, fin))), lead=[int(np.argmax(x)) for x in R],
                FINMARG=-float(np.sort(R[-1])[-1] - np.sort(R[-1])[-2]))


def load(tag):
    items = {x["id"]: x for x in json.load(open(os.path.join(CL, "sets_cl.json")))["items"]}
    data = {}
    for mk in MODELS:
        fn = os.path.join(HERE, f"lf_{tag}_{mk}.jsonl")
        if not os.path.exists(fn):
            continue
        fast = {}
        for l in open(os.path.join(CL, f"out_main_{mk}.jsonl")):
            x = json.loads(l)
            if "res" in x:
                lp = np.array([sum(o) for o in x["res"]["direct"]]); p = np.exp(lp - lp.max()); p /= p.sum()
                fast[x["id"]] = (int(lp.argmax()), float(p.max()))
        out = []; drop = 0; rows = []
        for l in open(fn):
            x = json.loads(l)
            if x["id"] == -1:
                print(mk, "checks:", x["checks"]); continue
            if x.get("drop"):
                drop += 1; continue
            rows.append(x)
        # token prior: mean lens log-prob per (family, token, layer) over items of OTHER clusters (leave-one-cluster-out)
        from collections import defaultdict
        acc = defaultdict(lambda: defaultdict(list))
        for x in rows:
            it = items[x["id"]]
            for k, t in enumerate(x["tokens"]):
                acc[(it["family"], t)][it["cluster"]].append([r[k] for r in x["lens"]])
        tot = {key: (np.sum([np.sum(v, 0) for v in cl.values()], 0), sum(len(v) for v in cl.values())) for key, cl in acc.items()}
        for x in rows:
            it = items[x["id"]]; bias = []
            for k, t in enumerate(x["tokens"]):
                S_, n_ = tot[(it["family"], t)]; own = acc[(it["family"], t)][it["cluster"]]
                n2 = n_ - len(own); bias.append((S_ - np.sum(own, 0)) / n2 if n2 > 0 else np.zeros(len(x["lens"])))
            x["bias"] = np.array(bias).T   # layers x options
        for x in rows:
            it = items[x["id"]]; a, pc = fast[x["id"]]; m = scores(x["lens"], x["bias"]); raw = scores(x["lens"])
            m["SETTLE_RAW"] = raw["SETTLE"]
            m["MAIN"] = m["SETTLE"]; m["LEN"] = float(len(it["prompt"].split())); m["CONF"] = 1 - pc
            m["lens_final"] = int(np.argmax(x["lens"][-1]))
            out.append(dict(it=it, m=m, ans=a, CONF=1 - pc, right=a == it["key"],
                            habit_err=it["habit"] is not None and a == it["habit"]))
        # TOKPRIOR (review B1): mean SETTLE of items of other clusters, same family, whose fast answer is the same option token
        for r, x in zip(out, rows):
            tk = x["tokens"][r["ans"]]
            v = [o["m"]["SETTLE"] for o, y in zip(out, rows) if o["it"]["family"] == r["it"]["family"]
                 and o["it"]["cluster"] != r["it"]["cluster"] and y["tokens"][o["ans"]] == tk]
            r["m"]["TOKPRIOR"] = float(np.mean(v)) if v else 0.0
        # CONF control (review S1): SETTLE minus its linear fit on CONF ranks within model x family x pop
        for f in FAMS:
            for pop in ("trap", "control"):
                g = [r for r in out if r["it"]["family"] == f and r["it"]["pop"] == pop]
                if len(g) > 3:
                    c = np.argsort(np.argsort([r["CONF"] for r in g])).astype(float); y = np.array([r["m"]["SETTLE"] for r in g])
                    A = np.c_[c, np.ones(len(c))]; beta = np.linalg.lstsq(A, y, rcond=None)[0]
                    for r, e in zip(g, y - A @ beta):
                        r["m"]["SETTLE_C"] = float(e)
        print(mk, "items", len(out), "dropped", drop, "lens final leader == DIRECT answer:",
              f"{np.mean([r['m']['lens_final'] == r['ans'] for r in out]):.1%}")
        data[mk] = out
    return data


RIVALS = ("CONF", "LEN", "TOKPRIOR", "FINMARG")


def q1(el, label):
    rng = np.random.default_rng(0)
    keys = ("MAIN",) + RIVALS
    pt = [{k: S.strat_auc(c["P"], c["N"], k)[0] for k in keys} for c in el]
    sg = [{k: np.sign(p[k] - 0.5) or 1.0 for k in p} for p in pt]
    st = {k: float(np.mean([abs(p[k] - 0.5) for p in pt])) for k in keys}
    diffs = {k: [] for k in RIVALS}
    for _ in range(S.NB):
        b = [S.boot_cell(c, keys, rng) for c in el]
        for k in RIVALS:
            diffs[k].append(np.nanmean([sg[i]["MAIN"] * (b[i]["MAIN"] - 0.5) - sg[i][k] * (b[i][k] - 0.5) for i in range(len(el))]))
    signs = [s_["MAIN"] for s_ in sg]; major = 1.0 if sum(x > 0 for x in signs) >= sum(x < 0 for x in signs) else -1.0
    cons = sum(x == major for x in signs); per = {}
    for c, p in zip(el, pt):
        per.setdefault(c["mk"], []).append(p["MAIN"] - 0.5)
    models_ok = sum(1 for mk, v in per.items() if (len(v) >= 2 or mk != "olmo13") and np.sign(np.mean(v)) == major)
    crit = {"strength >= 0.12": st["MAIN"] >= 0.12, "same direction >= 2/3 cells": cons >= 2 * len(el) / 3, "direction in >= 2 models": models_ok >= 2}
    for k in RIVALS:
        lo = np.percentile(diffs[k], 2.5); crit[f"beats {k} by >= 0.05, low > 0"] = st["MAIN"] - st[k] >= 0.05 and lo > 0
        print(f"  [{label}] SETTLE {st['MAIN']:.3f} vs {k} {st[k]:.3f}: diff {st['MAIN'] - st[k]:+.3f}, bootstrap low {lo:+.3f}")
    print(f"  [{label}] direction: {'later settle -> habit error' if major > 0 else 'later settle -> right'} in {cons}/{len(el)} cells, models agreeing {models_ok}")
    print(f"VERDICT Q1 [{label}]: {'PROMISING' if all(crit.values()) else 'no lamp here'}  {crit}")
    print(f"  [{label}] report-only strengths:", {k: round(float(np.nanmean([abs(S.strat_auc(c['P'], c['N'], k)[0] - 0.5) for c in el])), 3)
                                                for k in ("MAIN", "SETTLE_RAW", "SETTLE_C", "FRAC", "FLIP75", "LEADMID") + RIVALS})


def main(tag):
    data = load(tag)
    cells = []
    for mk, rs in data.items():
        for f in FAMS:
            t = [r for r in rs if r["it"]["family"] == f and r["it"]["pop"] == "trap"]
            P = [r for r in t if r["habit_err"]]; N = [r for r in t if r["right"]]
            a, npair, nt = S.strat_auc(P, N, "MAIN")
            cells.append(dict(mk=mk, f=f, P=P, N=N, pairs=npair, ntpl=nt, ok=npair >= S.MIN_PAIRS and nt >= S.MIN_TPL))
    print("\n== Q1 (primary): traps only, habit error (+) vs right (-), WITHIN-TEMPLATE AUC; > 0.5 = later settle -> habit error\n"
          "   MAIN = SETTLE on token-prior-debiased lens; SETTLE_RAW no debias; SETTLE_C = SETTLE with CONF regressed out")
    for c in cells:
        a = {k: S.strat_auc(c["P"], c["N"], k)[0] for k in ("MAIN", "SETTLE_RAW", "SETTLE_C", "FRAC", "CONF", "TOKPRIOR", "FINMARG")}
        print(f"  {c['mk']:7} {c['f']:8} habit {len(c['P']):3} right {len(c['N']):3} pairs {c['pairs']:5} tpl {c['ntpl']:2} "
              f"{'ELIGIBLE' if c['ok'] else 'skip    '} " + " ".join(f"{('SETTLE' if k == 'MAIN' else k)} {v:.2f}" for k, v in a.items()))
    el = [c for c in cells if c["ok"]]
    if el:
        q1(el, "all eligible cells")
        q1([c for c in el if c["f"] != "hindgen"], "without hindgen")
    else:
        print("VERDICT Q1: not testable")
    print("\n== SETTLE distribution (review S2): share at the maximum (last-but-one layer disagrees), share 0, median")
    for mk, rs in data.items():
        v = np.array([r["m"]["SETTLE"] for r in rs]); L = len(rs[0]["m"]["lead"])
        print(f"  {mk:7} at max {np.mean(v >= (L - 2) / (L - 1) - 1e-9):.0%}  zero {np.mean(v == 0):.0%}  median {np.median(v):.2f}")

    print("\n== Q2 (report): share with a late flip (SETTLE >= 0.75) — the 'smoke alarm' view")
    for mk, rs in data.items():
        for f in FAMS:
            g = lambda cond: [r for r in rs if r["it"]["family"] == f and cond(r)]
            parts = []
            for lab, cond in (("trap habit-err", lambda r: r["it"]["pop"] == "trap" and r["habit_err"]),
                              ("trap right", lambda r: r["it"]["pop"] == "trap" and r["right"]),
                              ("control", lambda r: r["it"]["pop"] == "control")):
                h = g(cond)
                parts.append(f"{lab} {100*np.mean([r['m']['SETTLE'] >= 0.75 for r in h]):3.0f}% (n {len(h)})" if h else f"{lab} -")
            print(f"  {mk:7} {f:8} " + " | ".join(parts))

    print("\n== Q3 (report): share of items whose CURRENT leader = the final answer, by depth (habit errors vs right, traps)")
    for mk, rs in data.items():
        for lab, cond in (("habit-err", lambda r: r["habit_err"]), ("right", lambda r: r["right"])):
            h = [r for r in rs if r["it"]["pop"] == "trap" and cond(r)]
            if not h:
                continue
            L = len(h[0]["m"]["lead"]); pts = [int(round(q * (L - 1))) for q in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)]
            print(f"  {mk:7} {lab:9} n {len(h):3} " + " ".join(f"d{q:.2f} {100*np.mean([r['m']['lead'][p] == r['m']['lead'][-1] for r in h]):3.0f}%"
                                                         for q, p in zip((0.5, 0.6, 0.7, 0.8, 0.9, 0.95), pts)))

    print("\n== confidence check: Spearman SETTLE vs CONF within traps")
    for mk, rs in data.items():
        out = []
        for f in FAMS:
            t = [r for r in rs if r["it"]["family"] == f and r["it"]["pop"] == "trap"]
            a = np.argsort(np.argsort([r["m"]["SETTLE"] for r in t])); b = np.argsort(np.argsort([r["CONF"] for r in t]))
            out.append(f"{f} {np.corrcoef(a, b)[0, 1]:+.2f}")
        print(f"  {mk:7} " + "  ".join(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
