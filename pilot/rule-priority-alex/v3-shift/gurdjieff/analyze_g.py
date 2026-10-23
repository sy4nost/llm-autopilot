"""Frozen analysis for the Gurdjieff sensor (PLAN.md v2 + v2.1). Called by modal_g.py on Modal CPU: run(meta, feats, H, LAYERS)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

B = 1000
CS = [0.003, 0.03, 0.3]
SHARE = 0.30
R4 = lambda v: round(float(v), 4)


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    p, n = s[y], s[~y]
    if not len(p) or not len(n):
        return float("nan")
    # average ranks for ties
    allv = np.concatenate([p, n]); order = np.argsort(allv, kind="mergesort"); ranks = np.empty(len(allv)); i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[order[j + 1]] == allv[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return float((ranks[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def groups_ok(idx, y, key, mn):
    g = {}
    for i in idx:
        g.setdefault(key[i], []).append(i)
    return {k: v for k, v in g.items() if sum(y[v]) >= mn and len(v) - sum(y[v]) >= mn}


def within_auc(s, y, groups):
    v = [auc(s[g], y[g]) for g in groups.values()]
    return float(np.mean(v)) if v else float("nan")


def fit_score(Xtr, ytr, Xte, C):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
    m.fit(Xtr, ytr)
    return m.decision_function(Xte)


def choose(Xs, y, tr, fam, pop):
    """leave-one-family-out inside the training pool; metric = within-population AUC of out-of-fold scores (min 10/10, else 5/5)"""
    best = None
    fams = sorted({fam[i] for i in tr})
    for name, X in Xs.items():
        for C in CS:
            oof = np.full(len(y), np.nan)
            for f in fams:
                te = [i for i in tr if fam[i] == f]; trn = [i for i in tr if fam[i] != f]
                if len(set(y[trn])) < 2:
                    continue
                oof[te] = fit_score(X[trn], y[trn], X[te], C)
            ok = [i for i in tr if not np.isnan(oof[i])]
            g = groups_ok(ok, y, pop, 10) or groups_ok(ok, y, pop, 5)
            a = within_auc(oof, y, g)
            if best is None or (not np.isnan(a) and (np.isnan(best[0]) or a > best[0])):
                best = (a, name, C)
    return best


def nmax(v):
    v = [x for x in v if not np.isnan(x)]
    return max(v) if v else float("nan")


def run(meta, feats, H, LAYERS, n_missing=0):
    out, SC = [], []
    P = lambda *a: out.append(" ".join(str(x) for x in a))
    N = len(meta)
    y = np.array([m["y"] for m in meta]); yp = np.array([m["yplus"] for m in meta])
    okd = np.array([m["ok_d"] for m in meta], float); okt = np.array([m["ok_t"] for m in meta], float)
    pool = np.array([m["pool"] for m in meta]); fam = np.array([f"{m['pool']}:{m['fam']}" for m in meta])
    pop = np.array([f"{m['pool']}:{m['pop']}" for m in meta]); cell = np.array([f"{m['pool']}:{m['cell']}" for m in meta])
    harm = np.array([m["harm"] for m in meta])
    clu = np.array([m["cluster"] for m in meta])
    P("## Sanity: recomputed forced-choice DIRECT pick == saved pick, per family (need >= 98% everywhere; items whose saved"
      " top-2 margin < 1e-3 are exact ties and not counted); also median |recomputed - saved| option score")
    sane = n_missing == 0
    P(f"  missing items {n_missing} (must be 0)")
    for f in sorted(set(fam)):
        ii = [i for i in range(N) if fam[i] == f]
        sv = lambda i: sorted(meta[i]["saved_direct"], reverse=True)
        jj = [i for i in ii if sv(i)[0] - sv(i)[1] >= 1e-3]
        a = np.mean([feats[meta[i]["id"]]["pick"] == int(np.argmax(meta[i]["saved_direct"])) for i in jj])
        dif = np.median([abs(x - z) for i in ii for x, z in zip(feats[meta[i]["id"]]["sums"], meta[i]["saved_direct"])])
        sane &= a >= 0.98
        P(f"  {f:14} n {len(ii):5}  ties {len(ii) - len(jj):3}  agree {a:.1%}  median score diff {dif:.4f}")
    P("SANITY", "PASS" if sane else "FAIL -> stop, nothing below is a verdict")

    def resample(pops, scheme):
        """item: items within population; cluster: clusters within population (run's cluster/grp; K/D by hook)"""
        o = {}
        for k, g in pops.items():
            g = np.asarray(g)
            if scheme == "item":
                o[k] = rng.choice(g, len(g))
            else:
                cs = {}
                for i in g:
                    cs.setdefault(clu[i], []).append(i)
                keys = list(cs); pick = rng.integers(0, len(keys), len(keys))
                o[k] = np.array([i for p in pick for i in cs[keys[p]]])
        return o
    # features
    dec = [np.sort(np.array(feats[m["id"]]["dec"]))[::-1] for m in meta]
    def outf(dv):
        p = np.exp(dv - dv.max()); p /= p.sum()
        return [dv[0], dv[1], dv[0] - dv[1], -(p * np.log(p + 1e-12)).sum()]
    XO = np.array([outf(d) for d in dec]); XS = np.array([m["surf"] for m in meta], float)
    sums = [sorted(feats[m["id"]]["sums"], reverse=True) for m in meta]
    margin = np.array([s[0] - s[1] for s in sums]); length = XS[:, 0]
    Xl = {f"L{L}": H[:, j, :] for j, L in enumerate(LAYERS)}
    rng = np.random.default_rng(0); jitter = rng.uniform(0, 1e-9, N)

    for split, trp, tep in (("S1 own->public", "own", "pub"), ("S2 public->own", "pub", "own")):
        tr = [i for i in range(N) if pool[i] == trp]; te = np.array([i for i in range(N) if pool[i] == tep])
        P(f"\n######## {split}: train {len(tr)}  test {len(te)}")
        best = choose(Xl, y, tr, fam, pop); P(f"  sensor: chose {best[1]} C={best[2]} (LOFO within-pop AUC in training pool {best[0]:.3f})")
        bo = choose({"out": XO}, y, tr, fam, pop); bs = choose({"surf": XS}, y, tr, fam, pop)
        S = {}
        S["sensor"] = fit_score(Xl[best[1]][tr], y[tr], Xl[best[1]], best[2])
        S["output-only"] = fit_score(XO[tr], y[tr], XO, bo[2]); S["surface"] = fit_score(XS[tr], y[tr], XS, bs[2])
        gtr = groups_ok(tr, y, pop, 10) or groups_ok(tr, y, pop, 5)
        for name, raw in (("low-confidence", -margin), ("length", length)):
            S[name] = raw if within_auc(raw, y, gtr) >= 0.5 else -raw  # sign fixed on the training pool
        RIV = ["low-confidence", "output-only", "surface", "length"]
        gte = groups_ok(te, y, pop, 10)
        P(f"  test populations with >= 10 flips and >= 10 non-flips: {len(gte)}")
        P(f"  {'population':26} {'n':>5} {'flips':>5} " + " ".join(f"{k[:11]:>11}" for k in ["sensor"] + RIV))
        for k, g in sorted(gte.items()):
            P(f"  {k:26} {len(g):5} {int(sum(y[g])):5} " + " ".join(f"{auc(S[n][g], y[g]):11.3f}" for n in ["sensor"] + RIV))
        # P1
        def p1(gs):
            a = within_auc(S["sensor"], y, gs); r = nmax([within_auc(S[n], y, gs) for n in RIV]); return a, a - r
        a0, d0 = p1(gte); los, his = [], []
        for scheme in ("item", "cluster"):
            bd = []
            for _ in range(B):
                gs = {k: v for k, v in resample(gte, scheme).items() if 0 < y[v].sum() < len(v)}
                bd.append(p1(gs)[1])
            bd = [x for x in bd if not np.isnan(x)] or [float("nan")]
            los.append(np.percentile(bd, 2.5)); his.append(np.percentile(bd, 97.5))
        lo, hi = min(los), max(his)
        ra = [within_auc(S[n], y, gte) for n in RIV]
        best_r = RIV[int(np.nanargmax(ra))] if not all(np.isnan(ra)) else RIV[0]
        c1 = R4(a0) >= 0.70 and R4(d0) >= 0.05 and R4(lo) > 0
        P(f"P1 within-population AUC sensor {a0:.3f}; minus best rival ({best_r} {within_auc(S[best_r], y, gte):.3f}) {d0:+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}] (wider of item/cluster) need >= 0.70 and >= +0.05, low > 0 -> {'PASS' if c1 else 'FAIL'}")
        # gates at the 30% test quantile
        tpops = {}
        for i in te:
            tpops.setdefault(pop[i], []).append(i)
        def gate(score, share=SHARE):
            s = score[te] + jitter[te]; t = np.quantile(s, 1 - share); g = np.zeros(N, bool); g[te] = s > t; return g
        def byfam(pops):  # families weighted equally (PLAN v1), items pooled inside a family
            f = {}
            for v in pops.values():
                f.setdefault(fam[v[0]], []).extend(list(v))
            return [np.array(v) for v in f.values()]
        def acc(g, pops):
            return float(np.mean([np.mean(np.where(g[v], okt[v], okd[v])) for v in byfam(pops)]))
        def acc_rand(pops, share=SHARE):
            return float(np.mean([np.mean(share * okt[v] + (1 - share) * okd[v]) for v in byfam(pops)]))
        G = {n: gate(S[n]) for n in S}
        P(f"  accuracy at 30% thinking (families weighted equally): DIRECT {100 * acc(np.zeros(N, bool), tpops):.1f}  "
          f"THINK {100 * acc(np.ones(N, bool), tpops):.1f}")
        P("   " + "  ".join(f"{n} {100 * acc(G[n], tpops):.1f}" for n in ["sensor"] + RIV) + f"  random {100 * acc_rand(tpops):.1f}")
        def p2(pops):
            a = acc(G["sensor"], pops); r = max([acc(G[n], pops) for n in RIV] + [acc_rand(pops)]); return a - r
        d2 = p2(tpops); los, his = [], []
        for scheme in ("item", "cluster"):
            bd = [p2(resample(tpops, scheme)) for _ in range(B)]
            los.append(np.percentile(bd, 2.5)); his.append(np.percentile(bd, 97.5))
        lo2, hi2 = min(los), max(his); c2 = R4(d2) >= 0.02 and R4(lo2) > 0
        P(f"P2 sensor gate - best rival gate {100 * d2:+.1f} [{100 * lo2:+.1f}, {100 * hi2:+.1f}] (wider of item/cluster) need >= +2, low > 0 -> {'PASS' if c2 else 'FAIL'}")
        P("   curve (sensor / low-confidence / random) at thinking share: " + "; ".join(
            f"{int(sh * 100)}% {100 * acc(gate(S['sensor'], sh), tpops):.1f}/{100 * acc(gate(S['low-confidence'], sh), tpops):.1f}/{100 * acc_rand(tpops, sh):.1f}"
            for sh in (0.1, 0.2, 0.3, 0.4)))
        # P3
        c3 = True
        for k, v in sorted(tpops.items()):
            if harm[v[0]]:
                dd = np.mean(np.where(G["sensor"][v], okt[v], okd[v])) - np.mean(okd[v]); c3 &= R4(dd) >= -0.03
                P(f"   harm {k:26} n {len(v):5} gate - DIRECT {100 * dd:+.1f}")
        P(f"P3 every harm population >= -3 points -> {'PASS' if c3 else 'FAIL'}")
        ok = c1 and c2 and c3 and sane
        if tep == "own":  # N1
            gc = groups_ok(te, y, cell, 5)
            a = within_auc(S["sensor"], y, gc); r = nmax([within_auc(S[n], y, gc) for n in RIV])
            n1 = bool(R4(a) >= 0.65 and (np.isnan(r) or a > r)) and len(gc) > 0
            P(f"N1 within-cell AUC on {len(gc)} cells: sensor {a:.3f} best rival {r:.3f} need >= 0.65 and > rival -> {'PASS' if n1 else 'FAIL'}")
            ok &= n1
        P(f"VERDICT {split}: {'internal lamp works (at a calibrated 30% share)' if ok else 'NOT SHOWN'}")
        P(f"  report: y+ (THINK fixes a wrong DIRECT) within-pop AUC sensor {within_auc(S['sensor'], yp, groups_ok(te, yp, pop, 5)):.3f} "
          f"low-confidence {within_auc(S['low-confidence'], yp, groups_ok(te, yp, pop, 5)):.3f}")
        for i in te:
            SC.append(dict(id=meta[i]["id"], split=split[:2], **{n: R4(S[n][i]) for n in S}, gate=bool(G["sensor"][i])))
    return "\n".join(out), SC
