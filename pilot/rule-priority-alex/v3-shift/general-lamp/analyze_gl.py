"""Frozen analysis for the general-lamp test (PLAN.md v2.1). Runs on Modal CPU next to the saved states (modal_gl.py::analyze),
or locally:  python3 -I analyze_gl.py --selftest   (random data, checks that everything runs).
Input: meta rows (one per scored prompt, see modal_gl.py) + feats(idx_list, layer) -> array of saved states.
Output: report text + per-item detector scores."""
import json, math, random
import numpy as np

LAYERS = [8, 12, 16, 20, 24, 28, 32, 36, 40, 44]
MASS_MIN = 0.5
C_REG = 0.1
B = 2000
MIN_CELL = 10
MIN_FAM = 30
DIRS = {
    "C->P": dict(train=["C-rule", "C-parcels", "C-bonus"], test=["P-redefine", "P-neqa", "P-quote"]),
    "P->C": dict(train=["P-redefine", "P-quote"], test=["C-rule", "C-parcels", "C-bonus", "P-neqa"]),
}
FAM_W = {"C-rule": 0.5, "C-parcels": 0.5}          # same structure: counted as one family where families are averaged


# ---------------------------------------------------------------- targets and filters
def softmax_lp(v):
    v = np.asarray(v, float); m = v.max(); return v - (m + math.log(np.exp(v - m).sum()))


def label_rows(meta):
    """adds: pick, y (habit error), use (passes filter + validity), cellkey, grp"""
    aux = {(r["family"], r["src_i"]): r for r in meta if r.get("aux")}
    rows = []
    for r in meta:
        if r.get("aux"):
            continue
        sc = np.asarray(r["scores"], float)
        pick = int(np.argmax(sc))
        mass = float(np.exp(sc).sum())
        r = dict(r, pick=pick, mass=mass, use=mass >= MASS_MIN)
        f = r["family"]
        if f.startswith("C-") or f == "P-native":
            r["y"] = int(r["cands"][pick] == r["old"] and r["old"] != r["key"])
            r["trap"] = int(r["old"] != r["key"])
        elif f == "mmlu":
            r["y"] = int(pick != r["gold"])
        else:
            r["y"] = int(pick != r["gold"])            # 2 options: the wrong one is the habitual one
            if f in ("P-neqa", "P-quote"):
                a = aux.get((f, r["src_i"]))
                if a is None:
                    r["use"] = False                    # NeQA without exactly one negation: excluded
                else:
                    asc = np.asarray(a["scores"], float)
                    # validity: model prefers the habitual (non-gold) option when the habit is the right answer
                    r["valid"] = bool(int(np.argmax(asc)) != r["gold"])
                    r["use"] = r["use"] and r["valid"]
        rows.append(r)
    return rows


# ---------------------------------------------------------------- metrics
def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks of ties
    ss = s[order]; i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def qualifying(rows):
    cells = {}
    for k, r in enumerate(rows):
        cells.setdefault(r["cell"], []).append(k)
    out = {}
    for c, ix in cells.items():
        y = np.array([rows[k]["y"] for k in ix])
        if y.sum() >= MIN_CELL and (len(y) - y.sum()) >= MIN_CELL:
            out[c] = ix
    return out


def fam_auc(rows, s, cells=None):
    """within-cell AUC of one family: mean over qualifying cells weighted by pairs; (auc, n_pos, n_neg, informative)"""
    cells = qualifying(rows) if cells is None else cells
    num = den = 0.0; npos = nneg = 0
    for c, ix in cells.items():
        y = np.array([rows[k]["y"] for k in ix]); a = auc(np.asarray(s)[ix], y)
        w = y.sum() * (len(y) - y.sum()); num += a * w; den += w; npos += y.sum(); nneg += len(y) - y.sum()
    return (num / den if den else float("nan")), int(npos), int(nneg), bool(npos >= MIN_FAM and nneg >= MIN_FAM)


# ---------------------------------------------------------------- detectors
def cell_center(X, rows):
    X = X.copy(); cells = {}
    for k, r in enumerate(rows):
        cells.setdefault((r["family"], r["cell"]), []).append(k)
    for ix in cells.values():
        X[ix] -= X[ix].mean(0)
    return X


def fam_weights(rows):
    cnt = {}
    for r in rows:
        cnt[(r["family"], r["y"])] = cnt.get((r["family"], r["y"]), 0) + 1
    return np.array([1.0 / (2 * cnt[(r["family"], r["y"])]) for r in rows])


class Lin:
    def __init__(self, X, y, w):
        from sklearn.linear_model import LogisticRegression
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.m = LogisticRegression(C=C_REG, max_iter=3000, solver="lbfgs")
        self.m.fit((X - self.mu) / self.sd, y, sample_weight=w * len(y) / w.sum())

    def score(self, X):
        return self.m.decision_function((X - self.mu) / self.sd)


def out_feats(rows):
    F = []
    for r in rows:
        lp = softmax_lp(r["scores"]); p = np.exp(lp); srt = np.sort(np.asarray(r["scores"], float))[::-1]
        F.append([p.max(), srt[0] - srt[1], -(p * lp).sum() / math.log(len(p)), math.log(max(r["mass"], 1e-12))])
    return np.array(F)


def margin(rows):
    return np.array([np.sort(np.asarray(r["scores"], float))[::-1][0] - np.sort(np.asarray(r["scores"], float))[::-1][1] for r in rows])


def by_family(rows, fams):
    return {f: [r for r in rows if r["family"] == f] for f in fams}


def mean_fams(vals):
    """vals: {family: auc}; equal family weights except FAM_W"""
    v = [(FAM_W.get(f, 1.0), a) for f, a in vals.items() if not math.isnan(a)]
    return sum(w * a for w, a in v) / sum(w for w, a in v) if v else float("nan")


def choose_layer(train, feats):
    """leave-one-family-out within training families; fallback: grouped 5-fold OOF on all training cells"""
    fams = sorted({r["family"] for r in train}); res = {}
    for L in LAYERS:
        X = cell_center(feats([r["row"] for r in train], L), train); folds = []
        for f in fams:
            tr = [k for k, r in enumerate(train) if r["family"] != f]; te = [k for k, r in enumerate(train) if r["family"] == f]
            te_rows = [train[k] for k in te]
            if not qualifying(te_rows) or len({train[k]["y"] for k in tr}) < 2:
                continue
            m = Lin(X[tr], np.array([train[k]["y"] for k in tr]), fam_weights([train[k] for k in tr]))
            folds.append(fam_auc(te_rows, m.score(X[te]))[0])
        res[L] = float(np.mean(folds)) if folds else float("nan")
    if all(math.isnan(v) for v in res.values()):
        for L in LAYERS:
            s = oof(train, feats, L); res[L] = mean_fams({f: fam_auc(x, s[[k for k, r in enumerate(train) if r["family"] == f]])[0]
                                                          for f, x in by_family(train, fams).items()})
        return max(res, key=lambda L: res[L]), res, "fallback-5fold"
    return max(res, key=lambda L: -1 if math.isnan(res[L]) else res[L]), res, "lofo"


def oof(train, feats, L):
    X = cell_center(feats([r["row"] for r in train], L), train)
    fold = np.array([hash_group(r["family"] + "/" + str(r["grp"])) % 5 for r in train]); s = np.zeros(len(train))
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        if len(te) == 0:
            continue
        m = Lin(X[tr], np.array([train[i]["y"] for i in tr]), fam_weights([train[i] for i in tr])); s[te] = m.score(X[te])
    return s


def hash_group(s):
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) % 1000003
    return h


# ---------------------------------------------------------------- bootstrap
def boot_dir(test_by_fam, scores, B=B, seed=0):
    """scores: {detector: {family: array}}. Returns {detector: [mean over families per replicate]}"""
    rng = np.random.default_rng(seed)
    plans = {}
    for f, rows in test_by_fam.items():
        if f == "P-redefine":
            strata = {}
            for k, r in enumerate(rows):
                strata.setdefault(r["cell"], []).append(k)
            plans[f] = ("strat", [np.array(v) for v in strata.values()])
        elif f.startswith("C-"):
            g = {}
            for k, r in enumerate(rows):
                g.setdefault(r["grp"], []).append(k)
            plans[f] = ("group", [np.array(v) for v in g.values()])
        else:
            plans[f] = ("item", np.arange(len(rows)))
    out = {d: [] for d in scores}
    for _ in range(B):
        idx = {}
        for f, (kind, p) in plans.items():
            if kind == "strat":
                idx[f] = np.concatenate([rng.choice(v, len(v)) for v in p])
            elif kind == "group":
                idx[f] = np.concatenate([p[j] for j in rng.integers(0, len(p), len(p))])
            else:
                idx[f] = rng.choice(p, len(p))
        for d in scores:
            vals = {}
            for f, rows in test_by_fam.items():
                sub = [rows[k] for k in idx[f]]
                vals[f] = fam_auc(sub, scores[d][f][idx[f]])[0]
            out[d].append(mean_fams(vals))
    return {d: np.array(v) for d, v in out.items()}


# ---------------------------------------------------------------- main
def run(meta, feats, l0feats, B_=B, log=print):
    rows = label_rows(meta)
    for k, r in enumerate(rows):
        r["row"] = r["row_index"]
    use = [r for r in rows if r["use"]]
    fams = sorted({r["family"] for r in use})
    log("## Families after filters (option mass >= %.1f, validity)" % MASS_MIN)
    for f in sorted({r["family"] for r in rows}):
        allf = [r for r in rows if r["family"] == f]; u = [r for r in use if r["family"] == f]
        if not allf:
            continue
        info = fam_auc(u, np.zeros(len(u))) if u else (float("nan"), 0, 0, False)
        log(f"  {f:11} scored {len(allf):5}  kept {len(u):5}  habit errors {sum(r['y'] for r in u):5}  "
            f"in qualifying cells: pos {info[1]} neg {info[2]}  informative {info[3]}  cells {len(qualifying(u)) if u else 0}")
    per_item = {}
    verdict = {}
    mm = [r for r in use if r["family"] == "mmlu"]
    mm_tr, mm_te = [r for r in mm if r["split"] == "train"], [r for r in mm if r["split"] == "heldout"]
    # B4: generic-error probe on MMLU
    b4L, b4res = None, {}
    for L in LAYERS:
        X = feats([r["row"] for r in mm_tr], L); y = np.array([r["y"] for r in mm_tr])
        fold = np.array([hash_group(r["subject"]) % 5 for r in mm_tr]); s = np.zeros(len(mm_tr))
        for k in range(5):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            s[te] = Lin(X[tr], y[tr], np.ones(len(tr))).score(X[te])
        b4res[L] = auc(s, y)
    b4L = max(b4res, key=lambda L: b4res[L])
    b4 = Lin(feats([r["row"] for r in mm_tr], b4L), np.array([r["y"] for r in mm_tr]), np.ones(len(mm_tr)))
    b4_valid = auc(b4.score(feats([r["row"] for r in mm_te], b4L)), [r["y"] for r in mm_te])
    log(f"\n## B4 generic-error probe (MMLU): layer {b4L}, CV AUC by layer {json.dumps({k: round(v, 3) for k, v in b4res.items()})}")
    log(f"  held-out MMLU AUC {b4_valid:.3f}  (validity needs >= 0.65) -> {'OK' if b4_valid >= 0.65 else 'WEAK (criterion 2 still applied)'}")
    for name, D in DIRS.items():
        log(f"\n######## {name}: train {D['train']} -> test {D['test']}")
        train = [r for r in use if r["family"] in D["train"]]
        test_by = by_family(use, D["test"])
        tr_info = {f: fam_auc(x, np.zeros(len(x)))[3] for f, x in by_family(train, D["train"]).items()}
        if name == "P->C" and not tr_info.get("P-quote", False):
            log("  P-quote not informative as a training family -> P->C NON-INFORMATIVE (pre-registered)")
            verdict[name] = "non-informative"
            continue
        L, lres, how = choose_layer(train, feats)
        log(f"  layer chosen {L} by {how}; held-out-family AUC by layer {json.dumps({k: round(v, 3) for k, v in lres.items()})}")
        Xtr = cell_center(feats([r["row"] for r in train], L), train); ytr = np.array([r["y"] for r in train])
        lamp = Lin(Xtr, ytr, fam_weights(train))
        lamp_unc = Lin(feats([r["row"] for r in train], L), ytr, fam_weights(train))
        X0 = cell_center(l0feats([r["row"] for r in train]), train); b3 = Lin(X0, ytr, fam_weights(train))
        b2 = Lin(cell_center(out_feats(train), train), ytr, fam_weights(train))
        sgn = 1.0 if mean_fams({f: fam_auc(x, margin(x))[0] for f, x in by_family(train, D["train"]).items()}) >= \
            mean_fams({f: fam_auc(x, -margin(x))[0] for f, x in by_family(train, D["train"]).items()}) else -1.0
        log(f"  B1 sign fitted on training: fires at {'LARGE' if sgn > 0 else 'SMALL'} margin")
        sc = {d: {} for d in ("lamp", "B1", "B2", "B3", "B4", "lamp_uncentred")}
        for f, x in test_by.items():
            if not x:
                continue
            Xt = cell_center(feats([r["row"] for r in x], L), x)
            sc["lamp"][f] = lamp.score(Xt)
            sc["lamp_uncentred"][f] = lamp_unc.score(feats([r["row"] for r in x], L))
            sc["B1"][f] = sgn * margin(x)
            sc["B2"][f] = b2.score(cell_center(out_feats(x), x))
            sc["B3"][f] = b3.score(cell_center(l0feats([r["row"] for r in x]), x))
            sc["B4"][f] = b4.score(feats([r["row"] for r in x], b4L))
            for k, r in enumerate(x):
                per_item.setdefault(r["uid"], {})[name] = {d: round(float(sc[d][f][k]), 4) for d in sc}
        inf = {f: fam_auc(x, sc["lamp"][f]) for f, x in test_by.items() if x}
        informative = [f for f, v in inf.items() if v[3]]
        log(f"  {'family':11}{'n_pos':>7}{'n_neg':>7}{'inform.':>9}" + "".join(f"{d:>16}" for d in sc))
        for f, x in test_by.items():
            if not x:
                log(f"  {f:11} (no rows)"); continue
            a = {d: fam_auc(x, sc[d][f])[0] for d in sc}
            log(f"  {f:11}{inf[f][1]:>7}{inf[f][2]:>7}{str(inf[f][3]):>9}" + "".join(f"{a[d]:>16.3f}" for d in sc))
        if len(informative) < 2:
            log(f"  informative test families {informative}: fewer than 2 -> {name} NON-INFORMATIVE")
            verdict[name] = "non-informative"
            continue
        tb = {f: test_by[f] for f in informative}
        bs = boot_dir(tb, {d: {f: sc[d][f] for f in informative} for d in sc}, B=B_)
        point = {d: mean_fams({f: fam_auc(tb[f], sc[d][f])[0] for f in informative}) for d in sc}
        ci = lambda v: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5))
        log(f"  mean within-cell AUC over informative families {informative}:")
        for d in sc:
            lo, hi = ci(bs[d]); log(f"    {d:15} {point[d]:.3f} [{lo:.3f}, {hi:.3f}]")
        c1 = point["lamp"] >= 0.65
        c2 = {}
        for d in ("B1", "B2", "B3", "B4"):
            diff = bs["lamp"] - bs[d]; lo, hi = ci(diff)
            c2[d] = (point["lamp"] - point[d] >= 0.05) and lo > 0
            log(f"    lamp - {d}: {point['lamp'] - point[d]:+.3f} [{lo:+.3f}, {hi:+.3f}]  need >= +0.05 and low > 0 -> {'PASS' if c2[d] else 'FAIL'}")
        c3 = all(inf[f][0] >= 0.55 for f in informative)
        # criterion 4: false alarms on MMLU (held-out, correct answers), threshold = 80% recall on out-of-fold training scores
        s_oof = oof(train, feats, L); pos = s_oof[ytr == 1]; thr = np.percentile(pos, 20)
        mu_pool = feats([r["row"] for r in train], L).mean(0)
        mm_ok = [r for r in mm_te if r["y"] == 0]
        fire = float((lamp.score(feats([r["row"] for r in mm_ok], L) - mu_pool) >= thr).mean()) if mm_ok else float("nan")
        c4 = fire <= 0.25
        log(f"  C1 AUC >= 0.65: {'PASS' if c1 else 'FAIL'} | C2 beats all baselines: {'PASS' if all(c2.values()) else 'FAIL'} "
            f"| C3 no family < 0.55: {'PASS' if c3 else 'FAIL'} | C4 MMLU false alarms {fire:.1%} <= 25%: {'PASS' if c4 else 'FAIL'}")
        verdict[name] = "pass" if (c1 and all(c2.values()) and c3 and c4) else "fail"
        # report only: without the 200 public items scored in earlier runs
        tb2 = {f: [r for r in x if not r.get("seen_before")] for f, x in tb.items()}
        log("  [report] without previously scored public items: " + ", ".join(
            f"{f} lamp {fam_auc(x, np.array([per_item[r['uid']][name]['lamp'] for r in x]))[0]:.3f}" for f, x in tb2.items() if x))
        # report only: pooled AUC vs a cell-identity predictor (cell's own habit rate)
        for f, x in test_by.items():
            if not x:
                continue
            rate = {}
            for r in x:
                rate.setdefault(r["cell"], []).append(r["y"])
            cellp = np.array([np.mean(rate[r["cell"]]) for r in x])
            log(f"  [report] {f}: pooled AUC lamp_uncentred {auc(sc['lamp_uncentred'][f], [r['y'] for r in x]):.3f}"
                f" vs cell-identity {auc(cellp, [r['y'] for r in x]):.3f}")
    # report only: C families -> P-native (twin pair), trap-situation control
    log("\n######## [report] C-families -> P-native (structural twin of C-rule)")
    train = [r for r in use if r["family"] in DIRS["C->P"]["train"]]; nat = [r for r in use if r["family"] == "P-native"]
    if nat:
        L, _, _ = choose_layer(train, feats)
        lamp = Lin(cell_center(feats([r["row"] for r in train], L), train), np.array([r["y"] for r in train]), fam_weights(train))
        s = lamp.score(cell_center(feats([r["row"] for r in nat], L), nat)); a = fam_auc(nat, s)
        log(f"  layer {L}: within-cell AUC {a[0]:.3f} (pos {a[1]}, neg {a[2]}, informative {a[3]}); B1 margin {fam_auc(nat, margin(nat))[0]:.3f}")
        tr2 = [r for r in train if r["family"] in ("C-parcels", "C-bonus")]
        tsit = Lin(feats([r["row"] for r in tr2], L), np.array([r["trap"] for r in tr2]), np.ones(len(tr2)))
        for f in DIRS["C->P"]["test"]:
            x = [r for r in use if r["family"] == f]
            if x:
                log(f"  trap-situation probe on {f}: within-cell AUC for habit errors {fam_auc(x, tsit.score(feats([r['row'] for r in x], L)))[0]:.3f}")
    log("\n######## PRE-REGISTERED VERDICT")
    v1, v2 = verdict.get("C->P"), verdict.get("P->C")
    if v1 == "non-informative":
        final = "NON-INFORMATIVE (not enough data in the main direction; no conclusion about the signal)"
    elif v1 == "pass" and v2 == "pass":
        final = "GENERAL SIGNAL FOUND (both directions pass)"
    elif v1 == "pass":
        final = "PARTIAL: a lamp learned on in-text copying also finds the training-memory habit (P->C %s)" % v2
    elif v2 == "pass":
        final = "PARTIAL: only the reverse direction passes"
    else:
        final = "NO general signal (C->P %s, P->C %s)" % (v1, v2)
    log(f"  C->P: {v1} | P->C: {v2} | OVERALL: {final}")
    return per_item


def selftest():
    rng = np.random.default_rng(1); meta = []; D = 64; store = {}
    def add(fam, cell, grp, y_habit, extra):
        i = len(meta); v = rng.normal(size=(len(LAYERS) + 1, D)); v[3, 0] += 1.5 * y_habit; store[i] = v
        sc = [-0.1, -3.0] if not y_habit else [-3.0, -0.1]
        r = dict(uid=f"u{i}", row_index=i, family=fam, cell=cell, grp=grp, scores=sc, **extra); meta.append(r)
    for fam in ("C-rule", "C-parcels", "C-bonus"):
        for g in range(40):
            for cell in ("a", "b"):
                y = int(rng.random() < 0.5); add(fam, cell, g, y, dict(cands=["X", "Y"], key="X", old="Y"))
    for fam, n in (("P-redefine", 300), ("P-neqa", 150), ("P-quote", 150)):
        for j in range(n):
            y = int(rng.random() < 0.4); add(fam, "t%d" % (j % 3) if fam == "P-redefine" else "all", j, y, dict(gold=0, src_i=j))
            if fam != "P-redefine":
                meta.append(dict(uid=f"a{j}{fam}", aux=True, family=fam, src_i=j, scores=[-3.0, -0.1]))
    for j in range(400):
        y = int(rng.random() < 0.3); add("mmlu", "all", j, y, dict(gold=0, split="train" if j < 300 else "heldout", subject="s%d" % (j % 7)))
    feats = lambda ix, L: np.array([store[i][LAYERS.index(L)] for i in ix])
    l0 = lambda ix: np.array([store[i][-1] for i in ix])
    run(meta, feats, l0, B_=50)


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
