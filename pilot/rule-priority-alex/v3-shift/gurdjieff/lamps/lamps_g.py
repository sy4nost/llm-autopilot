"""Many lamps: one per trap population, families, panel. See PREDICT.md. CPU only, saved snapshots.  modal run lamps_g.py"""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
img = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.3.4", "scikit-learn==1.7.2")
cache = modal.Volume.from_name("silo-hf-cache")
app = modal.App("silo-gurdjieff-lamps")
LAYERS = [8, 16, 24, 28, 32, 36, 40, 44]
USE = [24, 32, 40]
UNITS = ["own:D/traps", "own:K/traps", "own:N/traps", "own:R/traps", "own:R/cfright", "own:R/compute",
         "pub:hind/trap", "pub:memo/trap", "pub:neqa/trap", "pub:unknown/trap", "pub:pattern/trap", "pub:mmlu/trap"]


@app.function(image=img, volumes={"/cache": cache}, cpu=8, memory=16384, timeout=3 * 3600)
def run(meta: list):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    d = "/cache/gurdjieff/main"
    feats = {}
    for l in open(f"{d}/feats.jsonl"):
        x = json.loads(l); feats[x["id"]] = x
    unit = np.array([f"{m['pool']}:{m['pop']}" for m in meta])
    keep = np.isin(unit, UNITS); meta = [m for m, k in zip(meta, keep) if k]; unit = unit[keep]
    H = {L: [] for L in USE}
    for m in meta:
        h = np.load(f"{d}/h_{m['id']}.npy")
        for L in USE:
            H[L].append(h[LAYERS.index(L)])
    H = {L: np.stack(v).astype(np.float32) for L, v in H.items()}
    print("loaded", len(meta), flush=True)
    y = np.array([m["y"] for m in meta]); clu = np.array([m["cluster"] for m in meta])
    mg = np.array([(lambda s: s[0] - s[1])(sorted(feats[m["id"]]["sums"], reverse=True)) for m in meta])

    def outf(m):
        dv = np.sort(np.array(feats[m["id"]]["dec"]))[::-1]; p = np.exp(dv - dv.max()); p /= p.sum()
        return [dv[0], dv[1], dv[0] - dv[1], -(p * np.log(p + 1e-12)).sum()]
    XO = np.array([outf(m) for m in meta])
    auc = lambda s, t: float(roc_auc_score(t, s))
    mk = lambda C=0.003: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
    rng = np.random.default_rng(0)
    idx = {u: np.where(unit == u)[0] for u in UNITS}
    # step 1
    s1 = []
    for u in UNITS:
        ii = idx[u]; yy = y[ii]; g = clu[ii]
        nf = min(5, len(set(g)))
        oof = {k: np.zeros(len(ii)) for k in [f"L{L}" for L in USE] + ["out"]}
        for tr, te in GroupKFold(nf).split(ii, yy, g):
            if len(set(yy[tr])) < 2:
                continue
            for L in USE:
                oof[f"L{L}"][te] = mk().fit(H[L][ii[tr]], yy[tr]).decision_function(H[L][ii[te]])
            oof["out"][te] = mk(1.0).fit(XO[ii[tr]], yy[tr]).decision_function(XO[ii[te]])
        lamp = oof["L32"]; low = -mg[ii]
        a = {k: auc(v, yy) for k, v in oof.items()}; a["low"] = auc(low, yy)
        best = max(a["low"], a["out"])
        cs = sorted(set(g)); gi = {c: np.where(g == c)[0] for c in cs}; bd = []
        for _ in range(1000):
            pick = np.concatenate([gi[cs[k]] for k in rng.integers(0, len(cs), len(cs))])
            if 0 < yy[pick].sum() < len(pick):
                bd.append(auc(lamp[pick], yy[pick]) - max(auc(low[pick], yy[pick]), auc(oof["out"][pick], yy[pick])))
        lo = float(np.percentile(bd, 2.5))
        ok = a["L32"] >= 0.70 and a["L32"] - best >= 0.05 and lo > 0
        s1.append(dict(unit=u, n=len(ii), flips=int(yy.sum()), lamp32=round(a["L32"], 3), lamp24=round(a["L24"], 3), lamp40=round(a["L40"], 3),
                       low_conf=round(a["low"], 3), output_only=round(a["out"], 3), diff_low95=round(lo, 3), has_lamp=bool(ok)))
        print("unit", u, s1[-1], flush=True)
    # step 2: full-unit lamps, transfer matrix and cosines
    full, W = {}, {}
    for u in UNITS:
        m = mk().fit(H[32][idx[u]], y[idx[u]]); full[u] = m
        w = m[-1].coef_[0] / m[0].scale_; W[u] = w / np.linalg.norm(w)
    T = {u: {v: (round(auc(full[u].decision_function(H[32][idx[v]]), y[idx[v]]), 3) if u != v else None) for v in UNITS} for u in UNITS}
    COS = {u: {v: round(float(W[u] @ W[v]), 3) for v in UNITS} for u in UNITS}
    fam = [[u, v] for i, u in enumerate(UNITS) for v in UNITS[i + 1:] if T[u][v] >= 0.65 and T[v][u] >= 0.65]
    # step 3: panel of the other lamps, combiner trained on other units
    def zs(u, v):  # lamp u scored on unit v, standardised within v
        s = full[u].decision_function(H[32][idx[v]]); return (s - s.mean()) / (s.std() + 1e-9)
    s3 = []
    for j in UNITS:
        others = [u for u in UNITS if u != j]
        Xtr, ytr = [], []
        for v in others:
            Xtr.append(np.stack([zs(u, v) if u != v else np.zeros(len(idx[v])) for u in others], 1)); ytr.append(y[idx[v]])
        comb = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(np.concatenate(Xtr), np.concatenate(ytr))
        Xte = np.stack([zs(u, j) for u in others], 1)
        s3.append(dict(unit=j, panel=round(auc(comb.decision_function(Xte), y[idx[j]]), 3), low_conf=round(auc(-mg[idx[j]], y[idx[j]]), 3)))
    mp = float(np.mean([r["panel"] for r in s3])); ml = float(np.mean([r["low_conf"] for r in s3]))
    return dict(step1=s1, transfer=T, cos=COS, families=fam, step3=s3, panel_mean=round(mp, 3), low_mean=round(ml, 3),
                panel_works=bool(mp >= 0.65 and mp >= ml + 0.03))


@app.local_entrypoint()
def main():
    items = json.load(open(os.path.join(HERE, "..", "items_g.json")))["items"]
    meta = [dict(id=x["id"], y=x["y"], pool=x["pool"], pop=x["pop"], cluster=x["cluster"]) for x in items]
    out = run.remote(meta)
    json.dump(out, open(os.path.join(HERE, "lamps_out.json"), "w"), ensure_ascii=False, indent=1)
    for r in out["step1"]:
        print(r)
    print("families", out["families"])
    for r in out["step3"]:
        print(r)
    print("panel mean", out["panel_mean"], "low mean", out["low_mean"], "works", out["panel_works"])
