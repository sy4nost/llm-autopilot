"""Check of the "conflict" guess, see PREDICT.md. CPU only, saved layer-32 snapshots.  modal run conflict_g.py"""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
img = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.3.4", "scikit-learn==1.7.2")
cache = modal.Volume.from_name("silo-hf-cache")
app = modal.App("silo-gurdjieff-conflict")
LAYERS = [8, 16, 24, 28, 32, 36, 40, 44]
FAMS = ["R", "N", "K", "D"]


@app.function(image=img, volumes={"/cache": cache}, cpu=8, memory=16384, timeout=3600)
def run(meta: list):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    d = "/cache/gurdjieff/main"
    feats = {}
    for l in open(f"{d}/feats.jsonl"):
        x = json.loads(l); feats[x["id"]] = x
    X = np.stack([np.load(f"{d}/h_{m['id']}.npy")[LAYERS.index(32)] for m in meta]).astype(np.float32)
    y = np.array([m["y"] for m in meta]); pool = np.array([m["pool"] for m in meta]); fam = np.array([m["fam"] for m in meta])
    pop = np.array([m["pop"] for m in meta])
    mg = np.array([(lambda s: s[0] - s[1])(sorted(feats[m["id"]]["sums"], reverse=True)) for m in meta])
    print("loaded", X.shape, flush=True)
    mu, sd = X[pool == "own"].mean(0), X[pool == "own"].std(0) + 1e-6
    Z = (X - mu) / sd
    auc = lambda s, t: float(roc_auc_score(t, s)) if 0 < t.sum() < len(t) else float("nan")

    def axis(fams):
        v = []
        for f in fams:
            tr = (pool == "own") & (fam == f) & (pop == f"{f}/traps"); ag = (pool == "own") & (fam == f) & (pop == f"{f}/agree")
            dv = Z[tr].mean(0) - Z[ag].mean(0); v.append(dv / np.linalg.norm(dv))
        a = np.mean(v, 0); return a / np.linalg.norm(a)
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.003, max_iter=3000, class_weight="balanced"))
    own = pool == "own"
    proj = np.zeros(len(y)); s_lofo = np.zeros(len(y)); s_rm = np.zeros(len(y)); cos = {}
    for f in FAMS:
        a = axis([g for g in FAMS if g != f])
        te = own & (fam == f); tr = own & (fam != f)
        proj[te] = Z[te] @ a
        m1 = mk().fit(Z[tr], y[tr]); s_lofo[te] = m1.decision_function(Z[te])
        w = m1[-1].coef_[0] / m1[0].scale_; cos[f] = float(w @ a / np.linalg.norm(w))
        Zr = Z - np.outer(Z @ a, a)  # same snapshots with the conflict direction removed
        print("family", f, "sensor done", flush=True)
        s_rm[te] = mk().fit(Zr[tr], y[tr]).decision_function(Zr[te])
    rows = []
    for p in sorted(set(pop[own])):
        t = own & (pop == p)
        if y[t].sum() >= 10 and (1 - y[t]).sum() >= 10:
            rows.append(dict(pop=p, n=int(t.sum()), flips=int(y[t].sum()), conflict=round(auc(proj[t], y[t]), 3),
                             sensor=round(auc(s_lofo[t], y[t]), 3), sensor_wo_conflict=round(auc(s_rm[t], y[t]), 3),
                             low_conf=round(auc(-mg[t], y[t]), 3)))
    # public: axis from all four families
    a = axis(FAMS); pub = pool == "pub"; prow = []
    for p in sorted(set(pop[pub])):
        t = pub & (pop == p)
        prow.append(dict(pop=p, n=int(t.sum()), flips=int(y[t].sum()), mean_proj=round(float((Z[t] @ a).mean()), 2),
                         conflict=round(auc(Z[t] @ a, y[t]), 3) if y[t].sum() >= 10 and (1 - y[t]).sum() >= 10 else None))
    ownmean = {p: round(float((Z[own & (pop == p)] @ a).mean()), 2) for p in sorted(set(pop[own]))}
    return dict(own=rows, cos=cos, pub=prow, own_mean_proj=ownmean)


@app.local_entrypoint()
def main():
    items = json.load(open(os.path.join(HERE, "..", "items_g.json")))["items"]
    meta = [dict(id=x["id"], y=x["y"], fam=x["fam"], pool=x["pool"], pop=x["pop"]) for x in items]
    out = run.remote(meta)
    json.dump(out, open(os.path.join(HERE, "conflict_out.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
