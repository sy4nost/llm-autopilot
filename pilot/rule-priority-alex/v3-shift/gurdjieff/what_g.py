"""What did the own-trained sensor (S1: L32, C=0.003) learn? Post hoc, CPU only.  modal run what_g.py"""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
img = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.3.4", "scikit-learn==1.7.2")
cache = modal.Volume.from_name("silo-hf-cache")
app = modal.App("silo-gurdjieff-what")
LAYERS = [8, 16, 24, 28, 32, 36, 40, 44]


@app.function(image=img, volumes={"/cache": cache}, cpu=8, memory=16384, timeout=3600)
def what(meta: list):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    d = "/cache/gurdjieff/main"
    feats = {}
    for l in open(f"{d}/feats.jsonl"):
        x = json.loads(l); feats[x["id"]] = x
    own = [m for m in meta if m["pool"] == "own"]
    X = np.stack([np.load(f"{d}/h_{m['id']}.npy")[LAYERS.index(32)] for m in own]).astype(np.float32)
    y = np.array([m["y"] for m in own]); pop = np.array([m["pop"] for m in own]); fam = np.array([m["fam"] for m in own])
    okd = np.array([m["ok_d"] for m in own]); cell = np.array([m["cell"] for m in own])
    margin = np.array([sorted(feats[m["id"]]["sums"], reverse=True)[0] - sorted(feats[m["id"]]["sums"], reverse=True)[1] for m in own])

    def auc(s, t):
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(t, s)) if 0 < t.sum() < len(t) else float("nan")
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.003, max_iter=3000, class_weight="balanced"))
    full = mk().fit(X, y); s_in = full.decision_function(X)
    oof = np.zeros(len(y))  # leave-one-family-out scores (what it knows about a family it never saw)
    for f in sorted(set(fam)):
        t = fam == f; oof[t] = mk().fit(X[~t], y[~t]).decision_function(X[t])
    rows = []
    for p in sorted(set(pop)):
        t = pop == p
        rows.append(dict(pop=p, n=int(t.sum()), flips=int(y[t].sum()), flip_rate=round(float(y[t].mean()), 3),
                         mean_score=round(float(s_in[t].mean()), 3), auc_in=round(auc(s_in[t], y[t]), 3),
                         auc_lofo=round(auc(oof[t], y[t]), 3), auc_lowconf=round(auc(-margin[t], y[t]), 3),
                         corr_score_margin=round(float(np.corrcoef(s_in[t], margin[t])[0, 1]), 3),
                         auc_wrong_direct=round(auc(s_in[t], 1 - okd[t]), 3)))
    # how much of the score is "which population" vs "which item"
    tot = ((s_in - s_in.mean()) ** 2).sum(); within = sum(((s_in[pop == p] - s_in[pop == p].mean()) ** 2).sum() for p in set(pop))
    # same check with a sensor that knows only the population's flip rate
    pr = {p: y[pop == p].mean() for p in set(pop)}; base = np.array([pr[p] for p in pop])
    cells = []
    for c in sorted(set(cell)):
        t = cell == c
        if t.sum() >= 10:
            cells.append(dict(cell=c, n=int(t.sum()), flip_rate=round(float(y[t].mean()), 2), mean_score=round(float(s_in[t].mean()), 2)))
    return dict(rows=rows, pop_share=round(float(1 - within / tot), 3), auc_all_in=round(auc(s_in, y), 3),
                auc_all_poprate=round(auc(base, y), 3), auc_all_lofo=round(auc(oof, y), 3), cells=cells)


@app.local_entrypoint()
def main():
    items = json.load(open(os.path.join(HERE, "items_g.json")))["items"]
    meta = [dict(id=x["id"], y=x["y"], fam=x["fam"], pool=x["pool"], pop=x["pop"], cell=x["cell"], ok_d=x["ok_d"]) for x in items]
    out = what.remote(meta)
    json.dump(out, open(os.path.join(HERE, "what_own.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "cells"}, ensure_ascii=False, indent=1))
    for c in sorted(out["cells"], key=lambda c: -c["mean_score"]):
        print(c)
