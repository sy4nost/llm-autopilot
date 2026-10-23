"""Offline trained lamp on saved 14B readings (no new runs).  python3 -I probe_lamp.py lamp_Qwen2.5-14B-Instruct.jsonl
Target: habit error = model's answer is the old label while the right one differs.
Features are label-agnostic, taken relative to the label the model picked: margins of final / heads-off / heads-push / lens 20-29,
and whether heads-off and heads-push agree with the pick. Logistic regression trained on sets 0-14, scored on held-out 15-29."""
import json, sys
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
am = lambda v: int(np.argmax(v))
def marg(v, p):
    v = np.asarray(v, float); o = np.delete(v, p); return v[p] - o.max()
def feats(r):
    p = am(r["final"])
    f = [marg(r["final"], p), marg(r["off"], p), marg(r["push"], p), float(am(r["off"]) == p), float(am(r["push"]) == p)]
    f += [marg(r["lens"][L], p) for L in ("20", "24", "28", "29")]
    return f
y_of = lambda r: float(r["labels"][am(r["final"])] == r["old"] and r["old"] != r["key"])
tr = [r for r in rows if int(r["set"]) < 15]; te = [r for r in rows if int(r["set"]) >= 15]
X = np.array([feats(r) for r in tr]); Y = np.array([y_of(r) for r in tr])
mu, sd = X.mean(0), X.std(0) + 1e-9
Z = lambda A: np.c_[(A - mu) / sd, np.ones(len(A))]
def fit(X, Y, l2=1.0, it=3000, lr=0.1):
    w = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ w)); w -= lr * (X.T @ (p - Y) / len(Y) + l2 * w / len(Y))
    return w
def auc(s, y):
    s, y = np.asarray(s), np.asarray(y); pos, neg = s[y == 1], s[y == 0]
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())) if len(pos) and len(neg) else float("nan")
Xt = np.array([feats(r) for r in te]); Yt = np.array([y_of(r) for r in te])
names = ["final", "off", "push", "off_agrees", "push_agrees", "lens20", "lens24", "lens28", "lens29"]
print(f"habit errors: train {int(Y.sum())}/{len(Y)}, held-out {int(Yt.sum())}/{len(Yt)}")
w = fit(Z(X), Y); s = Z(Xt) @ w
print(f"all features   held-out AUC {auc(s, Yt):.2f}")
for k, n in enumerate(names):
    wk = fit(Z(X)[:, [k, -1]], Y); print(f"  only {n:12} AUC {auc(Z(Xt)[:, [k, -1]] @ wk, Yt):.2f}")
# lamp at threshold 0.5: fire rate per group; brake = take heads-off answer when lamp fires
p = 1 / (1 + np.exp(-s)); on = p > 0.5
G = {"chg D1+D3 (trap)": lambda r: "_change" in r["arm"] and r["probe"] in ("D1", "D3"),
     "chg D2+D4": lambda r: "_change" in r["arm"] and r["probe"] in ("D2", "D4"),
     "nochange": lambda r: "nochange" in r["arm"]}
for g, f in G.items():
    idx = [i for i, r in enumerate(te) if f(r)]
    right_n = np.mean([te[i]["labels"][am(te[i]["final"])] == te[i]["key"] for i in idx])
    right_l = np.mean([te[i]["labels"][am(te[i]["off"] if on[i] else te[i]["final"])] == te[i]["key"] for i in idx])
    print(f"  {g:18} lamp on {on[idx].mean():4.0%}  right: normal {right_n:4.0%} -> lamp+brake {right_l:4.0%}")
print(f"  all                right: normal {np.mean([r['labels'][am(r['final'])]==r['key'] for r in te]):.0%} -> lamp+brake {np.mean([te[i]['labels'][am(te[i]['off'] if on[i] else te[i]['final'])]==te[i]['key'] for i in range(len(te))]):.0%}")
