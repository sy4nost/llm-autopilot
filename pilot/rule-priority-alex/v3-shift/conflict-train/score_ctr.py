"""Score conflict-train outputs (PLAN.md v0.4 criteria). Reads out/<tag>_<arm>.json for U and L (adapter name), fits the W baselines here.
  python3 score_ctr.py --tag main --L L1
K1/K3 pool = two-class held-out sets: H1, H3, X_CICM, X_HIND (X_WIKI negatives are off-topic, so X_WIKI is report-only). Score = macro balanced accuracy (mean over sets).
K2 plain false alarms = X_MMLU; decoys = H1/H3 'no' items.
Reported only: own dev T1-T4 (in-distribution), X_WC (real chats, kind partly trained), X_ODD (12 items), X_MC (all yes: recall),
AUC from p_yes. Unparsed output counts as 'no'. Bootstrap: 2000 resamples of clusters within each set, paired across arms
(clusters: CICM conversation pair, WIKI question, ODD section pair, otherwise item)."""
import json, os, sys, argparse, random, re
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ctr_prompt  # noqa: E402

POOL = ["H1", "H3", "X_CICM", "X_HIND"]
CUES = re.compile(r"\b(actually|instead|change|changed|update|updated|correction|from now on|scratch that|no longer|not|now|switch|new|"
                  r"anymore|rather than|i said|i meant)\b", re.I)


def load_items():
    xs = []
    for l in open(os.path.join(HERE, "own_items.jsonl")):
        x = json.loads(l); x["id"] = f"own{x['id']}"; x["set"] = x["family"]; xs.append(x)
    xs += [json.loads(l) for l in open(os.path.join(HERE, "ext_items.jsonl"))]
    xs += [json.loads(l) for l in open(os.path.join(HERE, "wc_items.jsonl"))]
    for x in xs:
        mt = x.get("meta", {})
        key = {"X_ODD": "section", "X_CICM": "id", "X_WIKI": "qid"}.get(x["set"])
        x["cl"] = f"{x['set']}:{mt[key] if key else x['id']}"
    return xs


def last_users(x, k=2):
    return "\n".join(m["content"] for m in [m for m in x["messages"] if m["role"] == "user"][-k:])


def W_preds(xs):
    """Word-only controls trained on the same train split. Returns {name: {id: (pred, score)}}."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    tr = [x for x in xs if x.get("split") == "train"]; ev = [x for x in xs if x.get("split") != "train"]
    y = [x["label"] for x in tr]; out = {}
    for name, view in (("W", lambda x: ctr_prompt.transcript(x["messages"])), ("Wlast", last_users)):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)
        clf = LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced").fit(vec.fit_transform([view(x) for x in tr]), y)
        p = clf.predict_proba(vec.transform([view(x) for x in ev]))[:, 1]
        out[name] = {x["id"]: (bool(v >= 0.5), float(v)) for x, v in zip(ev, p)}
    out["Wcue"] = {x["id"]: (bool(CUES.search(last_users(x, 1))), float(len(CUES.findall(last_users(x, 1))))) for x in ev}
    return out


def ba(rows):
    pos = [p for p, y in rows if y]; neg = [p for p, y in rows if not y]
    tpr = float(np.mean(pos)) if pos else np.nan; tnr = 1 - float(np.mean(neg)) if neg else np.nan
    return tpr, tnr, float(np.nanmean([tpr, tnr]))


def auc(rows):
    pos = np.array([s for s, y in rows if y]); neg = np.array([s for s, y in rows if not y])
    if not len(pos) or not len(neg):
        return np.nan
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def macro(by, pred, arm):
    return float(np.mean([ba([(pred[arm][x["id"]][0], x["label"]) for x in by[s]])[2] for s in POOL]))


def boot(by, pred, arms, B=2000, seed=0):
    rng = random.Random(seed); cls = {s: defaultdict(list) for s in POOL}
    for s in POOL:
        for x in by[s]:
            cls[s][x["cl"]].append(x)
    out = {a: [] for a in arms}
    for _ in range(B):
        smp = {}
        for s in POOL:
            keys = list(cls[s]); smp[s] = [x for k in (rng.choice(keys) for _ in keys) for x in cls[s][k]]
        for a in arms:
            out[a].append(macro(smp, pred, a))
    return {a: np.array(v) for a, v in out.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="main"); ap.add_argument("--L", default="L1")
    a = ap.parse_args(); xs = load_items(); byid = {x["id"]: x for x in xs}
    raw = {arm: json.load(open(os.path.join(HERE, "out", f"{a.tag}_{arm}.json"))) for arm in ("U", a.L)}
    pred = {arm: {i: (bool(ctr_prompt.parse(v["text"])), v["p_yes"]) for i, v in raw[arm].items()} for arm in raw}
    unparsed = {arm: sum(ctr_prompt.parse(v["text"]) is None for v in raw[arm].values()) for arm in raw}
    pred.update(W_preds(xs)); arms = ["U", a.L, "W", "Wlast", "Wcue"]
    ev = [byid[i] for i in raw["U"] if i in raw[a.L]]
    by = defaultdict(list)
    for x in ev:
        by[x["set"]].append(x)
    missing = [s for s in POOL + ["X_MMLU"] if not by[s]]
    assert not missing or a.tag.startswith("smoke"), f"sets not evaluated: {missing}"
    res = dict(tag=a.tag, n=len(ev), unparsed=unparsed, sets={})
    print(f"items {len(ev)}  unparsed {unparsed}\n{'set':8} {'n':>4} " + " ".join(f"{m:>17}" for m in arms) + "   (TPR/TNR/BA; AUC below)")
    for s in sorted(by):
        rows = by[s]; res["sets"][s] = {}; cells, aucs = [], []
        for m in arms:
            t = ba([(pred[m][x["id"]][0], x["label"]) for x in rows]); u = auc([(pred[m][x["id"]][1], x["label"]) for x in rows])
            res["sets"][s][m] = dict(tpr=t[0], tnr=t[1], ba=t[2], auc=u)
            cells.append("/".join("  - " if np.isnan(v) else f"{v:.2f}" for v in t)); aucs.append("" if np.isnan(u) else f"auc {u:.2f}")
        print(f"{s:8} {len(rows):4} " + " ".join(f"{c:>17}" for c in cells) + "\n" + " " * 14 + " ".join(f"{c:>17}" for c in aucs))
    if missing:
        print("smoke run: pool incomplete, no verdict"); return
    pt = {m: macro(by, pred, m) for m in arms}; bs = boot(by, pred, arms)
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    bestW = max(("W", "Wlast", "Wcue"), key=lambda m: pt[m])
    d_U, d_W = bs[a.L] - bs["U"], bs[a.L] - bs[bestW]
    fa_plain = float(np.mean([pred[a.L][x["id"]][0] for x in by["X_MMLU"]]))
    fa_dec = float(np.mean([pred[a.L][x["id"]][0] for s in ("H1", "H3") for x in by[s] if not x["label"]]))
    K1 = pt[a.L] >= 0.75 and pt[a.L] - pt["U"] >= 0.10 and ci(d_U)[0] > 0
    K2 = fa_plain <= 0.10 and fa_dec <= 0.25
    K3 = pt[a.L] - pt[bestW] >= 0.10 and ci(d_W)[0] > 0
    res.update(pool_macro_ba=pt, best_word_control=bestW, L_minus_U=(pt[a.L] - pt["U"], ci(d_U)), L_minus_W=(pt[a.L] - pt[bestW], ci(d_W)),
               fa_plain=fa_plain, fa_decoys=fa_dec, K1=bool(K1), K2=bool(K2), K3=bool(K3), PASS=bool(K1 and K2 and K3))
    print(f"\nheld-out pool {POOL}: macro BA  " + "  ".join(f"{m} {pt[m]:.3f}" for m in arms))
    print(f"L-U {pt[a.L]-pt['U']:+.3f} CI {ci(d_U)[0]:+.3f}..{ci(d_U)[1]:+.3f}   L-{bestW} {pt[a.L]-pt[bestW]:+.3f} CI {ci(d_W)[0]:+.3f}..{ci(d_W)[1]:+.3f}")
    print(f"L false alarms: plain {fa_plain:.3f}  H1/H3 decoys {fa_dec:.3f}")
    print(f"K1 {K1}  K2 {K2}  K3 {K3}  PASS {K1 and K2 and K3}")
    sens = {m: float(np.mean([ba([(pred[m][x["id"]][1] > 0.5, x["label"]) for x in by[s]])[2] for s in POOL])) for m in ("U", a.L)}
    print("sensitivity, verdict from p_yes > 0.5: macro BA " + "  ".join(f"{m} {v:.3f}" for m, v in sens.items()))
    print("far transfer (report only): X_HIND BA " + "  ".join(f"{m} {res['sets']['X_HIND'][m]['ba']:.3f}" for m in arms) +
          " | X_WIKI recall " + "  ".join(f"{m} {res['sets']['X_WIKI'][m]['tpr']:.3f}" for m in arms))
    res["p_yes_sensitivity"] = sens
    json.dump(res, open(os.path.join(HERE, "out", f"score_{a.tag}.json"), "w"), default=float, indent=1)


if __name__ == "__main__":
    main()
