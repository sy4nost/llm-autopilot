"""Post hoc (Alex 2026-10-24 00:38Z): rank all words of a task by surprise; where does the critical-span word land?
Uses score_sp.py's words/span/measures on saved sp_main_*.jsonl. python3 -I rank_sp.py > report_rank.txt"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_sp as S

data = S.load("main")
raw = {}
for mk in S.MODELS:
    for l in open(os.path.join(S.HERE, f"sp_main_{mk}.jsonl")):
        x = json.loads(l)
        if x.get("tok"):
            raw[(mk, x["id"])] = x["tok"]


def ranks(it, rows):
    W, text = S.words(it, rows)
    s = np.array([w[2] for w in W]); n = len(s); idx = np.arange(S.SKIP, n)
    zt = (s - s.mean()) / max(s.std(), 1e-6); zl = np.full(n, -np.inf)
    for t in idx:
        nb = np.r_[s[max(0, t - S.WIN):t], s[t + 1:t + 1 + S.WIN]]
        zl[t] = (s[t] - nb.mean()) / max(nb.std(), S.FLOOR) if len(nb) >= S.MINNB else zt[t]
    sp = S.span(it, text); inside = [t for t in idx if W[t][0] < sp[1] and W[t][1] > sp[0]]
    out = {}
    for name, v in (("RAW", s), ("ZLOC", zl), ("MIN", np.minimum(zl, zt))):
        order = sorted(idx, key=lambda t: -v[t]); pos = {t: k + 1 for k, t in enumerate(order)}
        out[name] = min(pos[t] for t in inside)   # best rank of any word in the span
    best = max(inside, key=lambda t: s[t])
    return out, len(idx), text[W[best][0]:W[best][1]]


print("Rank of the critical word among all words of the task (1 = biggest stumble). Median rank, share in top 1 / top 3 / top 10%.")
print("Span: D the stated number; redef the definition sentence; hindgen the outcome word; parcels/bonus the New RULE sentence.")
for mk in S.MODELS:
    for f in S.FAMS:
        for pop in ("trap", "control"):
            g = [r for r in data[mk] if r["it"]["family"] == f and r["it"]["pop"] == pop]
            if not g:
                continue
            res = [ranks(r["it"], raw[(mk, r["it"]["id"])]) for r in g]
            n = np.median([x[1] for x in res])
            line = f"{mk:7} {f:8} {pop:8} words~{n:5.0f} |"
            for k in ("RAW", "ZLOC", "MIN"):
                rk = np.array([x[0][k] for x in res]); nn = np.array([x[1] for x in res])
                line += f" {k}: med {np.median(rk):5.0f} top1 {np.mean(rk == 1):4.0%} top3 {np.mean(rk <= 3):4.0%} top10% {np.mean(rk <= np.maximum(1, 0.1 * nn)):4.0%} |"
            print(line)
print("\nTraps only: rank (RAW) of the critical word, fast habit error vs fast right (median; share top 3)")
for mk in S.MODELS:
    for f in S.FAMS:
        g = [r for r in data[mk] if r["it"]["family"] == f and r["it"]["pop"] == "trap"]
        P = [ranks(r["it"], raw[(mk, r["it"]["id"])])[0]["RAW"] for r in g if r["habit_err"]]
        N = [ranks(r["it"], raw[(mk, r["it"]["id"])])[0]["RAW"] for r in g if r["right"]]
        if len(P) >= 5 and len(N) >= 5:
            print(f"  {mk:7} {f:8} habit n={len(P):3} med {np.median(P):5.0f} top3 {np.mean(np.array(P) <= 3):4.0%} | right n={len(N):3} med {np.median(N):5.0f} top3 {np.mean(np.array(N) <= 3):4.0%}")
