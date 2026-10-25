"""Scoring for surprise-lamp, PLAN v2 (after review/plan-review.md). python3 -I score_sp.py TAG > report_TAG.txt
Joins sp_<TAG>_<model>.jsonl (per-token surprise) with ../conftrap-lamp/sets_cl.json and out_main_<model>.jsonl (fast answers)."""
import json, os, sys, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); CL = os.path.join(HERE, "..", "conftrap-lamp")
MODELS = ["qwen14", "phi4", "olmo13"]; FAMS = ["hindgen", "redef", "parcels", "bonus", "D"]
SKIP, WIN, FLOOR, MINNB = 1, 10, 0.5, 6       # words: skip the first, +-10 neighbour words, sd floor 0.5 nats, >= 6 neighbours
MIN_PAIRS, MIN_TPL = 30, 2; NB = 2000
R_TAIL = "\n\nReply with only the answer, nothing else."
FIXED = ["Answer for the TEST case below. Do not explain. Return only the answer.", "\n\nReturn only the answer."]
SCORES = ["MAIN", "ZLOC", "ZTASK", "RAW", "SPAN", "LEN"]


def words(it, rows):
    """whitespace words of the task (tokens with no whitespace between them = one word), surprise summed over its tokens.
    Left out: the fixed answer-format lines and R_TAIL (identical in every item), words with no letter or digit."""
    p = it["prompt"]; text = p + R_TAIL; bad = [(len(p), 10**9)]
    for f in FIXED:
        a = p.rfind(f)
        if a >= 0:
            bad.append((a, a + len(f)))
    W = [[m.start(), m.end(), 0.0, 0] for m in re.finditer(r"\S+", text)]
    starts = [w[0] for w in W]
    import bisect
    for st, en, sv, _ in rows:   # a token belongs to the word holding its first non-space character; pure-space tokens dropped
        piece = text[st:en]; k = len(piece) - len(piece.lstrip())
        if k == len(piece):
            continue
        j = bisect.bisect_right(starts, st + k) - 1
        if j >= 0 and W[j][0] <= st + k < W[j][1]:
            W[j][2] += sv; W[j][3] += 1
    out = [(st, en, sv) for st, en, sv, n in W if n and any(ch.isalnum() for ch in text[st:en])
           and not any(st < b and en > a for a, b in bad)]
    return out, text


def span(it, text):
    """critical span (char range) fixed per family before the run (review B4)"""
    p, f = it["prompt"], it["family"]
    if f == "D":   # the stated count: "has N <part>" in the first sentence
        m = re.search(r" has (\d+) ", p); return m.span(1)
    if f == "redef":   # the redefinition sentence(s) before "Q:"
        return 0, p.index(" Q:")
    if f == "hindgen":   # the outcome word of the test question
        a = p.rindex("ends up ") + len("ends up "); return a, p.index(" ", a)
    if f in ("parcels", "bonus"):   # the new-rule sentence
        a = p.index("New "); return a, p.index("All other", a)
    return None


def measures(it, rows):
    W, text = words(it, rows)
    s = np.array([w[2] for w in W], float); idx = np.arange(SKIP, len(s))
    if len(idx) < 3:
        return None
    zt = (s - s.mean()) / max(s.std(), 1e-6)
    zl = np.full(len(s), -np.inf)
    for t in idx:
        nb = np.r_[s[max(0, t - WIN):t], s[t + 1:t + 1 + WIN]]
        zl[t] = (s[t] - nb.mean()) / max(nb.std(), FLOOR) if len(nb) >= MINNB else zt[t]
    m = np.minimum(zl, zt); j = int(idx[np.argmax(m[idx])])
    sp = span(it, text); inside = [t for t in idx if sp and W[t][0] < sp[1] and W[t][1] > sp[0]]
    return dict(MAIN=float(m[idx].max()), ZLOC=float(zl[idx].max()), ZTASK=float(zt[idx].max()), RAW=float(s[idx].max()),
                SPAN=float(max(s[t] for t in inside)) if inside else np.nan, LEN=float(len(s)),
                arg=W[j][:2], arg_in_span=j in inside, word=text[W[j][0]:W[j][1]])


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    d = pos[:, None] - neg[None, :]
    return float(((d > 0) + 0.5 * (d == 0)).mean())


def tpl(it):
    """template for the within-template AUC (review B1); hindgen traps: key Y = good bet lost, key N = bad bet won"""
    return f"key{it['key']}" if it["family"] == "hindgen" else it["cluster"]


def strat_auc(P, N, key):
    """pairs (habit error, fast right) from the same template only, pooled; returns auc, n pairs, n templates"""
    num = den = 0.0; nt = 0
    for t in {tpl(r["it"]) for r in P} & {tpl(r["it"]) for r in N}:
        a = np.array([score_of(r, key) for r in P if tpl(r["it"]) == t]); b = np.array([score_of(r, key) for r in N if tpl(r["it"]) == t])
        ok_a, ok_b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(ok_a) and len(ok_b):
            d = ok_a[:, None] - ok_b[None, :]; num += ((d > 0) + 0.5 * (d == 0)).sum(); den += d.size; nt += 1
    return (num / den if den else np.nan), int(den), nt


def score_of(r, k):
    return r["CONF"] if k == "CONF" else r["m"][k]


def load(tag):
    items = {x["id"]: x for x in json.load(open(os.path.join(CL, "sets_cl.json")))["items"]}
    data = {}
    for mk in MODELS:
        fn = os.path.join(HERE, f"sp_{tag}_{mk}.jsonl")
        if not os.path.exists(fn):
            continue
        fast = {}
        for l in open(os.path.join(CL, f"out_main_{mk}.jsonl")):
            x = json.loads(l)
            if "res" in x:
                lp = np.array([sum(o) for o in x["res"]["direct"]]); p = np.exp(lp - lp.max()); p /= p.sum()
                th = int(np.argmax([sum(o) for o in x["res"]["think"]]))
                fast[x["id"]] = (int(lp.argmax()), float(p.max()), th)
        out = []
        for l in open(fn):
            x = json.loads(l)
            if "checks" in x:
                print(mk, "checks:", {k: v for k, v in x["checks"].items() if k != "template"}); continue
            if not x.get("tok"):
                print(mk, "item", x["id"], "has no tokens:", x.get("err")); continue
            it = items[x["id"]]; m = measures(it, x["tok"])
            if m is None:
                print(mk, "item", x["id"], "too short"); continue
            a, pc, th = fast[x["id"]]
            out.append(dict(it=it, m=m, ans=a, CONF=1 - pc, think_right=th == it["key"],
                            right=a == it["key"], habit_err=it["habit"] is not None and a == it["habit"]))
        data[mk] = out
    return data


def boot_cell(c, keys, rng):
    """two-level resample: clusters within the cell, then items within clusters (review S2)"""
    cl = {}
    for r in c["P"] + c["N"]:
        cl.setdefault(r["it"]["cluster"], []).append(r)
    ks = list(cl); P, N = [], []
    for k in rng.choice(len(ks), len(ks)):
        g = cl[ks[k]]
        for i in rng.integers(0, len(g), len(g)):
            (P if g[i]["habit_err"] else N).append(g[i])
    return {k: strat_auc(P, N, k)[0] for k in keys}


def main(tag):
    data = load(tag)
    print("models:", list(data), "items per model:", {k: len(v) for k, v in data.items()})
    cells = []
    for mk, rs in data.items():
        for f in FAMS:
            t = [r for r in rs if r["it"]["family"] == f and r["it"]["pop"] == "trap"]
            P = [r for r in t if r["habit_err"]]; N = [r for r in t if r["right"]]
            a, npair, nt = strat_auc(P, N, "MAIN")
            cells.append(dict(mk=mk, f=f, P=P, N=N, other=len(t) - len(P) - len(N), pairs=npair, ntpl=nt,
                              ok=npair >= MIN_PAIRS and nt >= MIN_TPL))
    print("\n== Q1 (primary): traps only, fast habit error (+) vs fast right (-), WITHIN-TEMPLATE AUC; > 0.5 = more -> habit error")
    for c in cells:
        a = {k: strat_auc(c["P"], c["N"], k)[0] for k in SCORES + ["CONF"]}
        print(f"  {c['mk']:7} {c['f']:8} habit {len(c['P']):3} right {len(c['N']):3} other {c['other']:2} pairs {c['pairs']:5} tpl {c['ntpl']:2} "
              f"{'ELIGIBLE' if c['ok'] else 'skip    '} " + " ".join(f"{k} {v:.2f}" for k, v in a.items()))
    el = [c for c in cells if c["ok"]]
    if el:
        q1(el)
    else:
        print("VERDICT Q1: not testable (no eligible cell)")
    rest(data, el)


def q1(el):
    rng = np.random.default_rng(0)
    pt = [{k: strat_auc(c["P"], c["N"], k)[0] for k in ("MAIN", "CONF", "LEN")} for c in el]
    sg = [{k: np.sign(p[k] - 0.5) or 1.0 for k in p} for p in pt]   # each cell's direction, fixed from the full sample
    strength = lambda k, P=pt: float(np.mean([abs(p[k] - 0.5) for p in P]))
    sM, sC, sL = strength("MAIN"), strength("CONF"), strength("LEN")
    bc, bl = [], []
    for _ in range(NB):
        b = [boot_cell(c, ("MAIN", "CONF", "LEN"), rng) for c in el]
        dm = [sg[i]["MAIN"] * (b[i]["MAIN"] - 0.5) for i in range(len(el))]
        bc.append(np.nanmean([dm[i] - sg[i]["CONF"] * (b[i]["CONF"] - 0.5) for i in range(len(el))]))
        bl.append(np.nanmean([dm[i] - sg[i]["LEN"] * (b[i]["LEN"] - 0.5) for i in range(len(el))]))
    loC, loL = np.percentile(bc, 2.5), np.percentile(bl, 2.5)
    signs = [s["MAIN"] for s in sg]; major = 1.0 if sum(x > 0 for x in signs) >= sum(x < 0 for x in signs) else -1.0
    cons = sum(x == major for x in signs)
    per = {}
    for c, p in zip(el, pt):
        per.setdefault(c["mk"], []).append(p["MAIN"] - 0.5)
    models_ok = sum(1 for mk, v in per.items() if (len(v) >= 2 or mk != "olmo13") and np.sign(np.mean(v)) == major)
    print(f"\n  eligible cells {len(el)}; mean |AUC-0.5|: MAIN {sM:.3f}  CONF {sC:.3f}  LEN {sL:.3f}")
    print(f"  MAIN - CONF {sM - sC:+.3f} (bootstrap low {loC:+.3f});  MAIN - LEN {sM - sL:+.3f} (low {loL:+.3f})")
    print(f"  MAIN direction: majority {'more surprise -> habit' if major > 0 else 'more surprise -> right'}, in {cons} of {len(el)} cells; "
          f"models agreeing (OLMo counts only with >= 2 cells): {models_ok}")
    crit = {"|AUC-0.5| >= 0.12": sM >= 0.12, "beats CONF by >= 0.05, low > 0": sM - sC >= 0.05 and loC > 0,
            "beats LEN by >= 0.05, low > 0": sM - sL >= 0.05 and loL > 0, "same direction in >= 2/3 of cells": cons >= 2 * len(el) / 3,
            "direction in >= 2 models": models_ok >= 2}
    print(f"VERDICT Q1: {'PROMISING (worth a frozen test on fresh items)' if all(crit.values()) else 'no lamp here'}  {crit}")
    print("  report-only, mean |AUC-0.5| over eligible cells:", {k: round(float(np.nanmean([abs(strat_auc(c['P'], c['N'], k)[0] - 0.5) for c in el])), 3) for k in SCORES + ["CONF"]})


def rest(data, el):
    print("\n== Q2 (report-only, pipeline check): traps (+) vs controls (-), plain AUC; meaningful only for D and hindgen (review B3)")
    for mk, rs in data.items():
        line = []
        for f in ("D", "hindgen", "parcels", "bonus"):
            t = [r for r in rs if r["it"]["family"] == f and r["it"]["pop"] == "trap"]
            c = [r for r in rs if r["it"]["family"] == f and r["it"]["pop"] == "control"]
            line.append(f"{f}: MAIN {auc([r['m']['MAIN'] for r in t], [r['m']['MAIN'] for r in c]):.2f} SPAN {auc([r['m']['SPAN'] for r in t], [r['m']['SPAN'] for r in c]):.2f}")
        print(f"  {mk:7} " + " | ".join(line))

    print("\n== where the most surprising word (MAIN) lands: share inside the critical span; most common words")
    for mk, rs in data.items():
        for f in FAMS:
            for pop in ("trap", "control"):
                g = [r for r in rs if r["it"]["family"] == f and r["it"]["pop"] == pop]
                if not g:
                    continue
                from collections import Counter
                kind = Counter("span" if r["m"]["arg_in_span"] else "number" if any(ch.isdigit() for ch in r["m"]["word"])
                               else "Name" if r["m"]["word"][:1].isupper() else "other" for r in g)
                top = Counter(r["m"]["word"] for r in g).most_common(4)
                print(f"  {mk:7} {f:8} {pop:8} {dict(kind)}  top: {top}")

    print("\n== gate (report-only, review S6): within family on traps+controls, flag 30% by score; recall of fast errors;"
          " accuracy if reasoning only where flagged (thought answer = forced choice after reasoning)")
    for mk, rs in data.items():
        for f in ("D", "hindgen"):
            g = [r for r in rs if r["it"]["family"] == f]
            k = round(0.3 * len(g)); E = [i for i, r in enumerate(g) if not r["right"]]
            line = f"  {mk:7} {f:8} errors {len(E):3}/{len(g)}  never {np.mean([r['right'] for r in g]):.2f} always {np.mean([r['think_right'] for r in g]):.2f} |"
            for key in ("MAIN", "SPAN", "CONF", "LEN"):
                for sgn in (1, -1):
                    v = np.array([sgn * np.nan_to_num(score_of(r, key), nan=-1e9) for r in g])
                    top = set(np.argsort(-v, kind="stable")[:k])
                    rec = np.mean([i in top for i in E]) if E else np.nan
                    acc = np.mean([g[i]["think_right"] if i in top else g[i]["right"] for i in range(len(g))])
                    line += f" {key}{'+' if sgn > 0 else '-'} rec {rec:.2f} acc {acc:.2f};"
            print(line)

    print("\n== Spearman of MAIN with CONF within traps (same weights; is it one quantity?)")
    for mk, rs in data.items():
        out = []
        for f in FAMS:
            t = [r for r in rs if r["it"]["family"] == f and r["it"]["pop"] == "trap"]
            a = np.argsort(np.argsort([r["m"]["MAIN"] for r in t])); b = np.argsort(np.argsort([r["CONF"] for r in t]))
            out.append(f"{f} {np.corrcoef(a, b)[0, 1]:+.2f}")
        print(f"  {mk:7} " + "  ".join(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
