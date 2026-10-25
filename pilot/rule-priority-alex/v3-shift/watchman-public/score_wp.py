"""Frozen scoring for the watchman on public habit tests (watchman-public/PLAN.md); copy of conftrap-lamp/score_cl.py, changes marked WP.
python3 -I score_wp.py TAG [--smoke]
Reads out_<TAG>_<model>.jsonl. DIRECT = forced choice (summed log-prob). THINK = option parsed from the final "Answer:" line,
else the forced choice after "Answer:". --smoke prints pipeline checks only (no accuracy)."""
import json, os, re, sys
import numpy as np
from sklearn.metrics import roc_auc_score  # CL: imported first so a missing package fails before any output

HERE = os.path.dirname(os.path.abspath(__file__))
B = 2000
R4 = lambda v: round(float(v), 4)
MODELS = ["qwen14", "phi4", "olmo13"]
PRIM = ["cpkc", "modar", "base9"]   # WP: verdict families
ALLT = PRIM
import unicodedata
fold = lambda s: unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()   # WP review S5: accents folded
norm = lambda s: re.findall(r"[a-z0-9]+", fold(s).lower().replace("−", "-"))
num = lambda s: re.findall(r"(?<![\w.])-?\d+", re.sub(r"(?<=\d),(?=\d{3}\b)", "", s.replace("−", "-")))   # WP review B1: signs, 1,381
WORDS = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                                         "sixteen seventeen eighteen nineteen twenty".split())}
CLF = ("modar", "base9")  # WP: numeric families (off-list answers possible)
TAG = sys.argv[1]; SMOKE = "--smoke" in sys.argv
DROPPED = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--dropped=")]  # only for a load / chat-template failure (PLAN)
DROPPED = DROPPED[0].split(",") if DROPPED else []


def parse_line(task, cands, line, sample=False):
    """option named on an answer line, else None. sample=True: a short sampled fast answer (first word for memo, lenient lead-in)"""
    line = line.replace("<answer>", "").strip()
    if task in ("modar", "base9"):  # WP (review B1): signed numbers, thousands separators removed; the one option named
        hits = [cands.index(t) for t in num(line) if t in cands]   # (sampled answer: the first option named)
        if sample:
            return hits[0] if hits else None
        return hits[0] if hits and len(set(hits)) == 1 else None
    if sample:  # drop lead-in punctuation, quotes and words like "Answer", "option"
        line = re.sub(r"^(?:[\s:\"'*`(-]|answer\s*(?:is)?\s*:?|option\s+)+", "", line, flags=re.I)
    if task == "cpkc":  # WP: the one option whose full name appears on the line, as whole words (sampled answer: the first one to appear)
        l = " " + " ".join(norm(line)) + " "; pos = {i: l.find(" " + " ".join(norm(c)) + " ") for i, c in enumerate(cands)}
        pos = {i: p for i, p in pos.items() if p >= 0 and norm(cands[i])}
        if sample:
            return min(pos, key=pos.get) if pos else None
        return next(iter(pos)) if len(pos) == 1 else None
    if task in ("modar", "base9"):  # CL: the one option named on the line (sampled answer: the first option named)
        toks = [WORDS.get(t, t) for t in norm(line)]; cn = [" ".join(norm(c)) for c in cands]   # CL: number words -> digits (review S3)
        hits = [cn.index(t) for t in toks if t in cn]
        if sample:
            return hits[0] if hits else None
        return hits[0] if hits and len(set(hits)) == 1 else None
    if task == "memo":  # final word of the line only (sampled answer: its first word)
        toks = norm(line)
        cn = [norm(c)[-1] if norm(c) else c.lower() for c in cands]
        w = (toks[0] if sample else toks[-1]) if toks else None
        return cn.index(w) if w in cn and cn.count(w) == 1 else None
    if all(re.fullmatch(r"[A-M]", c) for c in cands):
        if re.match(r"^\(?[A-M]\)?\s*(?:,|and)\s*\(?[A-M]\b", line):  # several letters named -> no single answer
            return None
        # letter at the start ("B", "B.", "(B)", "B) ..."), else a letter followed by "." or ")" anywhere, else None
        m = re.match(r"^\**\(?([A-M])(?:[\).:\"',]|\s*$|\s*\*|\s+-)", line) or re.search(r"(?<![A-Za-z])\(?([A-M])[\.\)](?!\w)", line)
        return cands.index(m.group(1)) if m and m.group(1) in cands else None
    if cands == ["entailment", "non-entailment"]:
        l = line.lower()
        if re.search(r"non[\s-]?entail|\b(?:not|no|cannot|can't|never|neither)\b[\w\s-]{0,25}entail|n't[\w\s-]{0,25}entail", l):
            return 1
        return 0 if re.search(r"\bentail", l) else None
    yn = {"y": "y", "yes": "y", "n": "n", "no": "n"}  # Y/N
    cy = [yn.get(c.lower()) for c in cands]
    for t in norm(line):
        if t in yn and yn[t] in cy:
            return cy.index(yn[t])
    return None


def parse(r, g):
    if "Answer:" not in g:
        return None
    line = g[g.rindex("Answer:") + 7:].strip().split("\n")[0]
    p = parse_line(r["task"], r["cands"], line)
    if p is None and r["task"] in CLF:  # a final answer naming a number but no option = off-list (-1): differs (FLAG) and wrong
        ns = num(line)
        if ns and not set(ns) & set(r["cands"]):
            return -1
    return p


rows, checks = [], {}
SETS = json.load(open(os.path.join(HERE, "sets_smoke.json" if SMOKE else "sets_wp.json")))["items"]
for mk in MODELS:
    fn = os.path.join(HERE, f"out_{TAG}_{mk}.jsonl")
    if mk in DROPPED:
        print("DROPPED (declared):", mk); continue
    if not os.path.exists(fn):
        print("MISSING", fn, "-> ABORT: no verdict (a model may only be dropped with --dropped= for a load/template failure)"); sys.exit(1)
    for l in open(fn):
        x = json.loads(l)
        if "checks" in x:
            checks[mk] = x["checks"]
        else:
            x["task"] = x["family"]  # CL
            rows.append(x)
am = lambda sc, mean=False: int(np.argmax([sum(s) / len(s) if mean else sum(s) for s in sc]))
for r in rows:
    d = r["res"]["direct"]; p = parse(r, r["gen_think"])
    a = {"direct": am(d), "direct_mean": am(d, True), "think_forced": am(r["res"]["think"])}
    a["think"] = a["think_forced"] if p is None else p; r["parsed"] = p is not None and p >= 0; r["offlist"] = p == -1; r["a"] = a
    v = sorted((sum(s) for s in d), reverse=True); r["margin"] = v[0] - v[1]
    sp = [parse_line(r["task"], r["cands"], s.lstrip(" :\n").split("\n")[0], sample=True) for s in r["samples"]]
    r["samp_parsed"] = np.mean([q is not None for q in sp])
    r["samp_dis"] = {k: sum(q != a[k] for q in sp) for k in ("direct", "direct_mean")}  # unparsed sample counts as disagreeing
    r["samp_dis2"] = sum(q is not None and q != a["direct"] for q in sp) + 0.1 * sum(q is None for q in sp)  # CL report-only SAMPLE' (S3)
models = [m for m in MODELS if any(r["model"] == m for r in rows)]
print("models with output:", models)
want = {x["id"]: (x["family"], x["key"], x["cands"]) for x in SETS}  # CL
for mk in models:  # every model complete and matching the frozen item file (code review C4)
    got = {r["id"]: (r["task"], r["key"], r["cands"]) for r in rows if r["model"] == mk}
    assert mk in checks, ("no final checks line (run not finished)", mk)
    assert got == want, ("items differ from the frozen set", mk, len(got), len(want))
for mk in models:
    print(mk, checks.get(mk))
print("\n## Pipeline: items, parse rate of final Answer line, truncated generations, parse rate of sampled short answers")
for mk in models:
    for t in ALLT:
        rs = [r for r in rows if r["model"] == mk and r["task"] == t]
        if rs:
            print(f"  {mk:7} {t:8} n {len(rs):4}  parsed {np.mean([r['parsed'] for r in rs]):.0%}  off-list {np.mean([r['offlist'] for r in rs]):.0%}  truncated "
                  f"{np.mean([r['trunc_think'] for r in rs]):.0%}  samples parsed {np.mean([r['samp_parsed'] for r in rs]):.0%}")
if SMOKE:
    for mk in models:
        for t in ALLT:
            for r in [r for r in rows if r["model"] == mk and r["task"] == t][:2]:
                g = r["gen_think"]; line = g[g.rindex("Answer:"):].split("\n")[0] if "Answer:" in g else "(no Answer line)"
                print(f"  {mk} {t} answer line: {line[:100]!r}  sample: {r['samples'][0][:60]!r}")
    sys.exit(0)

if len(models) < 2:
    print("fewer than 2 models -> no verdict"); sys.exit(0)
need = max(2, len(models) - 1)  # per-model condition: > 0 in at least this many models
rng = np.random.default_rng(0)
jit = {}
for r in rows:
    jit[(r["model"], r["id"])] = rng.uniform(0, 1e-6)
cells = {}
for r in rows:
    cells.setdefault((r["model"], r["task"]), []).append(r)
for k in cells:
    cells[k].sort(key=lambda r: r["id"])
ids = {t: [r["id"] for r in cells[(models[0], t)]] for t in ALLT}
for mk in models:
    for t in ALLT:
        assert [r["id"] for r in cells[(mk, t)]] == ids[t], ("models must share items", mk, t)
clus = {t: [r["cluster"] for r in cells[(models[0], t)]] for t in ALLT}


def draws(t, scheme, g):
    """one resample of item positions for test t, shared by all models (PLAN v2, B4)"""
    n = len(ids[t])
    if scheme == "item":
        return g.integers(0, n, n)
    cl = {}
    for i, c in enumerate(clus[t]):
        cl.setdefault(c, []).append(i)
    ks = list(cl); return np.array([i for j in g.integers(0, len(ks), len(ks)) for i in cl[ks[j]]])


def boot(stat, keys):
    """stat(dict cell->rows) -> float; item and cluster schemes, same draw for every model, wider 95% interval"""
    pt = stat({k: cells[k] for k in keys}); lo, hi = [], []
    for scheme in ("item", "cluster"):
        g = np.random.default_rng(1); v = []
        for _ in range(B):
            dr = {t: draws(t, scheme, g) for t in sorted({k[1] for k in keys})}
            v.append(stat({k: [cells[k][i] for i in dr[k[1]]] for k in keys}))
        lo.append(np.nanpercentile(v, 2.5)); hi.append(np.nanpercentile(v, 97.5))
    return pt, min(lo), max(hi)


def pick_k(rs, key, k):
    """ids of the k items ranked highest by key (ties broken by fixed jitter)"""
    return set(r["id"] for r in sorted(rs, key=lambda r: -(key(r) + jit[(r["model"], r["id"])]))[:k])


def recalls(rs, D="direct", T="think"):
    """recall of fast errors for FLAG and the rivals at FLAG's rate"""
    flag = [r["a"][T] != r["a"][D] for r in rs]; E = [i for i, r in enumerate(rs) if r["a"][D] != r["key"]]
    if not E:
        return dict(FLAG=np.nan, RANDOM=np.nan, CONF=np.nan, SAMPLE=np.nan, SAMPLE2=np.nan, rate=np.nan)
    k = int(round(np.mean(flag) * len(rs)))
    mg = (lambda r: -r["margin"]) if D == "direct" else (lambda r: -sorted([sum(s) / len(s) for s in r["res"]["direct"]])[-1]
                                                          + sorted([sum(s) / len(s) for s in r["res"]["direct"]])[-2])
    conf = pick_k(rs, mg, k); samp = pick_k(rs, lambda r: r["samp_dis"][D], k); samp2 = pick_k(rs, lambda r: r["samp_dis2"], k)
    return dict(FLAG=np.mean([flag[i] for i in E]), RANDOM=k / len(rs), CONF=np.mean([rs[i]["id"] in conf for i in E]),
                SAMPLE=np.mean([rs[i]["id"] in samp for i in E]), SAMPLE2=np.mean([rs[i]["id"] in samp2 for i in E]), rate=k / len(rs))


acc = lambda rs, a: float(np.mean([r["a"][a] == r["key"] for r in rs]))
traps = lambda rs: [r for r in rs if r["pop"] == "trap"]
gain = lambda rs, D="direct", T="think": acc(traps(rs), T) - acc(traps(rs), D)
mean_cells = lambda f: (lambda d: float(np.nanmean([f(v) for v in d.values()])))

print("\n## Per cell (traps and controls together unless marked): fast right, thought right, flag rate, fast errors E, recall at flag rate")
for mk in models:
    for t in ALLT:
        rs = cells[(mk, t)]; R = recalls(rs); nE = sum(r["a"]["direct"] != r["key"] for r in rs)
        ap = np.mean([r["a"]["direct"] != r["key"] and r["a"]["think"] == r["key"] for r in rs])
        print(f"  {mk:7} {t:8} n {len(rs):4}  fast {100 * acc(rs, 'direct'):5.1f}  thought {100 * acc(rs, 'think'):5.1f}  flag {100 * R['rate']:5.1f}"
              f"  E {nE:4}  rec FLAG {100 * R['FLAG']:5.1f} RANDOM {100 * R['RANDOM']:5.1f} CONF {100 * R['CONF']:5.1f} SAMPLE {100 * R['SAMPLE']:5.1f}"
              f"  autopilot {100 * ap:5.1f}")
        for pop in ("trap", "control"):  # CL: every family
            q = [r for r in rs if r["pop"] == pop]
            if q and t != "mmlu":
                print(f"  {'':7} {t}/{pop:7} n {len(q):4} fast {100 * acc(q, 'direct'):5.1f}  thought {100 * acc(q, 'think'):5.1f}")


def eligible(tests, D="direct"):
    return [(mk, t) for mk in models for t in tests
            if 15 <= sum(r["a"][D] != r["key"] for r in cells[(mk, t)]) <= len(cells[(mk, t)]) - 15]


def h1(tests, D="direct", T="think", label="", verbose=True):
    el = eligible(tests, D)
    if len(el) < 4:
        print(f"H1{label}: only {len(el)} eligible cells (< 4) -> not testable"); return None, {}
    res, signs = True, {}
    for rv in ("RANDOM", "CONF", "SAMPLE"):
        f = mean_cells(lambda rs: (lambda R: R["FLAG"] - R[rv])(recalls(rs, D, T)))
        pt, lo, hi = boot(f, el); signs[rv] = np.sign(R4(pt))
        per = {mk: float(np.nanmean([(lambda R: R["FLAG"] - R[rv])(recalls(cells[c], D, T)) for c in el if c[0] == mk]))
               for mk in models if any(c[0] == mk for c in el)}
        npos = sum(v > 0 for v in per.values()); ok = R4(pt) >= 0.10 and R4(lo) > 0 and npos >= min(need, len(per)) and len(per) >= 2
        res &= ok
        if verbose:
            print(f"H1{label} FLAG - {rv:6} {100 * pt:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] need >= +10, low > 0; per model "
                  + ", ".join(f"{m} {100 * v:+.1f}" for m, v in per.items()) + f" -> {'pass' if ok else 'FAIL'}")
    return res, signs


def h2(tests, D="direct", T="think", label="", verbose=True):
    ks = [(mk, t) for mk in models for t in tests]
    f = mean_cells(lambda rs: gain(rs, D, T)); pt, lo, hi = boot(f, ks)
    per = {mk: float(np.mean([gain(cells[(mk, t)], D, T) for t in tests])) for mk in models}
    npos = sum(v > 0 for v in per.values()); ok = R4(pt) >= 0.05 and R4(lo) > 0 and npos >= need
    if verbose:
        print(f"H2{label} THINK - DIRECT (traps) {100 * pt:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] need >= +5, low > 0; per model "
              + ", ".join(f"{m} {100 * v:+.1f}" for m, v in per.items()) + f" -> {'PASS' if ok else 'FAIL'}")
    return ok, np.sign(R4(pt))


print(f"\n## H1 lamp, verdict families {PRIM} (need > 0 per model in >= {need} models)")
H1, s1 = h1(PRIM)
print(f"H1 -> {'PASS' if H1 else 'FAIL' if H1 is not None else 'not testable'}")
print("\n## Readout sensitivity: same sign required for every rival")
sens1 = True
for D, T in (("direct_mean", "think"), ("direct", "think_forced")):
    r1, t1 = h1(PRIM, D, T, f" [{D}/{T}]")
    same1 = H1 is None or (r1 is not None and all(t1.get(k) == s1.get(k) for k in s1))  # untestable sensitivity = not same
    sens1 &= same1
    print(f"  sign same as main: {same1}")
print(f"SENSITIVITY -> {'same sign' if sens1 else 'SIGN DIFFERS'}")
print("\n## Leave-one-family-out (review B1, required): every rival's point difference must stay > 0 with each family dropped")
lofo = H1 is not None
if H1 is not None:
    el = eligible(PRIM)
    for f in PRIM:
        sub = [c for c in el if c[1] != f]
        for rv in ("RANDOM", "CONF", "SAMPLE"):
            v = float(np.nanmean([(lambda R: R["FLAG"] - R[rv])(recalls(cells[c])) for c in sub])) if sub else np.nan
            ok = R4(v) > 0; lofo &= ok
            print(f"  without {f:8} FLAG - {rv:6} {100 * v:+.1f} ({len(sub)} cells) -> {'ok' if ok else 'SIGN LOST'}")
print(f"LOFO -> {'holds' if lofo else 'FAILS'}")
print(f"\nVERDICT V2 lamp vs fast-only rivals (expected by construction if reasoning fixes errors, see PLAN v2): "
      f"{'not testable' if H1 is None else 'PASS' if H1 and sens1 and lofo else 'NOT confirmed'}")

# WP (PLAN v2, after the conftrap-lamp audit): V1 = does reasoning fix fast errors on these public habit tests?
print(f"\n## V1 (primary) reasoning fixes habit traps: THINK - DIRECT on traps, need >= +5, low > 0, > 0 in >= {need} models")
V1, v1s = h2(PRIM, label=" V1")
V1r, v1rs = h2(PRIM, "direct", "think_forced", label=" V1 [think_forced]")
harm1 = True
for mk in models:
    q = [r for t in PRIM for r in cells[(mk, t)] if r["pop"] == "control"]; hc = acc(q, "think") - acc(q, "direct")
    ok = R4(hc) >= -0.05; harm1 &= ok
    print(f"  controls {mk:7} THINK - DIRECT {100 * hc:+.1f} need >= -5 -> {'ok' if ok else 'HARM'}")
lofo1 = True
for f in PRIM:
    sub = [t for t in PRIM if t != f]
    v = float(np.mean([gain(cells[(mk, t)]) for mk in models for t in sub])); ok = R4(v) > 0; lofo1 &= ok
    print(f"  without {f:8} THINK - DIRECT {100 * v:+.1f} -> {'ok' if ok else 'SIGN LOST'}")
print(f"VERDICT V1 reasoning fixes habit errors on public tests: {'PASS' if V1 and v1rs == v1s and harm1 and lofo1 else 'NOT shown'}")

print("\n## Traps only (report-only, review S3): FLAG minus rival, mean over cells eligible within traps")
tcells = {k: traps(v) for k, v in cells.items()}
for rv in ("RANDOM", "CONF", "SAMPLE"):
    el = [(mk, t) for mk in models for t in PRIM if 15 <= sum(r["a"]["direct"] != r["key"] for r in tcells[(mk, t)]) <= len(tcells[(mk, t)]) - 15]
    v = [(lambda R: R["FLAG"] - R[rv])(recalls(tcells[c])) for c in el]
    print(f"  traps-only FLAG - {rv:6} {100 * np.nanmean(v) if v else float('nan'):+.1f} ({len(el)} cells)")
for mk in models:  # modar by subtask
    for op, sym in (("add", " + "), ("sub", " - "), ("mul", " * ")):
        q = [r for r in tcells[(mk, "modar")] if sym in r["cluster"].strip().split("\n")[-1]]
        if q:
            print(f"  {mk:7} modar {op} traps n {len(q):3} fast {100 * acc(q, 'direct'):5.1f} thought {100 * acc(q, 'think'):5.1f}")
cp = {mk: np.mean([r["parsed"] is False for t in ["cpkc"] for r in cells[(mk, t)]]) for mk in models}
print("  cpkc final answers naming no single option (forced choice used): " + ", ".join(f"{m} {100 * v:.0f}%" for m, v in cp.items()))

print("\n## Same-information rival and unfixed errors (report-only, PLAN v2)")
def thinkprob(rs):
    """flag the k items where the forced-choice probability of the FAST answer, read after the reasoning, is lowest"""
    flag = [r["a"]["think"] != r["a"]["direct"] for r in rs]; k = int(round(np.mean(flag) * len(rs)))
    def pf(r):
        v = np.array([sum(x) for x in r["res"]["think"]]); p = np.exp(v - v.max()); p /= p.sum(); return p[r["a"]["direct"]]
    tp = pick_k(rs, lambda r: -pf(r), k); E = [r for r in rs if r["a"]["direct"] != r["key"]]
    return np.mean([r["id"] in tp for r in E]) if E else np.nan
for mk in models:
    el = [c for c in eligible(PRIM) if c[0] == mk]
    if el:
        print(f"  {mk:7} FLAG - THINKPROB {100 * np.nanmean([recalls(cells[c])['FLAG'] - thinkprob(cells[c]) for c in el]):+.1f}  (predicted ~0)")
for mk in models:
    for split, cond in (("fixed by reasoning", lambda r: r["a"]["think"] == r["key"]), ("NOT fixed", lambda r: r["a"]["think"] != r["key"])):
        tot = {k: [] for k in ("FLAG", "CONF", "SAMPLE", "RANDOM")}
        for t in PRIM:
            rs = cells[(mk, t)]; flag = {r["id"] for r in rs if r["a"]["think"] != r["a"]["direct"]}; k = len(flag)
            conf = pick_k(rs, lambda r: -r["margin"], k); samp = pick_k(rs, lambda r: r["samp_dis"]["direct"], k)
            for r in rs:
                if r["a"]["direct"] != r["key"] and cond(r):
                    tot["FLAG"].append(r["id"] in flag); tot["CONF"].append(r["id"] in conf); tot["SAMPLE"].append(r["id"] in samp)
                    tot["RANDOM"].append(k / len(rs))
        n = len(tot["FLAG"])
        print(f"  {mk:7} fast errors {split:18} n {n:4}: " + "  ".join(f"{k} {100 * np.mean(v):5.1f}" for k, v in tot.items() if n))

print("\n## Report-only")
for mk in models:  # SAMPLE' (parsed disagreements first)
    el = [c for c in eligible(PRIM) if c[0] == mk]
    if el:
        print(f"  {mk:7} FLAG - SAMPLE' (parsed disagreements first) {100 * np.nanmean([(lambda R: R['FLAG'] - R['SAMPLE2'])(recalls(cells[c])) for c in el]):+.1f}")
print("  thinking on traps (THINK - DIRECT):"); h2(PRIM, label=" traps")
for mk in models:  # harm where the habit is right
    q = [r for t in PRIM for r in cells[(mk, t)] if r["pop"] == "control"]
    print(f"  controls {mk:7} fast {100 * acc(q, 'direct'):.1f} thought {100 * acc(q, 'think'):.1f} (n {len(q)})")
for mk in models:  # R2 predicted-class prior rival, CONF AUC, precision of FLAG and CONF at equal rate
    for t in PRIM:
        rs = cells[(mk, t)]; R = recalls(rs); k = int(round(R["rate"] * len(rs)))
        cnt = np.bincount([r["a"]["direct"] for r in rs], minlength=len(rs[0]["cands"])); top = int(np.argmax(cnt))
        pr = pick_k(rs, lambda r: (r["a"]["direct"] == top) * 1e3 - r["margin"], k)
        E = [r for r in rs if r["a"]["direct"] != r["key"]]
        y = [r["a"]["direct"] != r["key"] for r in rs]
        auc = roc_auc_score(y, [-r["margin"] for r in rs]) if 0 < sum(y) < len(y) else np.nan
        print(f"  R2 {mk:7} {t:8} recall PRIOR-CLASS {100 * np.mean([r['id'] in pr for r in E]) if E else float('nan'):5.1f}  CONF AUC {auc:.2f}")
for mk in models:  # R3 think only on the 30% least confident
    for t in PRIM:
        rs = traps(cells[(mk, t)]); k = int(round(0.3 * len(rs))); low = pick_k(rs, lambda r: -r["margin"], k)
        a30 = np.mean([(r["a"]["think"] if r["id"] in low else r["a"]["direct"]) == r["key"] for r in rs])
        a0 = acc(rs, "direct"); a1 = acc(rs, "think"); sh = (a30 - a0) / (a1 - a0) if abs(a1 - a0) > 1e-9 else np.nan
        print(f"  R3 {mk:7} {t:8} never {100 * a0:5.1f}  think-on-30%-least-confident {100 * a30:5.1f}  always {100 * a1:5.1f}  share {sh:.0%}")
for mk in models:  # R6 disagreements
    rs = [r for t in PRIM for r in cells[(mk, t)] if r["a"]["think"] != r["a"]["direct"]]
    print(f"  R6 {mk:7} disagreements {len(rs)}: thought right {100 * np.mean([r['a']['think'] == r['key'] for r in rs]):.0f}%,"
          f" fast right {100 * np.mean([r['a']['direct'] == r['key'] for r in rs]):.0f}%")
