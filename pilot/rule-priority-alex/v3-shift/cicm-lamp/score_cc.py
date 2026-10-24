"""cicm-lamp scoring (frozen before the main run).  python3 score_cc.py main
Reads out_<tag>.json (modal_cc.py) + the CICM jsonl; applies PREREG.md criteria on confirmation rows."""
import json, os, sys, random
from collections import defaultdict
import numpy as np
import split_cc

HERE = os.path.dirname(os.path.abspath(__file__)); tag = sys.argv[1] if len(sys.argv) > 1 else "main"
NB, SEED = 2000, 20261024
rows = {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(HERE, "data", "cicm_natural_factorial_otherdist_l0.jsonl")))}
cal_list, conf_list = split_cc.split(list(rows.values())); cal_ids = {r["id"] for r in cal_list}
out = {o["id"]: o for o in json.load(open(os.path.join(HERE, f"out_{tag}.json")))}
if tag == "main":  # no partial-run verdicts
    assert os.path.exists(os.path.join(HERE, f"out_{tag}.DONE")), "run not finished"
    assert len(out) == 1200 and all(r["id"] in out for r in conf_list), "missing rows"
RAND = sorted(k[4:] for k in next(iter(out.values())) if k.startswith("off_rand"))


def lsm(v):
    v = np.asarray(v, float); return v - np.logaddexp.reduce(v)


def marg(v, p):
    v = lsm(v); o = np.delete(v, p); return v[p] - o.max()


def cell(r):
    c = r["factorial_cell"]; return (c["same_slot_stale_distance_bin"], c["recent_other_slot_distance_bin"])


rec = []
for i, o in out.items():
    r = rows[i]; c = o["cands"]; p = int(np.argmax(o["final"])); v = c[p]
    lab = "correct" if v == r["current_value"] else "stale" if v in r["stale_values"] else "other"
    d = dict(id=i, cal=i in cal_ids, cell=cell(r), ck=cell(r) + (r["k_overwrites"],), k=r["k_overwrites"],
             clus=(r["slot"], r["current_value"], r["k_overwrites"]) + cell(r), lab=lab, p=p, one=0,
             conf=-marg(o["final"], p), lamp=-marg(o["off_top20"], p), diff=marg(o["final"], p) - marg(o["off_top20"], p),
             off_pick=int(np.argmax(o["off_top20"])), cur=c.index(r["current_value"]), ntok=(o.get("ntok") or [0] * 7)[p],
             gen=o.get("gen", ""), r=r, o=o)
    for k in RAND:
        d["rand_" + k] = -marg(o["off_" + k], p)
    rec.append(d)
cal = [d for d in rec if d["cal"]]; conf = [d for d in rec if not d["cal"]]
print(f"tag {tag}: rows {len(rec)} (cal {len(cal)}, conf {len(conf)}); random head sets {RAND}")

agree = n = 0
for d in rec:
    hits = [v for v in d["r"]["slot_values"] if v.lower() in d["gen"].lower()]
    if len(hits) == 1:
        n += 1; agree += hits[0] == d["o"]["cands"][d["p"]]
print(f"sanity: forced pick = value in greedy answer on {agree}/{n} ({100 * agree / max(n, 1):.0f}%)")
for name, S in (("cal", cal), ("conf", conf)):
    print(f"labels {name}: " + ", ".join(f"{k} {sum(d['lab'] == k for d in S)}" for k in ("correct", "stale", "other")))
for lab in ("correct", "stale", "other"):
    cnt = defaultdict(int)
    for d in conf:
        if d["lab"] == lab:
            cnt[d["ntok"]] += 1
    print(f"pick token count, {lab} (conf): {dict(sorted(cnt.items()))}")

print("\nperturbation by head set (all rows): flip%  |dmarg|  KL")
for k in ["top20"] + RAND:
    fl = np.mean([int(np.argmax(d["o"]["off_" + k])) != d["p"] for d in rec])
    dm = np.mean([abs(marg(d["o"]["final"], d["p"]) - marg(d["o"]["off_" + k], d["p"])) for d in rec])
    kl = np.mean([float(np.sum(np.exp(lsm(d["o"]["final"])) * (lsm(d["o"]["final"]) - lsm(d["o"]["off_" + k])))) for d in rec])
    print(f"  {k:7s} {100 * fl:5.1f}  {dm:6.3f}  {kl:6.3f}")

cr = defaultdict(list)
for d in cal:
    if d["lab"] != "other":
        cr[d["ck"]].append(d["lab"] == "stale")
base = np.mean([d["lab"] == "stale" for d in cal if d["lab"] != "other"])
for d in rec:
    d["cellb"] = float(np.mean(cr[d["ck"]])) if cr[d["ck"]] else base


def auc(S, key, strat="cell", pos=("stale",), neg=("correct",)):
    g = defaultdict(lambda: ([], []))
    for d in S:
        if d["lab"] in pos:
            g[d[strat]][0].append(d[key])
        elif d["lab"] in neg:
            g[d[strat]][1].append(d[key])
    num = den = 0.0
    for P, N in g.values():
        if P and N:
            P = np.asarray(P); N = np.asarray(N)
            num += float((P[:, None] > N[None, :]).sum() + 0.5 * (P[:, None] == N[None, :]).sum()); den += len(P) * len(N)
    return num / den if den else float("nan")


def boot(S, f):
    rnd = random.Random(SEED); cl = defaultdict(list)
    for d in S:
        cl[d["clus"]].append(d)
    keys = sorted(cl); res = []
    for _ in range(NB):
        T = [d for kk in (rnd.choice(keys) for _ in keys) for d in cl[kk]]; res.append(f(T))
    res = sorted(x for x in res if x == x); return res[int(.025 * len(res))], res[int(.975 * len(res)) - 1]


print("\n== C1: AUC stale vs correct, confirmation (within cell / within cell*k / pooled) ==")
for key in ("lamp", "conf", "diff", "cellb") + tuple("rand_" + k for k in RAND):
    print(f"  {key:10s} {auc(conf, key):.3f}  {auc(conf, key, 'ck'):.3f}  {auc(conf, key, 'one'):.3f}")
a_l, a_c = auc(conf, "lamp"), auc(conf, "conf"); a_lk, a_ck = auc(conf, "lamp", "ck"), auc(conf, "conf", "ck")
best_r = max(RAND, key=lambda k: auc(conf, "rand_" + k))
a_r = auc(conf, "rand_" + best_r)
lo1, hi1 = boot(conf, lambda T: auc(T, "lamp") - auc(T, "conf"))
lo2, hi2 = boot(conf, lambda T: auc(T, "lamp") - auc(T, "rand_" + best_r))
lo3, hi3 = boot(conf, lambda T: auc(T, "lamp", "ck") - auc(T, "conf", "ck"))
print(f"  LAMP-CONF cell {a_l - a_c:+.3f} [{lo1:+.3f},{hi1:+.3f}]   LAMP-CONF cell*k {a_lk - a_ck:+.3f} [{lo3:+.3f},{hi3:+.3f}]")
print(f"  LAMP-best random set ({best_r}) {a_l - a_r:+.3f} [{lo2:+.3f},{hi2:+.3f}]")
C1 = (a_l >= 0.70 and a_l - a_c >= 0.05 and lo1 > 0 and a_l - a_r >= 0.05 and lo2 > 0 and a_lk >= 0.65 and lo3 > 0)
print(f"  C1 {'PASS' if C1 else 'FAIL'}")

print("\nspecificity (report): within-cell AUC, rows with k<6")
k6 = [d for d in conf if d["k"] < 6]
for key in ("lamp", "conf"):
    print(f"  {key:5s} other-vs-correct {auc(k6, key, pos=('other',)):.3f}   wrong-vs-correct {auc(k6, key, pos=('stale', 'other')):.3f}")
print("token count of pick (report): LAMP within-cell AUC single-token picks "
      f"{auc([d for d in conf if d['ntok'] == 1], 'lamp'):.3f}, multi-token {auc([d for d in conf if d['ntok'] > 1], 'lamp'):.3f}")

try:
    tr = [d for d in cal if d["lab"] != "other"]; Xt = np.array([d["o"]["hid"][1] for d in tr], float)
    yt = np.array([d["lab"] == "stale" for d in tr], float)
    mu = {c: Xt[[j for j, d in enumerate(tr) if d["cell"] == c]].mean(0) for c in {d["cell"] for d in tr}}
    cen = lambda S, A: A - np.array([mu[d["cell"]] for d in S])
    Zt = cen(tr, Xt); m, s = Zt.mean(0), Zt.std(0) + 1e-6; Zt = (Zt - m) / s
    w = np.zeros(Zt.shape[1]); b = 0.0
    for _ in range(500):
        pr = 1 / (1 + np.exp(-(Zt @ w + b))); w -= 0.05 * (Zt.T @ (pr - yt) / len(yt) + w / len(yt)); b -= 0.05 * np.mean(pr - yt)
    for d, z in zip(conf, ((cen(conf, np.array([d["o"]["hid"][1] for d in conf], float)) - m) / s) @ w + b):
        d["probe"] = float(z)
    print(f"PROBE (secondary, weak by design) within-cell {auc(conf, 'probe'):.3f}  cell*k {auc(conf, 'probe', 'ck'):.3f}")
except Exception as e:
    print("PROBE skipped:", repr(e)[:120])

print("\n== C2: brake = take heads-off (top20) pick when the selector fires ==")
def run(S, key, thr):
    fire = [d[key] > thr for d in S]
    pick = [d["off_pick"] if f else d["p"] for f, d in zip(fire, S)]
    ok = [pk == d["cur"] for pk, d in zip(pick, S)]; b0 = [d["p"] == d["cur"] for d in S]
    pres = sum(a and b for a, b in zip(ok, b0)) / max(1, sum(b0))
    st = [j for j, d in enumerate(S) if d["lab"] == "stale"]
    return dict(acc=float(np.mean(ok)), base=float(np.mean(b0)), pres=pres, corr=sum(ok[j] for j in st) / max(1, len(st)), fire=float(np.mean(fire)))


def pick_thr(key):
    best = None
    for thr in sorted({d[key] for d in cal}, reverse=True) + [float("-inf")]:  # prefer the highest threshold on ties
        q = run(cal, key, thr)
        if q["pres"] >= 0.95 and (best is None or q["acc"] > best[0] + 1e-12):
            best = (q["acc"], thr)
    return best[1] if best else float("inf")


t_l, t_c = pick_thr("lamp"), pick_thr("conf")
L, Cq, A = run(conf, "lamp", t_l), run(conf, "conf", t_c), run(conf, "lamp", float("-inf"))
rng = np.random.default_rng(SEED); racc = []
for _ in range(200):
    for d in conf:
        d["rf"] = float(rng.random() < L["fire"])
    racc.append(run(conf, "rf", 0.5)["acc"])
lo4, hi4 = boot(conf, lambda T: run(T, "lamp", t_l)["acc"] - run(T, "lamp", t_l)["base"])
lo5, hi5 = boot(conf, lambda T: run(T, "lamp", t_l)["acc"] - run(T, "conf", t_c)["acc"])
fmt = lambda q: f"acc {100 * q['acc']:.1f}  preservation {100 * q['pres']:.1f}  stale corrected {100 * q['corr']:.1f}  fires {100 * q['fire']:.1f}%"
print(f"  normal          acc {100 * L['base']:.1f}")
print(f"  LAMP brake      {fmt(L)}  (thr {t_l:.3f})  gain {100 * (L['acc'] - L['base']):+.1f} [{100 * lo4:+.1f},{100 * hi4:+.1f}]")
print(f"  CONF brake      {fmt(Cq)}  (thr {t_c:.3f})")
print(f"  RANDOM brake    acc {100 * np.mean(racc):.1f} (same fire rate as lamp, 200 draws)")
print(f"  always off      {fmt(A)}")
print(f"  LAMP-CONF brake {100 * (L['acc'] - Cq['acc']):+.1f} [{100 * lo5:+.1f},{100 * hi5:+.1f}]")
always_ok = A["acc"] - A["base"] >= 0.05 and A["pres"] >= 0.95
C2 = (L["acc"] - L["base"] >= 0.05 and lo4 > 0 and L["pres"] >= 0.95 and L["fire"] < 0.70 and lo5 > 0
      and (L["acc"] >= A["acc"] if always_ok else L["acc"] >= A["acc"] - 0.02))
print(f"  C2 {'PASS' if C2 else 'FAIL'}" + ("   (always-off alone meets +5/95%, so the lamp must not lose to it)" if always_ok else ""))
print(f"\nVERDICT: {'PASS' if C1 and C2 else 'FAIL'}  (C1 {C1}, C2 {C2})")
print("context: Berkeley published Qwen2.5-14B rerouting on these confirmation rows, free-form greedy: 40.5 -> 41.7 (+1.15), 13/491 stale corrected")
