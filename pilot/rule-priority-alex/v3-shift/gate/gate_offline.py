"""Offline "lamp -> brake + louder new rule" on saved 14B scores (no new runs).
python3 -I gate_offline.py ../eras/eras_Qwen2.5-14B-Instruct.jsonl ../eras/boost_Qwen2.5-14B-Instruct.jsonl
Lamp (label-agnostic, fixed in advance, no fitting): compare the answer with old experience as is (normal)
vs with period-1 experience quieted in the b40 copy heads (setting QUIET). Lamp fires when the picked label
changes, or when the picked label's lead shrinks by more than DROP of its normal lead.
Gated policy: lamp off -> normal answer; lamp on -> answer of setting FIX (louder new rule)."""
import json, sys, random
import numpy as np
E = {(r["set"], r["arm"], r["probe"]): r for r in map(json.loads, list(open(sys.argv[1]))[1:])}
B = {(r["set"], r["arm"], r["probe"]): r for r in map(json.loads, list(open(sys.argv[2]))[1:])}
am = lambda v: int(np.argmax(v))
def lead(v, p):
    v = np.asarray(v, float); return v[p] - np.delete(v, p).max()
def lamp(sc, quiet, drop):
    p = am(sc["normal"]); q = sc[quiet]
    if am(q) != p: return True
    return lead(q, p) < (1 - drop) * lead(sc["normal"], p)
def lab(r, v): return r["labels"][am(v)]
groups = [("A60", ("D1", "D3"), "trap"), ("A60", ("D2", "D4"), "same answer"),
          ("E_nocopy", ("D1", "D3"), "trap"), ("E_nocopy", ("D2", "D4"), "same answer"),
          ("E_copy", ("D1", "D3"), "trap"), ("E_copy", ("D2", "D4"), "same answer"),
          ("N_same", ("D1", "D2", "D3", "D4"), "rule unchanged")]
def run(quiet, drop, fix):
    print(f"\n=== lamp: quiet={quiet} drop>{drop:.0%}   fix when lamp on: {fix}")
    print(f"{'group':28}{'n':>4}{'lamp on':>9}{'right normal':>14}{'right fix-always':>18}{'right gated':>13}{'old gated':>11}")
    for arm, probes, tag in groups:
        ks = [k for k in E if k[1] == arm and k[2] in probes]
        on = [lamp(E[k]["scores"], quiet, drop) for k in ks]
        rn = [lab(E[k], E[k]["scores"]["normal"]) == E[k]["key"] for k in ks]
        if k_in := all(k in B for k in ks):
            ra = [lab(B[k], B[k]["scores"][fix]) == B[k]["key"] for k in ks]
            g = [lab(B[k], B[k]["scores"][fix] if o else E[k]["scores"]["normal"]) for k, o in zip(ks, on)]
            rg = [x == E[k]["key"] for x, k in zip(g, ks)]
            og = [x == E[k]["old"] and E[k]["old"] != E[k]["key"] for x, k in zip(g, ks)]
            print(f"{arm+' '+'+'.join(probes)+' ('+tag+')':28}{len(ks):>4}{np.mean(on):>9.0%}{np.mean(rn):>14.0%}{np.mean(ra):>18.0%}{np.mean(rg):>13.0%}{np.mean(og):>11.0%}")
        else:  # no FIX scores for this arm: lamp-off cases keep normal; lamp-on cases unknown
            known = [r for r, o in zip(rn, on) if not o]
            print(f"{arm+' '+'+'.join(probes)+' ('+tag+')':28}{len(ks):>4}{np.mean(on):>9.0%}{np.mean(rn):>14.0%}{'n/a':>18}{'= normal on '+str(len(known))+'/'+str(len(ks)):>24}")
for quiet in ("w0.1", "w0"):
    for drop in (0.5,):
        for fix in ("up4_all", "up16_all"):
            run(quiet, drop, fix)
# sensitivity of the lamp to DROP (fire rates only)
print("\nlamp fire rate vs DROP threshold (quiet=w0.1): trap A60 / trap E_nocopy / same-answer A60 / N_same")
for drop in (0.2, 0.35, 0.5, 0.65, 0.8, 1.01):
    f = lambda arm, pr: np.mean([lamp(E[k]["scores"], "w0.1", drop) for k in E if k[1] == arm and k[2] in pr])
    print(f"  drop>{drop:.0%}: {f('A60',('D1','D3')):.0%} / {f('E_nocopy',('D1','D3')):.0%} / {f('A60',('D2','D4')):.0%} / {f('N_same',('D1','D2','D3','D4')):.0%}")
