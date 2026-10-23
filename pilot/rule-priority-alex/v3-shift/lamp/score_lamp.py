"""Score the conflict lamp.  python3 -I score_lamp.py lamp_<model>.jsonl
Held-out sets 15-29 only (the heads were picked on 0-14). Label order in every set is (HOLD, PASS, REVIEW)-type.
Lamps: push vs lens at layer L (raw, and calibrated by subtracting the mean lens vector of sets 0-14);
       push vs the heads-off answer. Gated answer = heads-off answer when the lamp is on, else normal."""
import json, sys

rows = [json.loads(l) for l in open(sys.argv[1])]
arg = lambda v: max(range(3), key=lambda i: v[i])
mean = lambda v: sum(v) / len(v) if v else float("nan")
disc = [r for r in rows if r["set"] < 15]
held = [r for r in rows if r["set"] >= 15]
cal = {L: [mean([r["lens"][L][i] for r in disc]) for i in range(3)] for L in rows[0]["lens"]}
LAMPS = {f"lens{L}": (lambda r, L=L: arg(r["push"]) != arg(r["lens"][L])) for L in rows[0]["lens"]}
LAMPS.update({f"lens{L}cal": (lambda r, L=L: arg(r["push"]) != arg([a - b for a, b in zip(r["lens"][L], cal[L])]))
              for L in rows[0]["lens"]})
LAMPS["off"] = lambda r: arg(r["push"]) != arg(r["off"])
G = {"A chg D1+D3": lambda r: r["arm"] == "A_change_60" and r["probe"] in ("D1", "D3"),
     "B chg D1+D3": lambda r: r["arm"] == "B_change_60" and r["probe"] in ("D1", "D3"),
     "C chg D1+D3": lambda r: r["arm"] == "C_change_60" and r["probe"] in ("D1", "D3"),
     "chg D2+D4": lambda r: "_change" in r["arm"] and r["probe"] in ("D2", "D4"),
     "A nochg": lambda r: r["arm"] == "A_nochange_60",
     "B+C nochg": lambda r: r["arm"] in ("B_nochange_60", "C_nochange_60")}
lab = lambda r, v: r["labels"][arg(v)]
print("habit actually happening (normal answer = old label where old != key):",
      {g: f"{mean([lab(r, r['final']) == r['old'] and r['old'] != r['key'] for r in held if f(r)]):.0%}" for g, f in G.items()})
print("\nlamp ON rate, held-out:  " + " | ".join(G))
for name, lp in LAMPS.items():
    print(f"  {name:10} " + " | ".join(f"{mean([lp(r) for r in held if f(r)]):6.0%}" for f in G.values()))
print("\nright answers, held-out:  " + " | ".join(G) + " | all")
pol = {"normal": lambda r: r["final"], "heads off always": lambda r: r["off"]}
for name in ("lens28cal", "lens29cal", "off"):
    pol[f"lamp {name}"] = lambda r, n=name: r["off"] if LAMPS[n](r) else r["final"]
for name, p in pol.items():
    cells = [mean([lab(r, p(r)) == r["key"] for r in held if f(r)]) for f in G.values()]
    print(f"  {name:17} " + " | ".join(f"{c:6.0%}" for c in cells) + f" | {mean([lab(r, p(r)) == r['key'] for r in held]):.0%}")
print("\nwhat the 20 heads push (held-out): share pushing the OLD label | the RIGHT label")
for g, f in G.items():
    rs = [r for r in held if f(r)]
    print(f"  {g:12} {mean([lab(r, r['push']) == r['old'] for r in rs]):4.0%} | {mean([lab(r, r['push']) == r['key'] for r in rs]):4.0%}")
