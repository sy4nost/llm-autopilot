"""python3 -I score_conflict.py conflict_<model>.jsonl
Per task (held-out own facts + public): lamp-on rate; accuracy normal / brake always / brake only when lamp on, per alpha.
Main alpha = the one chosen on held-out own facts: best conflict accuracy with consistent accuracy not lower than normal - 2 pts."""
import json, random, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
meta = [r for r in rows if r.get("meta")][0]
rows = [r for r in rows if not r.get("meta") and r["test"]]
am = lambda v: max(range(len(v)), key=lambda k: v[k])
A = sorted(k for k in rows[0] if k.startswith("brake"))
ok = lambda r, k: am(r[k]) == r["gold"]
gate = lambda r, k: am(r[k] if r["lamp"] > 0.5 else r["normal"]) == r["gold"]
mean = lambda v: sum(v) / len(v) if v else float("nan")
rng = random.Random(1)
def ci(d):
    bs = sorted(mean([rng.choice(d) for _ in d]) for _ in range(2000)); return bs[50], bs[1949]
print(f"probe layer {meta['layer']}, held-out probe accuracy by layer max {max(meta['probe_acc']):.2f}")
tasks = list(dict.fromkeys(r["task"] for r in rows))
by = {t: [r for r in rows if r["task"] == t] for t in tasks}
pick = None
for k in A:
    c, s = mean([ok(r, k) for r in by["own-conflict"]]), mean([ok(r, k) for r in by["own-consistent"]])
    if s >= mean([ok(r, "normal") for r in by["own-consistent"]]) - 0.02 and (pick is None or c > pick[1]):
        pick = (k, c)
main = pick[0] if pick else A[0]
print("chosen on own facts:", main)
print(f"{'task':26} {'n':>4} {'lamp on':>8} {'normal':>7} " + " ".join(f"{k+' all':>12} {k+' lamp':>12}" for k in A))
for t in tasks:
    R = by[t]
    print(f"{t:26} {len(R):4} {mean([r['lamp'] > 0.5 for r in R]):8.0%} {mean([ok(r, 'normal') for r in R]):7.1%} "
          + " ".join(f"{mean([ok(r, k) for r in R]):12.1%} {mean([gate(r, k) for r in R]):12.1%}" for k in A))
print(f"\nmain {main}: gated - normal, 95% interval")
for t in tasks:
    d = [gate(r, main) - ok(r, "normal") for r in by[t]]; lo, hi = ci(d)
    d2 = [ok(r, main) - ok(r, "normal") for r in by[t]]; lo2, hi2 = ci(d2)
    print(f"  {t:26} lamp-gated {mean(d):+.1%} [{lo:+.1%}, {hi:+.1%}]   always {mean(d2):+.1%} [{lo2:+.1%}, {hi2:+.1%}]")
