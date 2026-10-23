"""Score the public watchman run (PREREG.md). python3 -I score_pub_watch.py pubwatch_Qwen2.5-14B-Instruct.jsonl"""
import json, sys, random
import numpy as np
L = [json.loads(l) for l in open(sys.argv[1])]
R = [r for r in L if "flags" in r]; chk = next(r["checks"] for r in L if "checks" in r)
print("checks: prompts with nothing flagged", chk["n_zero_flag"], "| max diff there", chk["max_zero_flag_diff"], "(must be 0)")
right = lambda r, p: float(int(np.argmax(r["scores"][p])) == r["gold"])
def boot(rows, pol, B=4000):
    d = np.array([right(r, pol) - right(r, "normal") for r in rows]); rng = np.random.default_rng(0)
    xs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return d.mean(), np.percentile(xs, 2.5), np.percentile(xs, 97.5)
verdict = {}
for t in ("hindsight", "algebra", "mmlu"):
    g = [r for r in R if r["task"] == t]
    fl = [x for r in g for x in r["flags"]]
    print(f"\n{t}: n={len(g)}  lamp on {np.mean([any(r['flags']) for r in g]):.0%}  examples flagged {np.mean(fl):.0%}")
    for p in ("normal", "watch_b40", "watch_all"):
        print(f"  {p:10} right {np.mean([right(r, p) for r in g]):.1%}")
    for p in ("watch_b40", "watch_all"):
        e, lo, hi = boot(g, p); print(f"  {p} - normal: {e:+.1%} [{lo:+.1%}, {hi:+.1%}]")
        if p == "watch_b40": verdict[t] = (e, lo, hi)
print("\nPRE-REGISTERED (watch_b40):")
e, lo, hi = verdict["mmlu"]; print(f"  no harm on MMLU: {e:+.1%} need >= -2.0% -> {'PASS' if e >= -0.02 else 'FAIL'}")
for t in ("hindsight", "algebra"):
    e, lo, hi = verdict[t]; print(f"  {t}: independent gain {'SHOWN' if lo > 0 else 'NOT SHOWN'} ({e:+.1%} [{lo:+.1%}, {hi:+.1%}]); harm {'none' if hi >= 0 else 'YES'}")
