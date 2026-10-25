"""why-think report (PLAN v2). python3 -I score_wt.py TAG > report_TAG.txt
Per arm: share RIGHT, share HABIT (fixed) or share following the swapped label (SWAP*), mean margin = logp(right) - logp(habit)
(fixed) in nats; bootstrap 95% over items for the pooled share right."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["qwen14", "phi4", "olmo13"]; FAMS = ["hindgen", "redef", "parcels", "bonus", "D"]
ARMS = ["FULL", "NONE", "FILLER", "FILLTXT", "T25", "T50", "T75", "NOLAST", "REDACT", "SWAP", "SWAP_ALL", "SWAP3",
        "KO_EX", "KO_EX_N", "KO_RULE", "KO_RULE_N", "KO_LAST", "KO_STEPS"]


def follows(x, a):
    m = int(np.argmax(x["res"][a]["s"]))
    if a in ("SWAP", "SWAP_ALL") and x["group"] == "fixed":
        return m == x["habit"]
    return m == x["arms_to"].get(a, -9)


def main(tag):
    rng = np.random.default_rng(0)
    J = json.load(open(os.path.join(HERE, "jobs_wt.json")))
    for mk in MODELS:
        fn = os.path.join(HERE, f"out_{tag}_{mk}.jsonl")
        if not os.path.exists(fn):
            continue
        meta = {j["id"]: j for j in J[mk]}
        R = [json.loads(l) for l in open(fn)]
        for x in R:
            if x["id"] == -1:
                print(mk, "checks:", x["checks"])
        R = [x for x in R if x["id"] != -1]
        for x in R:
            x["arms_to"] = {a: v.get("to") for a, v in meta[x["id"]]["arms"].items() if "to" in v}
            x["tw"] = {a: v.get("label_written") for a, v in meta[x["id"]]["arms"].items() if "label_written" in v}
        print(f"\n===== {mk}")
        for grp in ("fixed", "control"):
            g = [x for x in R if x["group"] == grp]
            print(f" -- {grp} ({len(g)} items). Per arm: right% | {'habit%' if grp == 'fixed' else ''} | follows-swap% | margin | n")
            for a in ARMS:
                h = [x for x in g if a in x["res"]]
                if not h:
                    continue
                am = [int(np.argmax(x["res"][a]["s"])) for x in h]
                r = np.array([m == x["key"] for m, x in zip(am, h)], float)
                b = [r[rng.integers(0, len(r), len(r))].mean() for _ in range(1000)]
                lo, hi = np.percentile(b, [2.5, 97.5])
                hb = np.mean([m == x["habit"] for m, x in zip(am, h)]) if grp == "fixed" else np.nan
                fs = np.mean([follows(x, a) for x in h]) if a.startswith("SWAP") else np.nan
                mg = np.mean([x["res"][a]["s"][x["key"]] - x["res"][a]["s"][x["habit"]] for x in h]) if grp == "fixed" else np.nan
                fam = " ".join(f"{f[:4]} {100*np.mean([m == x['key'] for m, x in zip(am, h) if x['family'] == f]):3.0f}"
                               for f in FAMS if any(x["family"] == f for x in h))
                extra = ""
                if a in ("T25", "T50", "T75"):
                    w = [x["tw"][a] for x in h]; extra = f"  label already written in {100*np.mean(w):.0f}%"
                print(f"   {a:9} right {100*r.mean():5.1f} [{100*lo:4.0f},{100*hi:4.0f}]  habit {100*hb:5.1f}  follows {100*fs:5.1f}  margin {mg:+6.2f}  n {len(h):3} | {fam}{extra}")
        # near ties at FULL
        g = [x for x in R if x["group"] == "fixed"]
        print(f"   FULL near-ties (|margin| < 1 nat): {np.mean([abs(x['res']['FULL']['s'][x['key']] - x['res']['FULL']['s'][x['habit']]) < 1 for x in g]):.0%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
