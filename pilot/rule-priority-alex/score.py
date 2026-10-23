"""Score results/<model>.json against sets.json.  python3 -I score.py <dir>  -> prints tables, writes scores.json
Per answer: parse lines 'P<n>: LABEL'. A probe counts correct only if its label equals the key exactly.
Main numbers per model x arm (over 30 sets; greedy and all 5 answers separately):
  P1P3 correct  - both lure probes right (Alex's primary outcome), share of answers
  wrong PASS on P1/P3 - share of P1/P3 answers labelled with the PASS-word
  all6 - all six probes right;  P5 lure (answered HOLD-word on P5);  format ok
"""
import json, os, re, sys

D = sys.argv[1]
sets = {(x["set"], x["arm"]): x for x in json.load(open(os.path.join(D, "sets.json")))}
ARMS = ["exp", "mixed", "control", "none"]


def parse(ans, labels):
    got = {}
    for m in re.finditer(r"P\s*([1-6])\s*[:\-]\s*\**\s*(?:LABEL\s*[:\-]?\s*)?([A-Za-z]+)", ans, re.I):
        got.setdefault("P" + m.group(1), m.group(2).upper())
    if not got:  # fallback: six bare label lines in P1..P6 order
        lines = [l.strip(" *.").upper() for l in ans.splitlines() if l.strip()]
        if len(lines) >= 6 and all(l in labels for l in lines[:6]):
            got = {f"P{j}": l for j, l in enumerate(lines[:6], 1)}
    return got


def score_one(ans, item):
    H, P, R = item["labels"]
    got = parse(ans, item["labels"])
    by_id = {pr["id"]: (got.get(pr["pos"]), pr["key"]) for pr in item["probes"]}
    ok = {i: g == k for i, (g, k) in by_id.items()}
    return dict(fmt=len(got) == 6 and all(g in (H, P, R) for g, _ in by_id.values()),
                p13=ok["P1"] and ok["P3"], all6=all(ok.values()),
                wrongpass=sum(by_id[i][0] == P for i in ("P1", "P3")) / 2,
                p5hold=by_id["P5"][0] == H, ok=ok)


rows = {}
for f in sorted(os.listdir(os.path.join(D, "results"))):
    res = json.load(open(os.path.join(D, "results", f)))
    m = res["model"].split("/")[-1].replace("Qwen2.5-", "").replace("-Instruct", "")
    for arm in ARMS:
        for mode in ("greedy", "all5"):
            sc = []
            for a in res["answers"]:
                if a["arm"] != arm:
                    continue
                answers = [a["greedy"]] if mode == "greedy" else [a["greedy"]] + a["samples"]
                sc += [score_one(t, sets[(a["set"], arm)]) for t in answers]
            n = len(sc)
            r = dict(n=n, p13=sum(s["p13"] for s in sc) / n, all6=sum(s["all6"] for s in sc) / n,
                     wrongpass=sum(s["wrongpass"] for s in sc) / n, p5hold=sum(s["p5hold"] for s in sc) / n,
                     fmt=sum(s["fmt"] for s in sc) / n,
                     probe={i: sum(s["ok"][i] for s in sc) / n for i in ("P1", "P2", "P3", "P4", "P5", "P6")})
            rows[(m, arm, mode)] = r
json.dump({"|".join(k): v for k, v in rows.items()}, open(os.path.join(D, "scores.json"), "w"), indent=1)

order = ["0.5B", "1.5B", "3B", "7B", "14B"]
for mode in ("greedy", "all5"):
    print(f"\n== {mode}: P1&P3 right / wrong PASS on P1,P3 / all 6 right / P5 lured to HOLD / format ok ==")
    for m in order:
        print(m.ljust(5), " | ".join(
            f"{arm}: {rows[(m, arm, mode)]['p13']:.0%}/{rows[(m, arm, mode)]['wrongpass']:.0%}/"
            f"{rows[(m, arm, mode)]['all6']:.0%}/{rows[(m, arm, mode)]['p5hold']:.0%}/{rows[(m, arm, mode)]['fmt']:.0%}"
            for arm in ARMS if (m, arm, mode) in rows))
print("\n== all5: per-probe accuracy ==")
for m in order:
    for arm in ARMS:
        if (m, arm, "all5") in rows:
            p = rows[(m, arm, "all5")]["probe"]
            print(m.ljust(5), arm.ljust(8), " ".join(f"{i}={v:.0%}" for i, v in p.items()))
