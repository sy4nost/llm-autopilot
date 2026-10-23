"""Build items_g.json for the Gurdjieff sensor (PLAN.md v2) from the two saved think-aloud runs.  python3 -I make_g.py
Answers come from each run's frozen scorer (own: score_ta.py forced choice; public: score_ti.py parse + forced fallback).
Nothing from the data files is executed; the scorers are our own code."""
import contextlib, io, json, os, runpy, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.join(HERE, "..")
HARM = {("R", "cfright"), ("R", "agree"), ("N", "agree"), ("N", "confirm"), ("K", "agree"), ("K", "plain"), ("D", "agree"),
        ("D", "plain"), ("memo", "control"), ("neqa", "control"), ("hind", "control"), ("alg", "control"), ("mmlu", "trap"),
        ("modus", "trap"), ("quote", "trap")}


def scored(script, out):
    sys.argv = [script, out]
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(script)["rows"]


def surface(t):
    return [len(t), t.count("\n"), sum(ch.isdigit() for ch in t), t.lower().count(" not ") + t.lower().count("n't"),
            t.count("Answer"), t.count("?")]


items = []
own = scored(os.path.join(V3, "thinkaloud/score_ta.py"), os.path.join(V3, "thinkaloud/out_main.jsonl"))
sets = {x["uid"]: x for x in json.load(open(os.path.join(V3, "thinkaloud/sets_ta.json")))["items"]}
for r in own:
    s = sets[r["uid"]]; d, t = r["ans"]["direct"], r["ans"]["think"]
    items.append(dict(pool="own", id=f"own-{r['uid']}", fam=r["family"], pop=f"{r['family']}/{r['pop']}", cell=f"{r['family']}/{r['pop']}/{r['cell']}",
                      harm=(r["family"], r["pop"]) in HARM, cluster=f"own-{r['cell'] if r['family'] in ('K', 'D') else r['grp']}",
                      mode="raw" if r["family"] == "N" else "tail", user=s["direct"], cue="The answer is",
                      opts=list(r["cands"]) if r["family"] == "N" else [" " + c for c in r["cands"]],
                      saved_direct=[round(sum(x), 4) for x in r["res"]["direct"]], direct=d, think=t, key=r["key"],
                      y=int(t != d), yplus=int(t == r["key"] and d != r["key"]), ok_d=int(d == r["key"]), ok_t=int(t == r["key"]),
                      surf=surface(s["direct"])))
pub = scored(os.path.join(V3, "thinkaloud-indep/score_ti.py"), os.path.join(V3, "thinkaloud-indep/out_main.jsonl"))
sets = {x["id"]: x for x in json.load(open(os.path.join(V3, "thinkaloud-indep/sets_ti.json")))["items"]}
for r in pub:
    if r["task"] == "redef":  # PLAN v2.4: leaks own R templates
        continue
    s = sets[r["id"]]; d, t = r["ans"]["direct"], r["ans"]["think"]
    items.append(dict(pool="pub", id=f"pub-{r['id']}", fam=r["task"], pop=f"{r['task']}/{r['pop']}", cell=f"{r['task']}/{r['pop']}",
                      harm=(r["task"], r["pop"]) in HARM, cluster=f"pub-{r['task']}-{r['cluster']}",
                      mode="tail", user=s["direct"], cue=s["stem"].strip() if s["stem"] else "The answer is",
                      opts=[" " + c for c in r["cands"]], saved_direct=[round(sum(x), 4) for x in r["res"]["direct"]],
                      direct=d, think=t, key=r["key"], y=int(t != d), yplus=int(t == r["key"] and d != r["key"]),
                      ok_d=int(d == r["key"]), ok_t=int(t == r["key"]), surf=surface(s["direct"])))
json.dump(dict(items=items), open(os.path.join(HERE, "items_g.json"), "w"), ensure_ascii=False)
from collections import Counter
print(len(items), Counter(x["pool"] for x in items))
for p in ("own", "pub"):
    xs = [x for x in items if x["pool"] == p]
    print(p, "y", np.mean([x["y"] for x in xs]).round(3), "acc direct", np.mean([x["ok_d"] for x in xs]).round(3),
          "think", np.mean([x["ok_t"] for x in xs]).round(3), "harm items", sum(x["harm"] for x in xs))
