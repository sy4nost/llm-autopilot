"""Build the arm specs for why-think (PLAN.md) from saved conftrap-lamp reasonings. python3 -I make_wt.py -> jobs_wt.json
Each arm: prefix = assistant text placed before the final "Answer:" (None = FILLER, built on the GPU side with the same
token count as the reasoning); ko = char ranges the final answer rows may not see: ("u", a, b) in the user text, ("a", a, b) in prefix."""
import json, os, re, random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); CL = os.path.join(HERE, "..", "conftrap-lamp")
MODELS = ["qwen14", "phi4", "olmo13"]; FAMS = ["hindgen", "redef", "parcels", "bonus", "D"]
NMAX, NCTRL = 20, 8


def reasoning(g):
    """saved reasoning without every trailing "Answer:" stub (OLMo often writes "Answer: Answer: X"); all arms then end with
    "\n\nAnswer:" (review B3)"""
    R = g[:g.rindex("Answer:")] if "Answer:" in g else g
    R = R.rstrip()
    while R.endswith("Answer:"):
        R = R[:-len("Answer:")].rstrip()
    return R


def lines(R):
    out, p = [], 0
    for ln in R.split("\n"):
        out.append((p, p + len(ln))); p += len(ln) + 1
    return [x for x in out if R[x[0]:x[1]].strip()]


def cut(R, frac):
    """cut at the end of the line nearest to frac of the characters (review: line ends, not token fractions)"""
    L = lines(R)
    if not L:
        return ""
    e = min(L, key=lambda x: abs(x[1] - frac * len(R)))[1]
    return R[:e]


def concl(R, cands):
    """the conclusion line: last line naming any option label; else the last non-empty line"""
    L = lines(R)
    if not L:
        return 0, 0
    for a, b in reversed(L):
        if any(wb(c).search(R[a:b]) for c in cands):
            return a, b
    return L[-1]


def spans_u(it, U):
    """examples / shots span and the rule (or definition) sentence span, in the THINK user text"""
    f = it["family"]; ex = rule = None
    if f in ("parcels", "bonus"):
        ex = (U.index("Example 01:"), U.index("IMPORTANT UPDATE"))
        a = U.index("New "); b = U.index("same.", a) + len("same."); rule = (a, b)
    elif f == "hindgen":
        ex = (0, U.rindex("Question:"))
    elif f in ("D", "redef"):
        rule = (0, U.index(" Q:"))
    return ex, rule


def wb(label):
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(label) + r"(?![A-Za-z0-9])")


def build():
    items = {x["id"]: x for x in json.load(open(os.path.join(CL, "sets_cl.json")))["items"]}
    jobs = {}
    for mk in MODELS:
        rows = [json.loads(l) for l in open(os.path.join(CL, f"out_main_{mk}.jsonl"))]
        rows = [r for r in rows if "res" in r]
        out = []
        for f in FAMS:
            fx, ct = [], []
            for r in rows:
                if r["family"] != f:
                    continue
                d = int(np.argmax([sum(o) for o in r["res"]["direct"]])); t = int(np.argmax([sum(o) for o in r["res"]["think"]]))
                if r["pop"] == "trap" and d == r["habit"] and t == r["key"]:
                    fx.append(r)
                if r["pop"] == "control" and d == r["key"] and t == r["key"]:
                    ct.append(r)
            rng = random.Random(0)
            sel = [("fixed", r) for r in rng.sample(fx, min(NMAX, len(fx)))] + [("control", r) for r in rng.sample(ct, min(NCTRL, len(ct)))]
            for grp, r in sel:
                it = items[r["id"]]; U = it["think"]; R = reasoning(r["gen_think"]); C = it["cands"]
                ex, rule = spans_u(it, U); la, lb = concl(R, C)
                right = C[it["key"]]; hab = C[it["habit"]] if it["habit"] is not None else None
                third = [c for i, c in enumerate(C) if i != it["key"] and c != hab]
                arms = {"FULL": dict(prefix=R), "NONE": dict(prefix=""), "FILLER": dict(prefix=None, n_of=R, fill=" ."),
                        "FILLTXT": dict(prefix=None, n_of=R, fill=" Let me think about this carefully."),
                        "NOLAST": dict(prefix=R[:la])}
                for k, fr in (("T25", .25), ("T50", .5), ("T75", .75)):
                    P = cut(R, fr); arms[k] = dict(prefix=P, label_written=bool(wb(right).search(P)))
                # REDACT: every option label in the reasoning replaced by a placeholder (steps stay, written answers go)
                red = R
                for c in sorted(C, key=len, reverse=True):
                    red = wb(c).sub("[?]", red)
                arms["REDACT"] = dict(prefix=red)
                L = R[la:lb]
                if hab is not None and wb(right).search(L) and not wb(hab).search(L):
                    arms["SWAP"] = dict(prefix=R[:la] + wb(right).sub(hab, L) + R[lb:])
                    arms["SWAP_ALL"] = dict(prefix=wb(right).sub(hab, R))
                    if third:
                        arms["SWAP3"] = dict(prefix=R[:la] + wb(right).sub(third[0], L) + R[lb:], to=C.index(third[0]))
                if grp == "control" and wb(right).search(L) and len(C) > 1:
                    oth = [c for i, c in enumerate(C) if i != it["key"]][0]
                    if not wb(oth).search(L):
                        arms["SWAP"] = dict(prefix=R[:la] + wb(right).sub(oth, L) + R[lb:], to=C.index(oth))
                for base in ("FULL", "NONE"):
                    sfx = "" if base == "FULL" else "_N"
                    P = arms[base]["prefix"]
                    if ex:
                        arms["KO_EX" + sfx] = dict(prefix=P, ko=[("u",) + ex])
                    if rule:
                        arms["KO_RULE" + sfx] = dict(prefix=P, ko=[("u",) + rule])
                if lb > 0:
                    arms["KO_LAST"] = dict(prefix=R, ko=[("a", la, len(R))])
                    if la > 0:
                        arms["KO_STEPS"] = dict(prefix=R, ko=[("a", 0, la)])
                out.append(dict(id=it["id"], family=f, group=grp, cands=C, key=it["key"], habit=it["habit"], user=U, arms=arms))
        jobs[mk] = out
        from collections import Counter
        print(mk, dict(Counter((x["family"], x["group"]) for x in out)), "SWAP-able fixed:",
              dict(Counter(x["family"] for x in out if x["group"] == "fixed" and "SWAP" in x["arms"])))
    return jobs


if __name__ == "__main__":
    j = build(); json.dump(j, open(os.path.join(HERE, "jobs_wt.json"), "w"), ensure_ascii=False)
    print("arms total", sum(len(x["arms"]) for v in j.values() for x in v))
