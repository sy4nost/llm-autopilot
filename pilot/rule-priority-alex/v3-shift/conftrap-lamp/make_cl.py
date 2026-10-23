"""Item set for the frozen lamp test on confident traps (PLAN.md).  python3 -I make_cl.py SEED [--smoke]  -> sets_cl.json / sets_smoke.json
Families chosen by the confident-trap pilot (../confident-traps/rezultaty.md) from fast accuracy only; the lamp was never looked at.
  hindgen  hindsight-neglect, fresh items in the published template            traps 100, controls 40
  redef    redefine-math, published items unused by any earlier run            traps 100 (the dataset has no controls)
  parcels  compound rule amended after 60 examples (newtasks2 generator)       traps 100, controls 40
  bonus    bonus rule amended after 60 examples (newtasks2 generator)         traps 100, controls 40
  D        stated definitions that contradict the real count ('a cube has 7 faces')  traps 100, controls 40
  mmlu     100 questions unused by earlier runs (report-only: false alarms)
No item of the pilot (sets_ct.json) or of any earlier set is reused (by prompt)."""
import json, os, random, re, sys, hashlib, importlib.util
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.join(HERE, "..")
THINK = "\n\nThink it through step by step. Then give your final answer on the last line as: Answer: <answer>"
NT, NC, NM = 100, 40, 100
SMOKE_T, SMOKE_C, SMOKE_M = 4, 2, 4


def mod(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(V3, path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


ct = mod("make_ct", "confident-traps/make_ct.py")   # loads make_ti, make_ta, make_new2 as well
ti, ta, n2 = ct.ti, ct.ta, ct.n2


def think_prompt(x):
    """THINK prompt: the task without 'answer only' instructions, plus the THINK tail"""
    p = x["prompt"]
    if x["mode"] == "label":
        p = p.replace("Answer for the TEST case below. Do not explain. Return only the answer.", "Answer for the TEST case below.")
        p = p.replace("\n\nReturn only the answer.", "")
        assert "Return only" not in p and "Do not explain" not in p, p[-200:]
        return p + THINK
    if x["family"] == "mmlu" or x["family"] == "hindgen":
        return p[:-len("\nAnswer:")] + THINK if p.endswith("\nAnswer:") else p + THINK
    return (p[:-3] if p.endswith(" A:") else p) + THINK


def build(S, smoke):
    rng = random.Random(S)
    nt, nc, nm = (SMOKE_T, SMOKE_C, SMOKE_M) if smoke else (NT, NC, NM)
    old = {x["prompt"] for x in json.load(open(os.path.join(V3, "confident-traps", "sets_ct.json")))["items"]}
    pref = set()
    if not smoke:
        old |= {x["prompt"] for x in json.load(open(os.path.join(HERE, "sets_smoke.json")))["items"]}
        # the first smoke run used an earlier smoke file; its saved clusters are the full prompt, or the part before "TEST CASE:"
        for l in open(os.path.join(HERE, "out_smoke_qwen14.jsonl")):
            x = json.loads(l)
            if "res" in x:
                (pref if x["mode"] == "label" else old).add(x["cluster"])
    out = []
    # hindgen: fresh items (make_ct generator), half with positive expected value
    ct.N_TRAP, ct.N_CTRL = nt, nc
    out += ct.build_hind(rng)
    # redef: published items unused by thinkaloud-indep, general-lamp, the pilot and the smoke
    used = ct.used("redef") | old
    # earlier 14B runs (public/, conflict/) used rows sorted(shuffle(range(900), Random(0))[:200]) of redefine-math (review S2)
    raw = [json.loads(l)["prompt"].replace("\r\n", "\n").replace("\ufeff", "") for l in open(os.path.join(V3, ti.SRC["redef"])) if l.strip()]
    ix = list(range(len(raw))); random.Random(0).shuffle(ix)
    used |= {raw[i] for i in ix[:200]}
    rs = [x for x in ti.load("redef") if x["prompt"] not in used]
    for x in rng.sample(rs, nt):
        c = [c.strip() for c in x["classes"]]; k = x["answer_index"]
        out.append(dict(ct.it("redef", "trap", x["prompt"], c, k, 1 - k, "q"), clu="redef:" + re.sub(r"\d+", "N", x["prompt"].split("Q:")[0])))
    # parcels / bonus: arm A60 of the newtasks2 generator; each generator seed gives 20 sets x (T1, T2) traps and (S1, S2) controls
    P = []
    for j in range(3 if not smoke else 1):
        P += [dict(x, gen=j) for x in n2.build(S * 10 + j) if x["arm"] == "A60"]
    for f in ("parcels", "bonus"):
        for pop, probes, n in (("trap", ("T1", "T2"), nt), ("control", ("S1", "S2"), nc)):
            xs = [x for x in P if x["task"] == f and x["probe"] in probes and x["prompt"] not in old
                  and x["prompt"].split("TEST CASE:")[0] not in pref]
            for x in rng.sample(xs, n):
                L = x["labels"]
                # clusters (review B1): parcels by test case (2 fixed trap cases), bonus by generator set
                cl = f"parcels:{x['prompt'].rsplit('TEST CASE:', 1)[1].strip()}" if f == "parcels" else f"bonus:{x['gen']}:{x['set']}"
                out.append(dict(ct.it(f, pop, x["prompt"], L, L.index(x["key"]), L.index(x["old"]) if pop == "trap" else None, "label"), clu=cl))
    # D: thinkaloud generator, the 15 hooks of the pilot (review S5)
    for pop, src, n in (("trap", "traps", nt), ("control", "agree", nc)):
        k = 0
        while k < n:
            x = ta.build_D(rng, src, 1)[0]
            if x["text"] in old or any(y["prompt"] == x["text"] for y in out):
                continue
            out.append(dict(ct.it("D", pop, x["text"], x["cands"], x["key"], x["habit"], "q"), clu="D:" + x["cell"])); k += 1
    # MMLU
    usedq = {c.split("\nA. ")[0] for c in ct.used("mmlu") | old}   # also the smoke questions (review S4)
    for x in ti.mmlu(rng, nm, usedq):
        out.append(ct.it("mmlu", "control", x["prompt"], [c.strip() for c in x["classes"]], x["answer_index"], None, "q"))
    seen, items = set(old), []
    for x in out:
        assert x["prompt"] not in seen, ("duplicate or reused item", x["family"], x["prompt"][-120:])
        seen.add(x["prompt"]); assert len(set(x["cands"])) == len(x["cands"]) and 0 <= x["key"] < len(x["cands"])
        x["id"] = len(items); x["think"] = think_prompt(x); x["cluster"] = x.pop("clu", x["prompt"]); items.append(x)
    return items


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv; S = int([a for a in sys.argv[1:] if not a.startswith("--")][0])
    assert smoke or (S not in ta.USED and S not in (2, 58, 78, 20261023)), "seed already used"
    items = build(S, smoke)
    fn = os.path.join(HERE, "sets_smoke.json" if smoke else "sets_cl.json")
    json.dump(dict(seed=S, think=THINK, items=items), open(fn, "w"), ensure_ascii=False)
    print(fn, len(items), hashlib.sha256(open(fn, "rb").read()).hexdigest())
    print(dict(sorted(Counter((x["family"], x["pop"]) for x in items).items())))
    print("key balance:", {f: dict(Counter(x["key"] for x in items if x["family"] == f)) for f in ("hindgen", "redef", "parcels", "bonus", "D")})
