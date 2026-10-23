"""Generated families for the general-lamp test (PLAN.md v2.1). python3 -I make_gl.py S [out.json] [--small]
  C-rule    match/make_match.py   90 sets  (3 generator seeds)  A/B/C_change_60 x D1-D4
  C-parcels newtasks2/make_new2.py 80 sets (4 seeds)            A60, E_nocopy x T1 T2 S1 S2
  C-bonus   newtasks2/make_new2.py 160 sets (8 seeds)           A60, E_nocopy x T1 T2 S1 S2
  P-native  native/make_native.py pools, 90 sets (3 seeds)      change_0 x D1-D4      (report only)
Generator seeds are S, S+1, ... skipping every seed used before in this project.
Every item: family, cell (arm|probe), grp (set id, unique within family), prompt, cands, key, old.
--small: 2 sets per family (technical smoke)."""
import json, os, sys, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.join(HERE, "..")
USED = {12345, 25, 32, 185, 20261018, 20261020, 0}


def seeds(S, k):
    out, s = [], S
    while len(out) < k:
        if s not in USED:
            out.append(s)
        s += 1
    return out


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def build(S, small=False):
    items = []
    sys.path.insert(0, V3)
    sys.path.insert(0, os.path.join(V3, "match"))
    import make_match as mm                       # C-rule first: make_native patches this module's word pools
    for g, sd in enumerate(seeds(S, 3)):
        for x in mm.build(sd):
            pr = x["probes"][0]
            if x["arm"] in ("A_change_60", "B_change_60", "C_change_60") and (not small or x["set"] < 1):
                items.append(dict(family="C-rule", cell=x["arm"] + "|" + pr["id"], grp=f"{g}-{x['set']}", prompt=x["prompt"],
                                  cands=list(x["labels"]), key=pr["key"], old=pr["old"]))
    n2 = load(os.path.join(V3, "newtasks2", "make_new2.py"), "make_new2_gl")
    need = {"parcels": 80, "bonus": 160}
    for g, sd in enumerate(seeds(S, 8)):
        for x in n2.build(sd):
            if x["arm"] not in ("A60", "E_nocopy"):
                continue
            if x["set"] + 20 * g >= (2 if small else need[x["task"]]):
                continue
            items.append(dict(family="C-" + x["task"], cell=x["arm"] + "|" + x["probe"], grp=f"{g}-{x['set']}", prompt=x["prompt"],
                              cands=list(x["labels"]), key=x["key"], old=x["old"]))
    nat = load(os.path.join(V3, "native", "make_native.py"), "make_native_gl")   # patches mm pools to road words
    for g, sd in enumerate(seeds(S, 3)):
        for x in nat.mm.build(sd):
            pr = x["probes"][0]
            if x["arm"] == "change_0" and (not small or x["set"] < 2):
                items.append(dict(family="P-native", cell=x["arm"] + "|" + pr["id"], grp=f"{g}-{x['set']}", prompt=x["prompt"],
                                  cands=list(x["labels"]), key=pr["key"], old=pr["old"]))
    return items


if __name__ == "__main__":
    S = int(sys.argv[1])
    small = "--small" in sys.argv
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    out = os.path.join(HERE, args[0] if args else "sets_gl.json")
    items = build(S, small)
    json.dump(dict(seed=S, gen_seeds=seeds(S, 8), items=items), open(out, "w"), ensure_ascii=False)
    from collections import Counter
    c = Counter(x["family"] for x in items)
    print(len(items), "prompts ->", out, dict(c), "| habit possible (old != key):",
          {f: sum(x["old"] != x["key"] for x in items if x["family"] == f) for f in c})
