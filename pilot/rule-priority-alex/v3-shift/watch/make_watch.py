"""Tasks for the 'watchman' test (PREREG.md): all four unfamiliar types from runs 1 and 2, generated with ONE new seed.
  robot, points    (newtasks/make_new.py, simple change; the model had no habit there -> no-harm check)
  parcels, bonus   (newtasks2/make_new2.py, compound change; strong habit)
Adds to each item the example lines (text spans) and the ground truth per example: does its recorded answer contradict the
rules in force at the test question (taken from the independent text checkers check_new*.py, not from the generators).
python3 -I make_watch.py SEED -> sets_watch.json"""
import json, os, re, sys, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def checker(path):
    """the text-only solver of a check script (everything above its command-line part)"""
    src = open(path).read()
    src = src[:src.index("\nd = json.load")]
    g = {}
    exec(compile(src, path, "exec"), g)
    return g


EX = re.compile(r"Example \d+: (.+?) -> (\S+)\n?")


def truth_rows(chk, kind_parse, it):
    p = it["prompt"]
    if "parse_head" in chk:
        kind, base = chk["parse_head"](p)
    else:
        kind, base = chk["parse_rules"](p)
    m = re.search(r"IMPORTANT UPDATE: (.*)\n", p)
    cur = dict(base)
    if m and "reviewed and stays the same" not in m.group(1):
        t = m.group(1)
        if kind == "robot":
            r = re.search(r"the word '(\w+)' means (\w+)", t); cur[r.group(1)] = r.group(2)
        elif kind == "points":
            r = re.search(r"an? (\w+) token is worth (\d+) point", t); cur[r.group(1)] = int(r.group(2))
        elif kind == "parcels":
            cur["r2"] = re.search(r"insured and, at the same time, (express|standard) goes by", t).group(1)
        else:
            cur["yneg"] = re.search(r"at the same time, (no|an?) \w+ token, add", t).group(1) == "no"
    rows = []
    for mm in EX.finditer(p):
        rows.append(dict(start=mm.start(), end=mm.end(), case=mm.group(1), ans=mm.group(2),
                         stale=chk["solve"](kind, cur, mm.group(1)) != mm.group(2)))
    return rows


def build(seed):
    v3 = os.path.join(HERE, "..")
    g1 = load(os.path.join(v3, "newtasks", "make_new.py"), "make_new")
    g2 = load(os.path.join(v3, "newtasks2", "make_new2.py"), "make_new2")
    c1 = checker(os.path.join(v3, "newtasks", "check_new.py"))
    c2 = checker(os.path.join(v3, "newtasks2", "check_new2.py"))
    items = []
    for gen, chk in ((g1, c1), (g2, c2)):
        for it in gen.build(seed):
            it = {k: it[k] for k in ("task", "set", "arm", "probe", "prompt", "labels", "key", "old", "changed")}
            it["examples"] = truth_rows(chk, None, it)
            items.append(it)
    return items


if __name__ == "__main__":
    seed = int(sys.argv[1])
    items = build(seed)
    out = os.path.join(HERE, sys.argv[2] if len(sys.argv) > 2 else "sets_watch.json")
    json.dump(dict(seed=seed, items=items), open(out, "w"), ensure_ascii=False)
    n_st = {}
    for x in items:
        n_st.setdefault((x["task"], x["arm"]), []).append(sum(e["stale"] for e in x["examples"]))
    print(len(items), "prompts ->", out)
    for k, v in sorted(n_st.items()):
        print(f"  {k[0]:8} {k[1]:9} stale examples per prompt: mean {sum(v)/len(v):5.1f}  min {min(v)}  max {max(v)}")
