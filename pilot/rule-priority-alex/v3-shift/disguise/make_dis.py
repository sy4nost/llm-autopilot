"""Items for the disguise / shake test (PLAN.md v2).  python3 -I make_dis.py S [out.json] [--small]
N  native/make_native.py change_0 (90 sets, generator seeds S.. skipping used ones), with its fictional twin from
   match/make_match.py built in a SEPARATE process (make_native patches make_match's word pools in place).
   N-traps D1,D3 | N-agree D2,D4 (both with the change note) | N-confirm D1-D4 ("reviewed and stays the same")
C  match/make_match.py A_change_60 x D1-D4 (report only)
R  redefine-math test split (same 810 as general-lamp, split seed 12345): R-traps, R-same (T0,1,2,7,8, definition
   removed), R-agree (from T4 / T6, gold = habit).  --dev: the 90 dev questions instead (normal only, for the pre-run).
Every item: family, cell, grp, prompt, cands, key(index), old(index or None), plus 'twin' (N) / 'disg' (R)."""
import json, os, re, subprocess, sys, random

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.join(HERE, "..")
USED = {12345, 25, 32, 185, 20261018, 20261020, 0} | set(range(88, 96))
SHAKES = {
    "bang": "BANG! A loud alarm suddenly goes off, then stops. Silence.",
    "wake": "Wake up. Look at this task as if you are seeing it for the first time.",
    "placebo": "The next line continues the same task, as written above it.",
    "instr": "Follow the rules stated in this task exactly, even if they contradict what you know.",
}
CHANGED = re.compile(r"IMPORTANT UPDATE \(applies to the TEST case below\): Rule 2 has been changed\..*?All other rules stay the same\.\n\n", re.S)
CONFIRM = "IMPORTANT UPDATE (applies to the TEST case below): Rule 2 has been reviewed and stays the same. All other rules stay the same.\n\n"
DMAP = {"π": "zorp", "e": "quib", "+": "§", "÷": "¤"}


def seeds(S, k):
    out, s = [], S
    while len(out) < k:
        if s not in USED:
            out.append(s)
        s += 1
    return out


def emit(kind, sd):
    """runs in a fresh interpreter: native or fictional items for one generator seed"""
    sys.path.insert(0, V3); sys.path.insert(0, os.path.join(V3, "match"))
    if kind == "native":
        import importlib.util
        spec = importlib.util.spec_from_file_location("mn", os.path.join(V3, "native", "make_native.py"))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); mm = m.mm
    else:
        import make_match as mm
    keep = ("change_0", "A_change_60")
    out = [dict(set=x["set"], arm=x["arm"], probe=x["probes"][0]["id"], prompt=x["prompt"], labels=x["labels"],
                key=x["probes"][0]["key"], old=x["probes"][0]["old"]) for x in mm.build(sd) if x["arm"] in keep]
    print(json.dumps(out))


def get(kind, sd):
    r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--emit", kind, str(sd)], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def disguise_R(p):
    """module rule: disguise the redefined symbol X of 'Redefine X as' / 'View X as' everywhere except inside the value"""
    m = re.match(r"(Redefine|View) (\S+) as ", p)
    if not m or m.group(2) not in DMAP:
        return p
    qi = p.index(" Q:")
    X = m.group(2); d, q = p[:qi], p[qi:]
    d = d[:m.start(2)] + DMAP[X] + d[m.end(2):]
    q = re.sub(r"\be\b", DMAP[X], q) if X == "e" else q.replace(X, DMAP[X])
    return d + q


def build_R(part):
    L = [json.loads(l) for l in open(os.path.join(V3, "indep", "data", "redefine-math.jsonl"))]
    dev = set()
    for t in range(9):
        dev |= set(random.Random(12345 + t).sample(range(t * 100, t * 100 + 100), 10))
    out = []
    for i, r in enumerate(L):
        if (i in dev) != (part == "dev"):
            continue
        t = i // 100; p = r["prompt"].replace("\r\n", "\n"); c = [x.strip() for x in r["classes"]]; g = r["answer_index"]
        out.append(dict(family="R-traps", cell="T%d" % t, grp=i, prompt=p, disg=disguise_R(p), cands=c, key=g, old=1 - g))
        if part == "dev":
            continue
        q = p[p.index("Q:"):]
        if t in (0, 1, 2, 7, 8):
            out.append(dict(family="R-same", cell="T%d" % t, grp=q, prompt=q, disg=disguise_R(q), cands=c, key=1 - g, old=None))
        if t == 4:
            n = re.match(r"Redefine π as (\d+)\.", p).group(1); m = "3" + n[1:]
            pa = p.replace(f"Redefine π as {n}.", f"Redefine π as {m}.", 1)
            out.append(dict(family="R-agree", cell="T4", grp=i, prompt=pa, disg=disguise_R(pa), cands=["3", c[g]], key=0, old=None))
        if t == 6 and c[g] != "2":
            ex = re.match(r"Redefine e as ([0-9+]+)\.", p).group(1)
            v = str(sum(int(x) for x in ex.split("+"))); m = "2" + v[-2:].rjust(2, "0")
            pa = p.replace(f"Redefine e as {ex}.", f"Redefine e as {m}.", 1)
            out.append(dict(family="R-agree", cell="T6", grp=i, prompt=pa, disg=disguise_R(pa), cands=["2", c[g]], key=0, old=None))
    return out


def build(S, small=False):
    items = []
    for g, sd in enumerate(seeds(S, 3)):
        nat, fic = get("native", sd), get("fictional", sd)
        assert len(nat) == len(fic)
        for a, b in zip(nat, fic):
            assert (a["set"], a["arm"], a["probe"], a["key"] and a["labels"].index(a["key"]), a["labels"].index(a["old"])) == \
                   (b["set"], b["arm"], b["probe"], b["key"] and b["labels"].index(b["key"]), b["labels"].index(b["old"]))
            assert a["labels"].index(a["key"]) == b["labels"].index(b["key"])
            if small and a["set"] >= 2:
                continue
            if a["arm"] == "change_0":
                road = re.compile(r"\b(barrier|boom|crosswalk|zebra|pedestrians|walkers|people)\b")
                assert road.search(a["prompt"]) and not road.search(b["prompt"]), "twin words not different"
                assert CHANGED.search(a["prompt"]) and CHANGED.search(b["prompt"])
                base = dict(grp=f"{g}-{a['set']}", cands=a["labels"], twin_cands=b["labels"])
                fam = "N-traps" if a["probe"] in ("D1", "D3") else "N-agree"
                items.append(dict(base, family=fam, cell=a["probe"], prompt=a["prompt"], twin=b["prompt"],
                                  key=a["labels"].index(a["key"]), old=a["labels"].index(a["old"])))
                items.append(dict(base, family="N-confirm", cell=a["probe"], prompt=CHANGED.sub(CONFIRM, a["prompt"]),
                                  twin=CHANGED.sub(CONFIRM, b["prompt"]), key=a["labels"].index(a["old"]), old=None))
            else:
                items.append(dict(family="C-rule", cell=a["probe"], grp=f"{g}-{a['set']}", prompt=b["prompt"], cands=b["labels"],
                                  key=b["labels"].index(b["key"]), old=b["labels"].index(b["old"])))
    items += build_R("test")
    if small:
        cnt, keep = {}, []
        for x in items:
            k = (x["family"], x["cell"])
            if x["family"][0] == "R" and cnt.get(k, 0) >= 3:
                continue
            cnt[k] = cnt.get(k, 0) + 1; keep.append(x)
        items = keep
    return items


if __name__ == "__main__":
    if sys.argv[1] == "--emit":
        emit(sys.argv[2], int(sys.argv[3])); sys.exit()
    if sys.argv[1] == "--dev":
        out = os.path.join(HERE, "sets_dev.json")
        json.dump(dict(items=build_R("dev")), open(out, "w"), ensure_ascii=False); print("dev R items ->", out); sys.exit()
    S = int(sys.argv[1]); small = "--small" in sys.argv
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    out = os.path.join(HERE, args[0] if args else "sets_dis.json")
    items = build(S, small)
    for k, x in enumerate(items):
        x["uid"] = k
    json.dump(dict(seed=S, gen_seeds=seeds(S, 3), shakes=SHAKES, items=items), open(out, "w"), ensure_ascii=False)
    from collections import Counter
    print(len(items), "items ->", out, dict(Counter(x["family"] for x in items)))
