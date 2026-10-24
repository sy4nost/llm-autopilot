"""many-instr/mix scoring.  python3 score_mix.py main
First number in the answer not echoed from the question -> last (5th, valid instruction) / ex (the number the examples showed)
/ old (another instruction) / memory / other; "multi" if 2+ different candidate numbers (instructions or example value) appear.
Bootstrap over (fact, seed) clusters."""
import json, os, re, sys, random
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1] if len(sys.argv) > 1 else "main"
items = {i["id"]: i for i in json.load(open(os.path.join(HERE, "sets_mix.json")))["items"]}
WORDS = {w: k for k, w in enumerate("zero _one_ two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?|\b(?:" + "|".join(WORDS) + r")\b", re.I)
MODELS = ["qwen14", "phi4", "olmo13"]; NS = [0, 1, 2, 3, 5, 8]; ARMS = ["E1_first", "E4_penultimate", "EX_new"]
LABS = ("last", "ex", "old", "memory", "multi", "other")


def nums(s):
    return [float(WORDS[m.group().lower()]) if m.group().lower() in WORDS else float(m.group()) for m in NUM.finditer(s)]


COMMIT = re.compile(r"(?:answer(?:\s+is)?|:)\s*\**\s*(\d+)", re.I)


def committed(it, s):
    """v2 (review): the committed value. Candidates = instruction numbers + example number. One distinct candidate -> it.
    Several -> the number after "answer is"/"Answer:"/":" if it is a candidate, else a candidate the answer starts with,
    else multi. No candidate -> memory if the true answer appears, else other."""
    s = re.sub(r"^\s*\d+[.)]\s+", "", s)
    echo = set(nums(it["final"])); vs = [int(v) for v in nums(s) if v not in echo and v == int(v)]
    cands = set(it["vals"]) | ({it["ex"]} if it["ex"] is not None else set())
    hit = [v for v in vs if v in cands]
    if len(set(hit)) == 1:
        return hit[0]
    if len(set(hit)) >= 2:
        for m in COMMIT.finditer(s):
            if int(m.group(1)) in cands:
                return int(m.group(1))
        first = re.match(r"\s*\**\s*(\d+)", s)
        if first and int(first.group(1)) in cands:
            return int(first.group(1))
        return "multi"
    return "memory" if it["M"] in vs else "other"


def cls(it, s):
    v = committed(it, s)
    if v in ("multi", "memory", "other"):
        return v
    if v == it["vals"][-1]:
        return "last"
    if it["ex"] is not None and v == it["ex"]:
        return "ex"
    return "old"


def boot(cl, f, nb=2000, seed=0):
    if not cl:
        return float("nan"), float("nan")
    rnd = random.Random(seed); keys = sorted(cl)
    res = sorted(r for r in (f([x for k in (rnd.choice(keys) for _ in keys) for x in cl[k]]) for _ in range(nb)) if r == r)
    if len(res) < .95 * nb:
        return float("nan"), float("nan")
    m = len(res); return res[int(.025 * m)], res[int(.975 * m) - 1]


def pct(xs):
    return 100 * sum(xs) / len(xs) if xs else float("nan")


def main():
    for mk in MODELS:
        p = os.path.join(HERE, f"out_{tag}_{mk}.json")
        if not os.path.exists(p):
            print(mk, "missing"); continue
        out = {int(k): v for k, v in json.load(open(p)).items()}
        C = {i: cls(items[i], s) for i, s in out.items()}
        print(f"\n=== {mk} ({len(out)} of {len(items)}) ===")
        n0 = [i for i in C if items[i]["N"] == 0]
        print(f"  N=0 (instructions only) n={len(n0)}: " + "  ".join(f"{k} {pct([C[i] == k for i in n0]):5.1f}" for k in LABS))
        for arm in ARMS:
            print(f"  {arm}")
            for N in NS[1:]:
                ids = [i for i in C if items[i]["arm"] == arm and items[i]["N"] == N]
                print(f"    N={N}  n={len(ids):3d}  " + "  ".join(f"{k} {pct([C[i] == k for i in ids]):5.1f}" for k in LABS))
        print("  pre-registered contrasts (bootstrap over fact x seed):")
        base = {(items[i]["fact"], items[i]["seed"]): C[i] == "last" for i in n0}
        n0ans = {(items[i]["fact"], items[i]["seed"]): committed(items[i], out[i]) for i in n0}
        for arm in ARMS:
            cl = defaultdict(list)
            for i in C:
                it = items[i]
                if it["arm"] == arm and it["N"] == 3 and (it["fact"], it["seed"]) in base:
                    cl[(it["fact"], it["seed"])].append((C[i] == "last") - base[(it["fact"], it["seed"])])
            f = lambda xs: 100 * sum(xs) / len(xs) if xs else float('nan'); lo, hi = boot(cl, f)
            print(f"    P1 {arm:15s} last% N=3 minus N=0: {f([x for v in cl.values() for x in v]):+6.1f} [{lo:+6.1f},{hi:+6.1f}]"
                  + "   (N=1: %+.1f, N=2: %+.1f)" % tuple(pct([C[i] == 'last' for i in C if items[i]['arm'] == arm and items[i]['N'] == N]) - pct(list(base.values())) for N in (1, 2)))
        for arm in ("E1_first", "E4_penultimate"):
            cl = defaultdict(list)
            for i in C:
                it = items[i]
                if it["arm"] in (arm, "EX_new") and it["N"] in (1, 2, 3):
                    cl[(it["fact"], it["seed"])].append((it["arm"] == arm, C[i] == "ex"))
            g = lambda xs: pct([x[1] for x in xs if x[0]]) - pct([x[1] for x in xs if not x[0]])
            lo, hi = boot(cl, g)
            k = 0 if arm == "E1_first" else 3
            b0 = pct([n0ans[key] == items[i]["vals"][k] for i in n0 for key in [(items[i]["fact"], items[i]["seed"])]])
            print(f"    P2 {arm:15s} example-copy% minus EX_new, N=1..3: {g([x for v in cl.values() for x in v]):+6.1f} [{lo:+6.1f},{hi:+6.1f}]"
                  f"   (N=0 rate of that number: {b0:.1f}%)")
        print("  headline: ex% minus last% per N (paired CI); first N where ex% > last% with CI above 0")
        for arm in ARMS:
            row, first = [], None
            for N in NS[1:]:
                cl = defaultdict(list)
                for i in C:
                    it = items[i]
                    if it["arm"] == arm and it["N"] == N:
                        cl[(it["fact"], it["seed"])].append((C[i] == "ex") - (C[i] == "last"))
                f = lambda xs: 100 * sum(xs) / len(xs) if xs else float('nan'); lo, hi = boot(cl, f); d = f([x for v in cl.values() for x in v])
                row.append(f"N{N} {d:+.0f}[{lo:+.0f},{hi:+.0f}]")
                if first is None and lo == lo and lo > 0:
                    first = N
            print(f"    {arm:15s} " + "  ".join(row) + f"   -> first N: {first}")
        mm = [C[i] for i in C if items[i]["N"] >= 1]
        print(f"  multi {pct([c == 'multi' for c in mm]):.1f}%  other {pct([c == 'other' for c in mm]):.1f}%  memory {pct([c == 'memory' for c in mm]):.1f}% (N>=1)")


if __name__ == "__main__":
    main()
