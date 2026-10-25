"""judge scoring.  python3 score_j.py main
judge mode: first standalone letter A/B/C in the answer -> chosen source (via the item's letter map); none -> 'none'.
direct mode: committed value among X, X2 (instruction kind), Y (examples kind), truth M (as cycle3 v2) -> correct if = target.
Bootstrap over (fact, seed)."""
import json, os, re, sys, random
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1] if len(sys.argv) > 1 else "main"
items = {i["id"]: i for i in json.load(open(os.path.join(HERE, "sets_j.json")))["items"]}
WORDS = {w: k for k, w in enumerate("zero _one_ two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?|\b(?:" + "|".join(WORDS) + r")\b", re.I)
COMMIT = re.compile(r"(?:answer(?:\s+is)?|:)\s*\**\s*(\d+)", re.I)
MODELS = ["qwen14", "phi4", "olmo13"]; KINDS = ["instruction", "examples", "truth"]
J3_KIND = {"qwen14": "examples", "phi4": "truth", "olmo13": "truth"}   # v2 (review R4): fixed before main run


def nums(s):
    return [int(WORDS[m.group().lower()]) if m.group().lower() in WORDS else int(float(m.group())) for m in NUM.finditer(s)]


def judge_letter(s):
    # v2 (review O2): bare letter, then leading "X)"/"X.", then "answer is X", then first standalone capital not followed by a word
    t = s.strip().strip("*").strip()
    if re.fullmatch(r"\(?([ABC])\)?\.?", t):
        return re.fullmatch(r"\(?([ABC])\)?\.?", t).group(1)
    for pat in (r"^\(?([ABC])[).:]", r"answer(?:\s+is)?\s*:?\s*\**\(?([ABC])\b", r"(?<![A-Za-z])([ABC])(?![A-Za-z])(?!\s+[a-z])"):
        m = re.search(pat, t, re.I if "answer" in pat else 0)
        if m:
            return m.group(1).upper()
    return None


def judge_pick(it, s):
    inv = {v: k for k, v in it["letters"].items()}
    L = judge_letter(s)
    if L:
        return inv[L]
    vs = nums(s)                                   # numbers-only options: a bare number maps to its option
    hit = {src for src, v in it["nums"].items() if v in vs}
    return hit.pop() if len(hit) == 1 else "none"


def direct_pick(it, s):
    echo = set(nums(it["final"])); vs = [v for v in nums(s) if v not in echo]
    cand = {it["M"]: "truth"}
    if it["kind"] == "instruction":
        cand[it["X2"]] = "instruction"; cand[it["X"]] = "examples"   # examples showed the old instruction number X
    elif it["kind"] == "examples":
        cand[it["Y"]] = "examples"; cand[it["X"]] = "instruction"
    else:
        cand[it["X"]] = "instruction"; cand[it["Y"]] = "examples"    # v2: truth kind examples show Y
    hit = [cand[v] for v in vs if v in cand]
    if not hit:
        return "other"
    if len(set(hit)) > 1:
        for m in COMMIT.finditer(s):
            if int(m.group(1)) in cand:
                return cand[int(m.group(1))]
        f = re.match(r"\s*\**\s*(\d+)", s)
        if f and int(f.group(1)) in cand:
            return cand[int(f.group(1))]
    return hit[0]


def boot(cl, f, nb=2000, seed=0):
    rnd = random.Random(seed); keys = sorted(cl)
    res = sorted(f([x for k in (rnd.choice(keys) for _ in keys) for x in cl[k]]) for _ in range(nb))
    return res[int(.025 * nb)], res[int(.975 * nb) - 1]


def pct(xs):
    return 100 * sum(xs) / len(xs) if xs else float("nan")


def main():
    for mk in MODELS:
        p = os.path.join(HERE, f"out_{tag}_{mk}.json")
        if not os.path.exists(p):
            print(mk, "missing"); continue
        out = {int(k): v for k, v in json.load(open(p)).items()}
        P = {i: (judge_pick if items[i]["mode"] == "judge" else direct_pick)(items[i], s) for i, s in out.items()}
        print(f"\n=== {mk} ({len(out)} of {len(items)}) ===")
        for mode in ("judge", "direct"):
            print(f"  {mode}: rows = right source, columns = chosen (instruction / examples / truth / none-other), % ; accuracy")
            for k in KINDS:
                ids = [i for i in P if items[i]["mode"] == mode and items[i]["kind"] == k]
                ch = [P[i] for i in ids]
                print(f"    {k:12s} n={len(ids):3d}  " + " / ".join(f"{pct([c == x for c in ch]):5.1f}" for x in KINDS)
                      + f" / {pct([c in ('none', 'other') for c in ch]):5.1f}   accuracy {pct([c == k for c in ch]):5.1f}")
            ids = [i for i in P if items[i]["mode"] == mode]
            cl = defaultdict(list)
            for i in ids:
                cl[(items[i]["fact"], items[i]["seed"])].append(P[i] == items[i]["kind"])
            lo, hi = boot(cl, pct); print(f"    overall accuracy {pct([P[i] == items[i]['kind'] for i in ids]):.1f} [{lo:.1f},{hi:.1f}]")
        # judge vs direct per kind (paired)
        print("  judge minus direct accuracy, per kind (paired by fact, seed):")
        for k in KINDS:
            cl = defaultdict(list)
            for i in P:
                it = items[i]
                if it["kind"] == k:
                    cl[(it["fact"], it["seed"])].append((it["mode"] == "judge", P[i] == k))
            g = lambda xs: pct([x[1] for x in xs if x[0]]) - pct([x[1] for x in xs if not x[0]])
            lo, hi = boot(cl, g); print(f"    {k:12s} {g([x for v in cl.values() for x in v]):+6.1f} [{lo:+6.1f},{hi:+6.1f}]")
        jl = [P[i] for i in P if items[i]["mode"] == "judge"]
        print(f"  judge replies that are not a bare letter: {pct([not re.fullmatch(r'[*(]*[ABC][).*]*', out[i].strip()) for i in P if items[i]['mode'] == 'judge']):.1f}%")
        k3 = J3_KIND[mk]; cl = defaultdict(list)
        for i in P:
            it = items[i]
            if it["kind"] == k3:
                cl[(it["fact"], it["seed"])].append((it["mode"] == "judge", P[i] == k3))
        g = lambda xs: pct([x[1] for x in xs if x[0]]) - pct([x[1] for x in xs if not x[0]])
        d = g([x for v in cl.values() for x in v]); lo, hi = boot(cl, g)
        print(f"  J3 ({k3} kind, fixed in plan): judge - direct {d:+.1f} [{lo:+.1f},{hi:+.1f}]  "
              + ("BETTER" if lo > 0 else "NOT WORSE" if lo > -10 else "WORSE/unclear"))
        print("  judge letter-position check: chosen letter shares " + " ".join(
            f"{L} {pct([items[i]['letters'][P[i]] == L for i in P if items[i]['mode'] == 'judge' and P[i] in items[i]['letters']]):.0f}%" for L in "ABC"))


if __name__ == "__main__":
    main()
