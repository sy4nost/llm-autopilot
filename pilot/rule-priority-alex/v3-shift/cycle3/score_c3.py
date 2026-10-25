"""cycle3 scoring.  python3 score_c3.py main
Committed value (as many-instr/mix v2): candidates = X (instruction), X2 (correction), Y (examples); one distinct candidate
-> it; several -> number after "answer is"/"Answer:"/":" if a candidate, else the candidate the answer starts with, else multi;
no candidate -> truth if M appears, else other. Numbers compared after rounding to 2 decimals (pi 3.14159 = 3.14).
Bootstrap over (fact, seed)."""
import json, os, re, sys, random
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1] if len(sys.argv) > 1 else "main"
items = {i["id"]: i for i in json.load(open(os.path.join(HERE, "sets_c3.json")))["items"]}
WORDS = {w: k for k, w in enumerate("zero _one_ two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?|\b(?:" + "|".join(WORDS) + r")\b", re.I)
COMMIT = re.compile(r"(?:answer(?:\s+is)?|:)\s*\**\s*(\d+(?:\.\d+)?)", re.I)
MODELS = ["qwen14", "phi4", "olmo13"]
ARMS = ["T_alone", "IT_instr", "II_correction", "ET_examples_N2", "ET_examples_N4", "CE_neutral_examples_N2", "CE_neutral_examples_N4",
        "GE_game_examples_N2", "GE_game_examples_N4",
        "IE_instr_examples_N2", "IE_instr_examples_N4"]
LABS = ("X", "X2", "Y", "truth", "other")


def r2(v):
    return round(float(v), 2)


def nums(s):
    return [r2(WORDS[m.group().lower()]) if m.group().lower() in WORDS else r2(m.group()) for m in NUM.finditer(s)]


def cls(it, s):
    """v2 (review R1, R2): no list-marker strip; truth M is a candidate like the others. Candidates present in the arm:
    X (IT/II/IE), X2 (II), Y (examples arms), truth M (always). One distinct candidate -> it; several -> the number after
    "answer is"/"Answer:"/":" if a candidate, else the first candidate the answer starts with, else the first candidate
    mentioned; none -> other."""
    echo = set(nums(it["final"])); vs = [v for v in nums(s) if v not in echo]
    a = it["arm"]
    present = ({"X"} if a.startswith(("IT", "II", "IE")) else set()) | ({"X2"} if a.startswith("II") else set()) | ({"Y"} if "examples" in a else set())
    cand = {r2(it[k]): k for k in present}; cand[r2(it["M"])] = "truth"
    hit = [cand[v] for v in vs if v in cand]
    if not hit:
        return "other"
    if len(set(hit)) == 1:
        return hit[0]
    if len(set(hit)) >= 2:
        for m in COMMIT.finditer(s):
            if r2(m.group(1)) in cand:
                return cand[r2(m.group(1))]
        f = re.match(r"\s*\**\s*(\d+(?:\.\d+)?)", s)
        if f and r2(f.group(1)) in cand:
            return cand[r2(f.group(1))]
        return hit[0]
    return "other"


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


def diff(C, arms, a, b, facts=None):
    """share(label a) - share(label b) within arms; pairs per (fact, seed); bootstrap"""
    cl = defaultdict(list)
    for i, c in C.items():
        it = items[i]
        if it["arm"] in arms and (facts is None or it["fact"] in facts):
            cl[(it["fact"], it["seed"])].append((c == a) - (c == b))
    f = lambda xs: 100 * sum(xs) / len(xs) if xs else float("nan")
    lo, hi = boot(cl, f); return f([x for v in cl.values() for x in v]), lo, hi


TALLY = {"cycle": 0, "C4": 0, "n": 0}


def main():
    INT = ["hexagon", "spider", "week", "year", "hand"]
    for mk in MODELS:
        p = os.path.join(HERE, f"out_{tag}_{mk}.json")
        if not os.path.exists(p):
            print(mk, "missing"); continue
        out = {int(k): v for k, v in json.load(open(p)).items()}
        C = {i: cls(items[i], s) for i, s in out.items()}
        print(f"\n=== {mk} ({len(out)} of {len(items)}) ===")
        for scope, facts in (("integer facts", INT), ("pi", ["pi"])):
            print(f"  [{scope}] shares %: " + " / ".join(LABS))
            for arm in ARMS:
                ids = [i for i in C if items[i]["arm"] == arm and items[i]["fact"] in facts]
                print(f"    {arm:22s} n={len(ids):3d}  " + "  ".join(f"{k} {pct([C[i] == k for i in ids]):5.1f}" for k in LABS))
        print("  pre-registered (integer facts; bootstrap over fact x seed):")
        c1 = diff(C, ["IT_instr"], "X", "truth", INT); c2 = diff(C, ["IE_instr_examples_N4"], "Y", "X", INT)
        c3 = diff(C, ["ET_examples_N4"], "truth", "Y", INT); c4 = None
        cl = defaultdict(list)
        for i, c in C.items():
            it = items[i]
            if it["fact"] in INT and it["arm"] in ("CE_neutral_examples_N2", "CE_neutral_examples_N4", "GE_game_examples_N2", "GE_game_examples_N4"):
                cl[(it["fact"], it["seed"])].append((it["arm"].startswith("GE"), c == "truth"))
        g = lambda xs: pct([x[1] for x in xs if x[0]]) - pct([x[1] for x in xs if not x[0]])
        lo, hi = boot(cl, g); c4 = (g([x for v in cl.values() for x in v]), lo, hi)
        for name, (d, lo, hi), ok in (("C1 instruction beats truth  (IT: X% - truth%)", c1, c1[1] > 0),
                                      ("C2 example beats instruction (IE N4: Y% - X%)", c2, c2[1] > 0),
                                      ("C3 truth beats example     (ET N4: truth% - Y%)", c3, c3[1] > 0),
                                      ("C4 number game removes truth (GE - CE truth%, N2+N4)", c4, c4[2] < 0)):
            print(f"    {name}: {d:+6.1f} [{lo:+6.1f},{hi:+6.1f}]  {'HOLDS' if ok else 'no'}")
        cyc = c1[1] > 0 and c2[1] > 0 and c3[1] > 0; TALLY["cycle"] += cyc; TALLY["C4"] += c4[2] < 0; TALLY["n"] += 1
        tt = pct([C[i] == "truth" for i in C if items[i]["arm"] == "T_alone" and items[i]["fact"] in INT])
        print(f"    cycle for {mk}: {'YES' if cyc else 'no'}   (T_alone truth {tt:.0f}%{'; UNTESTABLE <80' if tt < 80 else ''})")
        d, lo, hi = diff(C, ["ET_examples_N2", "ET_examples_N4"], "truth", "truth", INT)
        e = [pct([C[i] == "truth" for i in C if items[i]["arm"].startswith(p) and items[i]["fact"] in INT]) for p in ("ET", "CE", "GE")]
        print(f"    truth% ET / CE / GE (N2+N4): {e[0]:.1f} / {e[1]:.1f} / {e[2]:.1f}   other {pct([C[i] == 'other' for i in C if items[i]['fact'] in INT]):.1f}%")
        d, lo, hi = diff(C, ["II_correction"], "X2", "X", INT); print(f"    II sanity (last correction X2% - first X%): {d:+.1f} [{lo:+.1f},{hi:+.1f}]")

    print(f"\nOVERALL: cycle in {TALLY['cycle']}/{TALLY['n']} models (need 2); C4 in {TALLY['C4']}/{TALLY['n']} (need 2)")


if __name__ == "__main__":
    main()
