"""many-instr scoring.  python3 score_mi.py main
F1/F2: first number in the answer not echoed from the question -> position of the instruction it matches (1..K), memory (true
answer M), or other. F3: first word of the answer (letters only, upper-cased) -> position among the K words; else "word
elsewhere" if any instruction word appears later; else none. Also: is the true answer M present anywhere (F3 correctness).
Bootstrap over (fact, seed) clusters."""
import json, os, re, sys, random
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1] if len(sys.argv) > 1 else "main"
items = {i["id"]: i for i in json.load(open(os.path.join(HERE, "sets_mi.json")))["items"]}
WORDS = {w: k for k, w in enumerate("zero _one_ two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?|\b(?:" + "|".join(WORDS) + r")\b", re.I)
MODELS = ["qwen14", "phi4", "olmo13"]; KS = [1, 2, 5, 10]; FMTS = ["F1_one_message", "F2_turns", "F3_words", "F4_uniform"]


def nums(s):
    return [float(WORDS[m.group().lower()]) if m.group().lower() in WORDS else float(m.group()) for m in NUM.finditer(s)]


def cls(it, s):
    """returns (label, position) ; label in first / last / middle / multi / memory / other / elsewhere / none ;
    multi = the answer names 2+ different instruction values (recaps) -> excluded from P1/P2 numerators"""
    K, vals = it["K"], it["vals"]
    if it["fmt"] == "F3_words":
        w = re.findall(r"[A-Za-z]+", s)
        if len({x.upper() for x in w if x.upper() in vals}) >= 2:
            return "multi", None
        pos = vals.index(w[0].upper()) + 1 if w and w[0].upper() in vals else None
        if pos is None:
            for x in w:
                if x.upper() in vals:
                    return "elsewhere", vals.index(x.upper()) + 1
            return "none", None
    else:
        echo = set(nums(it["final"])); vs = [v for v in nums(s) if v not in echo]
        if len({int(v) for v in vs if v == int(v) and int(v) in vals}) >= 2:
            return "multi", None
        if not vs:
            return "other", None
        v = vs[0]
        pos = vals.index(int(v)) + 1 if v == int(v) and int(v) in vals else None
        if pos is None:
            return ("memory" if v == it["M"] else "other"), None
    return ("last" if pos == K else "first" if pos == 1 else "middle"), pos


def boot(cl, f, nb=2000, seed=0):
    rnd = random.Random(seed); keys = sorted(cl)
    res = [f([x for k in (rnd.choice(keys) for _ in keys) for x in cl[k]]) for _ in range(nb)]
    res = sorted(r for r in res if r == r)  # drop NaN resamples
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
        R = {i: cls(items[i], s) for i, s in out.items() if items[i]["fmt"] != "Z_baseline"}
        print(f"\n=== {mk} ({len(out)} of {len(items)}) ===")
        print("  baseline: " + "; ".join(f"{items[i]['fact']}: {out[i].strip()[:25]!r}" for i in sorted(out) if items[i]["fmt"] == "Z_baseline"))
        for fmt in FMTS:
            print(f"  {fmt}   shares %: last / first / middle / memory / other(+none,elsewhere)" + ("   | true answer present" if fmt == "F3_words" else ""))
            for K in KS:
                ids = [i for i in R if items[i]["fmt"] == fmt and items[i]["K"] == K]
                if not ids:
                    continue
                lab = [R[i][0] for i in ids]
                line = (f"    K={K:2d} n={len(ids):3d}  last {pct([l == 'last' for l in lab]):5.1f}  first {pct([l == 'first' for l in lab]) if K > 1 else 0:5.1f}  "
                        f"middle {pct([l == 'middle' for l in lab]):5.1f}  memory {pct([l == 'memory' for l in lab]):5.1f}  other {pct([l in ('other', 'none') for l in lab]):5.1f}  multi {pct([l == 'multi' for l in lab]):5.1f}  elsewhere {pct([l == 'elsewhere' for l in lab]):5.1f}")
                if fmt == "F3_words":
                    line += f"   | M present {pct([any(v == items[i]['M'] for v in nums(out[i])) for i in ids]):5.1f}"
                if K == 10:
                    pos = [R[i][1] for i in ids if R[i][0] in ("first", "middle", "last")]
                    line += "   positions 1..10: " + " ".join(str(sum(p == k for p in pos)) for k in range(1, 11))
                print(line)
        print("  pre-registered contrasts (bootstrap over fact x seed):")
        for fmt in FMTS:
            cl = defaultdict(list)
            for i in R:
                it = items[i]
                if it["fmt"] == fmt and it["K"] in (2, 10):
                    cl[(it["fact"], it["seed"])].append((it["K"], R[i][0] == "last"))
            f = lambda xs: pct([x[1] for x in xs if x[0] == 10]) - pct([x[1] for x in xs if x[0] == 2])
            lo, hi = boot(cl, f); print(f"    P1 {fmt:15s} last% K=10 minus K=2: {f([x for v in cl.values() for x in v]):+6.1f} [{lo:+6.1f},{hi:+6.1f}]")
            cl2 = defaultdict(list)
            for i in R:
                it = items[i]
                if it["fmt"] == fmt and it["K"] == 10 and R[i][0] in ("first", "middle"):
                    cl2[(it["fact"], it["seed"])].append(R[i][1])
            g = lambda xs: pct([x == 1 for x in xs]) - 100 / 8 * pct([1 < x < 10 for x in xs]) / 100 if xs else float("nan")
            allx = [x for v in cl2.values() for x in v]
            if len(allx) >= 30 and len(cl2) >= 10:
                lo, hi = boot(cl2, g); print(f"    P2 {fmt:15s} among non-last K=10 answers (n={len(allx)}): first% minus mean middle-position%: {g(allx):+6.1f} [{lo:+6.1f},{hi:+6.1f}]; position 9 share {pct([x == 9 for x in allx]):.1f}%")
            else:
                print(f"    P2 {fmt:15s} untestable: {len(allx)} non-last answers in {len(cl2)} clusters (need 30 in 10); positions 1..9: " + " ".join(str(allx.count(k)) for k in range(1, 10)))


if __name__ == "__main__":
    main()
