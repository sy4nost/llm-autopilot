"""three-forces scoring (v2 after review).  python3 score_tf.py main
Answer = first number in the answer that is not echoed from the final question (digits or number word, list markers skipped),
compared after rounding to 2 decimals. Class: said / shown / memory / other; in the nonce arm D the real M counts as memory_leak.
Bootstrap over (fact, seed) clusters."""
import json, os, re, sys, random
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(sys.argv[0] if sys.argv[0].endswith(".py") else "."))
tag = sys.argv[1] if len(sys.argv) > 1 else "main"
items = {i["id"]: i for i in json.load(open(os.path.join(HERE, "sets_tf.json")))["items"]}
WORDS = {w: k for k, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen".split())}
ARMS = ["A_all_differ", "B_said_is_memory", "C_shown_is_memory", "D_no_memory", "E_shown_no_instr"]; NS = [0, 1, 2, 4, 8, 16, 32, 50]
NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?|\b(?:" + "|".join(WORDS) + r")\b", re.I)


def nums(s):
    s = re.sub(r"^\s*\d+[.)]\s+", "", s)  # list marker "1. 8"
    return [float(WORDS[m.group().lower()]) if m.group().lower() in WORDS else float(m.group()) for m in NUM.finditer(s)]


def same(v, t):
    return t is not None and abs(round(v, 2) - round(t, 2)) < 1e-6  # pi: 3.14159 counts as 3.14


def cls(it, s):
    echo = set(nums(it["final"]))
    vs = [v for v in nums(s) if v not in echo]
    if not vs:
        return "other"
    v = vs[0]
    if same(v, it["said"]):
        return "said"
    if same(v, it["shown"]):
        return "shown"
    if same(v, it["M"]):
        return "memory_leak" if it["nonce"] else "memory"
    return "other"


def boot(clusters, f, nb=2000, seed=0):
    rnd = random.Random(seed); keys = sorted(clusters); res = []
    for _ in range(nb):
        res.append(f([x for k in (rnd.choice(keys) for _ in keys) for x in clusters[k]]))
    res.sort(); return res[int(.025 * nb)], res[int(.975 * nb) - 1]


def pct(xs):
    return 100 * sum(xs) / len(xs) if xs else float("nan")


def main():
    for mk in ["qwen14", "phi4", "olmo13"]:
        p = os.path.join(HERE, f"out_{tag}_{mk}.json")
        if not os.path.exists(p):
            print(mk, "missing"); continue
        out = {int(k): v for k, v in json.load(open(p)).items()}
        C = {i: cls(items[i], s) for i, s in out.items()}
        anyM = {i: any(same(v, items[i]["M"]) for v in nums(s)) for i, s in out.items()}
        print(f"\n=== {mk} ({len(out)} of {len(items)} answers) ===")
        print("  baseline (final question alone): " + "; ".join(
            f"{items[i]['fact']}{'(nonce)' if items[i]['nonce'] else ''}: {out[i].strip()[:30]!r}" for i in sorted(out) if items[i]["arm"] == "Z_baseline"))
        print("  shares %, all facts pooled: said / shown / memory / memory_leak / other")
        for arm in ARMS:
            print(f"  {arm}")
            for N in NS:
                ids = [i for i in out if items[i]["arm"] == arm and items[i]["N"] == N]
                if ids:
                    print(f"    N={N:2d} n={len(ids):3d}  " + "  ".join(f"{k} {pct([C[i] == k for i in ids]):5.1f}" for k in ("said", "shown", "memory", "memory_leak", "other"))
                          + f"   M anywhere {pct([anyM[i] for i in ids]):5.1f}")
        by = defaultdict(dict)
        for i in out:
            it = items[i]
            if it["N"] >= 1 and it["arm"] != "Z_baseline":
                by[(it["fact"], it["seed"], it["N"])][it["arm"]] = i
        full = {k: v for k, v in by.items() if all(a in v for a in ARMS)}
        if not full:
            print("  (no complete paired sets yet)"); continue

        def contrast(a, b, ka, kb):
            cl = defaultdict(list)
            for k, v in full.items():
                cl[(k[0], k[1])].append((C[v[a]] == ka) - (C[v[b]] == kb))
            f = lambda xs: 100 * sum(xs) / len(xs)
            lo, hi = boot(cl, f); return f([x for xs in cl.values() for x in xs]), lo, hi
        memA = pct([C[v["A_all_differ"]] == "memory" for v in full.values()])
        print("  summaries over N=1..50, paired, bootstrap over (fact, seed) clusters:")
        print("    Q2 said%% B - A              %+.1f [%+.1f,%+.1f]   (null = memory share in A, %.1f%%: in B said = memory)" % (contrast("B_said_is_memory", "A_all_differ", "said", "said") + (memA,)))
        print("    Q3 shown%% C - A             %+.1f [%+.1f,%+.1f]   (in C shown = memory)" % contrast("C_shown_is_memory", "A_all_differ", "shown", "shown"))
        print("    Q4 said%% A - D (real-nonce) %+.1f [%+.1f,%+.1f]" % contrast("A_all_differ", "D_no_memory", "said", "said"))
        print("    Q4b memory share in A %.1f%%; memory_leak share in D %.1f%%" % (memA, pct([C[v["D_no_memory"]] == "memory_leak" for v in full.values()])))
        print("    E (no instruction): shown %.1f%%, memory %.1f%%" % (pct([C[v["E_shown_no_instr"]] == "shown" for v in full.values()]), pct([C[v["E_shown_no_instr"]] == "memory" for v in full.values()])))
        print("  per fact (N>=1): A said/shown/memory | B-A said | C-A shown | A-D said | E shown/memory | #other in A, example")
        for f in sorted({k[0] for k in full}):
            vs = [v for k, v in full.items() if k[0] == f]
            a = lambda arm, k: pct([C[v[arm]] == k for v in vs])
            oth = [out[v["A_all_differ"]] for v in vs if C[v["A_all_differ"]] == "other"]
            print(f"    {f:16s} {a('A_all_differ', 'said'):5.1f}/{a('A_all_differ', 'shown'):5.1f}/{a('A_all_differ', 'memory'):5.1f} | "
                  f"{a('B_said_is_memory', 'said') - a('A_all_differ', 'said'):+6.1f} | {a('C_shown_is_memory', 'shown') - a('A_all_differ', 'shown'):+6.1f} | "
                  f"{a('A_all_differ', 'said') - a('D_no_memory', 'said'):+6.1f} | {a('E_shown_no_instr', 'shown'):5.1f}/{a('E_shown_no_instr', 'memory'):5.1f} | "
                  f"{len(oth)} {(oth[0].strip()[:40] if oth else '')!r}")
        print("  Q5 formula vs words (arm A): compare rows pi_formula vs pi_words, hexagon_formula vs hexagon")


if __name__ == "__main__":
    main()
