"""three-forces/base scoring.  python3 score_base.py main
Answer text = generation cut at the first "\\nUser" (base models keep writing the dialogue). Then the same classifier as
../score_tf.py: first number not echoed from the final question -> said / shown / memory / memory_leak / other.
Paired base - instruct contrasts per family, bootstrap over (fact, seed) clusters."""
import json, os, re, sys, random
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1] if len(sys.argv) > 1 else "main"
items = {i["id"]: i for i in json.load(open(os.path.join(HERE, "sets_base.json")))["items"]}
WORDS = {w: k for k, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen".split())}
NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?|\b(?:" + "|".join(WORDS) + r")\b", re.I)
ARMS = ["A_all_differ", "B_said_is_memory", "C_shown_is_memory", "D_no_memory", "E_shown_no_instr"]; NS = [0, 1, 2, 4, 8, 16, 32, 50]
PAIRS = [("qwen14_base", "qwen14_inst"), ("olmo13_base", "olmo13_inst")]
KS = ("said", "shown", "memory", "memory_leak", "other")


def cut(s):
    return re.split(r"\n\s*User", s)[0]


def nums(s):
    s = re.sub(r"^\s*\d+[.)]\s+", "", s)
    return [float(WORDS[m.group().lower()]) if m.group().lower() in WORDS else float(m.group()) for m in NUM.finditer(s)]


def same(v, t):
    return t is not None and abs(round(v, 2) - round(t, 2)) < 1e-6


def cls(it, s):
    echo = set(nums(it["final"])); vs = [v for v in nums(cut(s)) if v not in echo]
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


def boot(cl, f, nb=2000, seed=0):
    rnd = random.Random(seed); keys = sorted(cl); res = sorted(f([x for k in (rnd.choice(keys) for _ in keys) for x in cl[k]]) for _ in range(nb))
    return res[int(.025 * nb)], res[int(.975 * nb) - 1]


def pct(xs):
    return 100 * sum(xs) / len(xs) if xs else float("nan")


def load(mk):
    p = os.path.join(HERE, f"out_{tag}_{mk}.json")
    if not os.path.exists(p):
        return None
    return {int(k): v for k, v in json.load(open(p)).items()}


def main():
    C = {}
    for pair in PAIRS + [("qwen14_chat_nopen",)]:
        for mk in pair:
            out = load(mk)
            if out is None:
                print(mk, "missing"); continue
            C[mk] = {i: cls(items[i], s) for i, s in out.items()}
            print(f"\n=== {mk} ({len(out)} of {len(items)}) ===")
            print("  baseline: " + "; ".join(f"{items[i]['fact']}{'(n)' if items[i]['nonce'] else ''}: {cut(out[i]).strip()[:24]!r}" for i in sorted(out) if items[i]["arm"] == "Z_baseline"))
            for arm in ARMS:
                print(f"  {arm}")
                for N in NS:
                    ids = [i for i in out if items[i]["arm"] == arm and items[i]["N"] == N]
                    if ids:
                        print(f"    N={N:2d} n={len(ids):3d}  " + "  ".join(f"{k} {pct([C[mk][i] == k for i in ids]):5.1f}" for k in KS))
    print("\n=== paired base - instruct (same prompt), bootstrap over (fact, seed); raw | normalised x/(said+shown+memory) | other% base/inst ===")
    TESTS = [("P3 said, arm A, N=0", "A_all_differ", "said", [0]), ("C1 shown, arm E, N=1,2", "E_shown_no_instr", "shown", [1, 2]),
             ("C1 shown, arm A, N=1,2", "A_all_differ", "shown", [1, 2]), ("P1 said, arm A, N>=1", "A_all_differ", "said", NS[1:]),
             ("said, arm B, N>=1", "B_said_is_memory", "said", NS[1:]), ("memory, arm A, N>=1", "A_all_differ", "memory", NS[1:]),
             ("memory, arm E, N>=1", "E_shown_no_instr", "memory", NS[1:]), ("shown, arm E, N>=1", "E_shown_no_instr", "shown", NS[1:])]
    VALID = ("said", "shown", "memory")
    for b, n in PAIRS:
        if b not in C or n not in C:
            continue
        common = set(C[b]) & set(C[n])
        print(f"  {b} - {n}  ({len(common)} common items)")
        for label, arm, k, Ns in TESTS:
            cl, cln, oth = defaultdict(list), defaultdict(list), []
            for i in common:
                it = items[i]
                if it["arm"] == arm and it["N"] in Ns:
                    key = (it["fact"], it["seed"])
                    cl[key].append((C[b][i] == k) - (C[n][i] == k)); oth.append((C[b][i] == "other", C[n][i] == "other"))
                    cln[key].append((C[b][i] == k, C[b][i] in VALID, C[n][i] == k, C[n][i] in VALID))
            if not cl:
                continue
            f = lambda xs: 100 * sum(xs) / len(xs); lo, hi = boot(cl, f)
            def g(xs):
                nb, db, ni, di = (sum(x[j] for x in xs) for j in range(4))
                return 100 * (nb / db if db else 0) - 100 * (ni / di if di else 0)
            nlo, nhi = boot(cln, g)
            print(f"    {label:24s} {f([x for xs in cl.values() for x in xs]):+6.1f} [{lo:+6.1f},{hi:+6.1f}] | "
                  f"{g([x for xs in cln.values() for x in xs]):+6.1f} [{nlo:+6.1f},{nhi:+6.1f}] | other {pct([o[0] for o in oth]):4.1f}/{pct([o[1] for o in oth]):4.1f}")
        print("    per N, arm A said / shown, base vs inst:  " + "  ".join(
            f"N{N}: {pct([C[b][i] == 'said' for i in common if items[i]['arm'] == 'A_all_differ' and items[i]['N'] == N]):.0f}/"
            f"{pct([C[b][i] == 'shown' for i in common if items[i]['arm'] == 'A_all_differ' and items[i]['N'] == N]):.0f} vs "
            f"{pct([C[n][i] == 'said' for i in common if items[i]['arm'] == 'A_all_differ' and items[i]['N'] == N]):.0f}/"
            f"{pct([C[n][i] == 'shown' for i in common if items[i]['arm'] == 'A_all_differ' and items[i]['N'] == N]):.0f}" for N in NS))
        print("    per fact, arm A N>=1 said% base/inst | shown% base/inst | memory% base/inst")
        for fact in sorted({items[i]["fact"] for i in common}):
            ids = [i for i in common if items[i]["fact"] == fact and items[i]["arm"] == "A_all_differ" and items[i]["N"] >= 1]
            print(f"      {fact:16s} " + " | ".join(f"{pct([C[b][i] == k for i in ids]):5.1f}/{pct([C[n][i] == k for i in ids]):5.1f}" for k in VALID))
    print("\n=== format check (R5): arm A said% at N=0,1,2  plain instruct vs chat instruct ===")
    ref = {"qwen14_inst": ("qwen14_chat_nopen", None), "olmo13_inst": (None, os.path.join(HERE, "..", f"out_main_olmo13.json"))}
    for mk, (chat_mk, path) in ref.items():
        if mk not in C:
            continue
        chat = load(chat_mk) if chat_mk else ({int(k): v for k, v in json.load(open(path)).items()} if os.path.exists(path) else None)
        if chat is None:
            print(" ", mk, "no chat reference"); continue
        Cc = {i: cls(items[i], s) for i, s in chat.items()}
        print(f"  {mk}: " + "  ".join(f"N{N} plain {pct([C[mk][i] == 'said' for i in C[mk] if items[i]['arm'] == 'A_all_differ' and items[i]['N'] == N]):.0f} "
                                      f"chat {pct([Cc[i] == 'said' for i in Cc if items[i]['arm'] == 'A_all_differ' and items[i]['N'] == N]):.0f}" for N in (0, 1, 2)))
    if "qwen14_chat_nopen" in C or load("qwen14_chat_nopen"):
        nop = load("qwen14_chat_nopen"); old = os.path.join(HERE, "..", "out_main_qwen14.json")
        if nop and os.path.exists(old):
            Co = {int(k): cls(items[int(k)], v) for k, v in json.load(open(old)).items()}; Cn = {i: cls(items[i], s) for i, s in nop.items()}
            print("\n=== qwen14 chat: main run (repetition penalty 1.05) vs penalty off; memory% / shown% ===")
            for arm in ("A_all_differ", "E_shown_no_instr"):
                for fact in sorted({items[i]["fact"] for i in Cn}):
                    ids = [i for i in Cn if items[i]["arm"] == arm and items[i]["fact"] == fact and items[i]["N"] >= 1 and i in Co]
                    if ids:
                        print(f"    {arm:18s} {fact:16s} memory {pct([Co[i] == 'memory' for i in ids]):5.1f} -> {pct([Cn[i] == 'memory' for i in ids]):5.1f}   shown {pct([Co[i] == 'shown' for i in ids]):5.1f} -> {pct([Cn[i] == 'shown' for i in ids]):5.1f}")


if __name__ == "__main__":
    main()
