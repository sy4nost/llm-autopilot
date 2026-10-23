"""Reviewer check: is the native family a structural twin of the fictional (make_match) family for the same seed?
Each family is built in its own fresh interpreter (make_native patches make_match's module globals in place).
python3 -I check_twin.py <v3-shift dir> seed [seed ...]"""
import json, os, re, subprocess, sys

V3 = os.path.abspath(sys.argv[1])
CHILD = r'''
import json, os, sys
V3 = sys.argv[1]; fam = sys.argv[2]; seed = int(sys.argv[3])
sys.path.insert(0, V3); sys.path.insert(0, os.path.join(V3, "match"))
if fam == "native":
    sys.path.insert(0, os.path.join(V3, "native"))
    import make_native                       # patches make_match pools
    mm = make_native.mm
else:
    import make_match as mm
keep = []
items = mm.build(seed, keep=keep)
pools = dict(F1=mm.F1_NAMES, F2=mm.F2_NAMES, F3=mm.F3_NAMES, F4=mm.F4_NAMES)
sets = [dict(s=s, hists={"|".join(map(str, k)): v for k, v in h.items()}) for s, h in keep]
print(json.dumps(dict(items=items, sets=sets, pools=pools, labels=mm.LABELS,
                      vals=[mm.F1_VALS, mm.F2_VALS, mm.F3_VALS, mm.F4_VALS])))
'''

def run(fam, seed):
    out = subprocess.run([sys.executable, "-I", "-c", CHILD, V3, fam, str(seed)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)

def skeleton(prompt, s):
    """replace every set-specific word (names, values, labels) by a positional placeholder"""
    words = {}
    for k, w in enumerate(s["n"]): words.setdefault(w, f"<N{k}>")
    for k, pair in enumerate(s["v"]):
        for j, w in enumerate(pair): words.setdefault(w, f"<V{k}{j}>")
    for k, w in enumerate(s["lab"]): words.setdefault(w, f"<L{k}>")
    for w in sorted(words, key=len, reverse=True):
        prompt = re.sub(r"(?<![A-Za-z])" + re.escape(w) + r"(?![A-Za-z])", words[w], prompt)
    return prompt

for seed in map(int, sys.argv[2:]):
    F, N = run("fict", seed), run("native", seed)
    assert len(F["items"]) == len(N["items"]), "item count"
    bad = dict(arm=0, set=0, probe=0, keypos=0, oldpos=0, skeleton=0, hist=0, order=0, labidx=0, collide_set=0)
    native_words = {"pedestrians", "crosswalk", "barrier", "STOP"}
    leak = 0
    for a, b in zip(F["items"], N["items"]):
        bad["arm"] += a["arm"] != b["arm"]; bad["set"] += a["set"] != b["set"]
        pa, pb = a["probes"][0], b["probes"][0]
        bad["probe"] += pa["id"] != pb["id"]
        bad["keypos"] += a["labels"].index(pa["key"]) != b["labels"].index(pb["key"])
        bad["oldpos"] += a["labels"].index(pa["old"]) != b["labels"].index(pb["old"])
        sa, sb = F["sets"][a["set"]]["s"], N["sets"][b["set"]]["s"]
        bad["skeleton"] += skeleton(a["prompt"], sa) != skeleton(b["prompt"], sb)
        leak += any(w in a["prompt"] for w in ("pedestrians", "crosswalk", "barrier"))
    for x, y in zip(F["sets"], N["sets"]):
        bad["order"] += x["s"]["order"] != y["s"]["order"]
        bad["labidx"] += F["labels"].index(x["s"]["lab"]) != N["labels"].index(y["s"]["lab"])
        bad["hist"] += x["hists"] != y["hists"]
        # word collisions inside one native set (two fields share a word, or a value equals a label)
        ws = y["s"]["n"] + [w for p in y["s"]["v"] for w in p] + list(y["s"]["lab"])
        bad["collide_set"] += len(ws) != len(set(ws))
    # does the fictional twin itself carry road-like words?
    roadlike = {"STOP", "GO", "CHECK", "HALT", "PROCEED", "gate", "signal", "pass", "ticket", "GREEN", "ACTIVE",
                "PRESENT", "ABSENT", "OPEN", "CLOSED", "LOCKED"}
    fict_road = sum(any(w in roadlike for w in x["s"]["n"] + [v for p in x["s"]["v"] for v in p] + list(x["s"]["lab"]))
                    for x in F["sets"])
    fict_stopgo = sum(x["s"]["lab"][0] in ("STOP", "HALT") for x in F["sets"])
    c0 = [(a, b) for a, b in zip(F["items"], N["items"]) if a["arm"] == "change_0"]
    print(f"seed {seed}: items {len(F['items'])} (change_0 {len(c0)}) mismatches {bad} | fictional prompts with road nouns {leak}"
          f" | fictional sets with any road-like word {fict_road}/{len(F['sets'])}, with STOP/HALT label {fict_stopgo}")
    if seed == int(sys.argv[2]):
        a, b = c0[0]
        print("--- example change_0 fictional:\n" + a["prompt"] + "\n--- native:\n" + b["prompt"])
