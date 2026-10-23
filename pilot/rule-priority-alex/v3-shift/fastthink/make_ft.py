"""Build the item set for the fast-vs-thought test (PLAN.md).  python3 -I make_ft.py SEED [--smoke]
Writes sets_ft.json (or sets_smoke.json). Data files are read as plain json; nothing from them is executed."""
import json, os, random, re, sys, hashlib
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
TI = os.path.join(HERE, "..", "thinkaloud-indep")
sys.path.insert(0, TI)
import make_ti as ti  # memo/hindsight loaders and clusters, identical to the earlier run

THINK = ti.THINK
NEW = ["sigfigs", "tqa", "hans", "epi"]
REPL = ["memo", "hind"]
N = {"sigfigs": 200, "tqa": 250, "hans": 300, "epi": 200, "memo": 150, "hind": 115, "mmlu": 400}
SMOKE_N = 6
WORD_OPTS = ["entailment", "non-entailment"]


def rows(fn):
    return [json.loads(l) for l in open(fn) if l.strip()]


def strat(rng, xs, n, keyf):
    """n items, as equal as possible per key value (key = right option index)"""
    by = {}
    for x in xs:
        by.setdefault(keyf(x), []).append(x)
    ks = sorted(by); out = []
    for j, k in enumerate(ks):
        want = n // len(ks) + (1 if j < n % len(ks) else 0)
        out += rng.sample(by[k], min(want, len(by[k])))
    return out


def used_ti():
    """clusters/questions used by thinkaloud-indep (main sets and its smoke)"""
    old = set()
    for x in json.load(open(os.path.join(TI, "sets_ti.json")))["items"]:
        old.add((x["task"], x["cluster"]))
    for l in open(os.path.join(TI, "out_smoke.jsonl")):
        x = json.loads(l)
        if "task" in x:
            old.add((x["task"], x["cluster"]))
    return old


def item(task, pop, prompt, cands, key, cluster, stem=""):
    core = re.sub(r"(^|\s)(Answer|A|Relation):\s*$", "", prompt).rstrip()
    return dict(task=task, pop=pop, prompt=prompt, core=core, cands=cands, key=key, cluster=cluster, stem=stem)


def build(S, smoke):
    rng = random.Random(S); n = (lambda k: SMOKE_N) if smoke else (lambda k: N[k]); out = []
    old = used_ti()
    if not smoke:  # never reuse a fastthink smoke item (by test and cluster)
        sm = {(x["task"], x["cluster"]) for x in json.load(open(os.path.join(HERE, "sets_smoke.json")))["items"]}
    else:
        sm = set()
    keep = lambda t, c: (t, c) not in sm
    # sig-figs: published prompt ends with options and "Answer:"; classes " A", " B", " C"
    sf = {}
    for x in rows(os.path.join(TI, "data", "sig-figs.jsonl")):
        p = x["prompt"].replace("\r\n", "\n")
        sf.setdefault(p, x)
    sf = [x for x in sf.values() if len(x["classes"]) == 3]  # 3-option items only (PLAN v2, B1): no answer-position confound
    sf = [x for x in sf if keep("sigfigs", x["prompt"].replace("\r\n", "\n"))]
    for x in strat(rng, sf, n("sigfigs"), lambda x: x["answer_index"]):
        p = x["prompt"].replace("\r\n", "\n")
        out.append(item("sigfigs", "trap", p, [c.strip() for c in x["classes"]], x["answer_index"], p))
    # TruthfulQA MC1: first choice is the right one in the source -> shuffle with the seed
    tq = [x for x in rows(os.path.join(HERE, "data", "truthfulqa.jsonl")) if keep("tqa", x["question"].strip())]
    for x in rng.sample(tq, n("tqa")):
        ch = x["mc1_targets"]["choices"]; lab = x["mc1_targets"]["labels"]; assert lab.count(1) == 1
        order = list(range(len(ch))); rng.shuffle(order)
        L = "ABCDEFGHIJKLM"[:len(ch)]
        p = "Question: " + x["question"].strip() + "\n" + "\n".join(f"{L[j]}. {ch[o]}" for j, o in enumerate(order)) + "\nAnswer:"
        out.append(item("tqa", "trap", p, list(L), [lab[o] for o in order].index(1), x["question"].strip()))
    # HANS: label 0 = entailment (control, overlap heuristic right), 1 = non-entailment (trap)
    hs = rows(os.path.join(HERE, "data", "hans.jsonl")); seen = set(); hu = []
    for x in hs:
        k = (x["sentence1"], x["sentence2"])
        if k not in seen and keep("hans", x["sentence1"].replace(" .", ".").replace(" ,", ",")):
            seen.add(k); hu.append(x)
    for lab, pop in ((1, "trap"), (0, "control")):
        for x in rng.sample([y for y in hu if y["labels"] == lab], n("hans") // 2):
            s1 = x["sentence1"].replace(" .", ".").replace(" ,", ","); s2 = x["sentence2"].replace(" .", ".").replace(" ,", ",")
            p = f"Premise: {s1}\nHypothesis: {s2}\nDoes the premise entail the hypothesis? Answer entailment or non-entailment."
            out.append(item("hans", pop, p, WORD_OPTS, lab, s1))
    # BIG-bench epistemic_reasoning
    ep = [x for x in rows(os.path.join(HERE, "data", "epistemic.jsonl")) if keep("epi", x["inputs"].strip())]
    for x in strat(rng, ep, n("epi"), lambda x: x["targets"][0]):
        assert x["multiple_choice_targets"] == WORD_OPTS
        out.append(item("epi", "trap", x["inputs"].strip(), WORD_OPTS, WORD_OPTS.index(x["targets"][0]), x["inputs"].strip()))
    # replication: memo-trap and hindsight items unused by thinkaloud-indep (by cluster)
    for k in REPL:
        rs = [x for x in ti.load(k) if (k, ti.cluster(k, x["prompt"])) not in old and keep(k, ti.cluster(k, x["prompt"]))]
        if k == "memo":
            rs = [x for x in rs if ti.MEMO.match(x["prompt"]) and
                  ti.MEMO.match(x["prompt"]).group(1) == x["classes"][x["answer_index"]].strip().rstrip(".!?,;")]
        for x in strat(rng, rs, min(n(k), len(rs)), lambda x: x["answer_index"]):
            P = x["prompt"]
            out.append(item(k, "trap", P, [c.strip() for c in x["classes"]], x["answer_index"], ti.cluster(k, P), ti.stem(k, P)))
    # MMLU questions not used by thinkaloud-indep
    usedq = {c.split("\nA. ")[0] for t, c in old | {(t, c) for t, c in sm} if t == "mmlu"}
    for x in ti.mmlu(rng, n("mmlu"), usedq):
        out.append(item("mmlu", "trap", x["prompt"], [c.strip() for c in x["classes"]], x["answer_index"], x["prompt"]))
    for i, x in enumerate(out):
        assert len(set(x["cands"])) == len(x["cands"]) and 0 <= x["key"] < len(x["cands"])
        x["id"] = i; x["think"] = x["core"] + THINK
    return out


if __name__ == "__main__":
    S = int(sys.argv[1]); smoke = "--smoke" in sys.argv
    items = build(S, smoke)
    fn = os.path.join(HERE, "sets_smoke.json" if smoke else "sets_ft.json")
    json.dump(dict(seed=S, think=THINK, items=items), open(fn, "w"), ensure_ascii=False)
    print(fn, len(items), hashlib.sha256(open(fn, "rb").read()).hexdigest())
    print(Counter((x["task"], x["pop"]) for x in items))
    print("key balance:", {k: dict(Counter(x["key"] for x in items if x["task"] == k)) for k in NEW + REPL})
