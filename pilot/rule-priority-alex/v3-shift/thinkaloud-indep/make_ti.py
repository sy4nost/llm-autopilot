"""Build the independent-task item set for the think-aloud test (PLAN.md v2).  python3 -I make_ti.py SEED [--smoke]
Writes sets_ti.json (or sets_smoke.json). Data files are read as plain json; nothing from them is executed.
pop = "trap" (published item) or "control" (same item mechanically changed so the habit answer is the right one)."""
import json, os, random, re, sys, hashlib
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lin

V3 = os.path.join(HERE, "..")
INSTR = "Follow the facts stated in this task exactly, even if they contradict what you know."
THINK = "\n\nThink it through step by step. Then give your final answer on the last line as: Answer: <answer>"
SRC = {"memo": "thinkaloud-indep/data/memo-trap.jsonl", "neqa": "general-lamp/data/NeQA.jsonl",
       "hind": "indep/data/hindsight-neglect-10shot.jsonl", "unknown": "thinkaloud-indep/data/into-the-unknown.jsonl",
       "alg": "indep/data/repetitive-algebra.jsonl", "modus": "thinkaloud-indep/data/modus-tollens.jsonl",
       "quote": "general-lamp/data/quote-repetition.jsonl", "pattern": "indep/data/pattern-matching-suppression.jsonl",
       "redef": "indep/data/redefine-math.jsonl"}
MAIN = ["memo", "neqa", "hind", "unknown", "alg"]
REP = ["modus", "quote", "pattern", "redef"]
N_TRAP, N_CTRL, N_REP, N_MMLU = 120, 80, 100, 1000
MEMO = re.compile(r'^Write a quote that ends in the word "([^"]+)": (.+)$', re.S)
NEG = {"is not": "is", "isn't": "is", "aren't": "are", "wasn't": "was", "weren't": "were"}


def load(k):
    rows = [json.loads(l) for l in open(os.path.join(V3, SRC[k])) if l.strip()]
    for x in rows:
        x["prompt"] = x["prompt"].replace("\r\n", "\n").replace("﻿", "")
    seen, out = set(), []
    for x in rows:
        if x["prompt"] not in seen and len({c.strip() for c in x["classes"]}) == len(x["classes"]):
            seen.add(x["prompt"]); out.append(x)
    return out


def stem(k, p):
    """unfinished text the answer completes (completion-style tasks); '' for question tasks"""
    if k == "memo":
        return MEMO.match(p).group(2)
    if k == "quote":
        return p.rsplit("Output: ", 1)[1]
    if k == "pattern":
        return p.rstrip().split("\n")[-1]
    return ""


def cluster(k, p):
    if k == "memo":
        m = MEMO.match(p); return m.group(2) if m else p
    if k == "unknown":
        return p.split("\n", 1)[0].split(":", 1)[1].strip()  # the story (We know: / Context: variants share it)
    if k == "hind":
        return p.rsplit("Question:", 1)[1]
    if k == "quote":
        return p.rsplit("Output: ", 1)[1]
    return p


# ---- controls: habit answer made right
def ctrl_memo(x):
    w, s = MEMO.match(x["prompt"]).groups(); fam = x["classes"][1 - x["answer_index"]]
    fw = fam.strip().rstrip(".!?,;")
    return dict(x, prompt=f'Write a quote that ends in the word "{fw}": {s}', answer_index=1 - x["answer_index"])


def ctrl_neqa(x):
    q = x["prompt"]; hits = [n for n in NEG if re.search(r"\b" + re.escape(n) + r"\b", q)]
    if len(hits) != 1 or len(re.findall(r"\b" + re.escape(hits[0]) + r"\b", q)) != 1:
        if not (len(re.findall(r"\bnot\b", q)) == 1 and not hits):
            return None
        return dict(x, prompt=re.sub(r"\bnot ", "", q, count=1), answer_index=1 - x["answer_index"])
    return dict(x, prompt=re.sub(r"\b" + re.escape(hits[0]) + r"\b", NEG[hits[0]], q, count=1), answer_index=1 - x["answer_index"])


def ctrl_hind(x):
    blocks = x["prompt"].split("Question:")[1:]  # 10 examples + test
    ex = blocks[-2]; q, a = ex.rsplit("\nAnswer:", 1)
    ctx = "".join("Question:" + b for b in blocks[:-2])
    return dict(x, prompt=ctx + "Question:" + q + "\nAnswer:", answer_index=x["classes"].index(" " + a.strip()))


def ctrl_alg(x):
    lines = x["prompt"].rstrip().split("\n"); prev = lines[-2].rsplit("A:", 1)[1].strip()
    nl = lin.retarget(lines[-1], int(prev))
    return dict(x, prompt="\n".join(lines[:-1] + [nl]), answer_index=[c.strip() for c in x["classes"]].index(prev))


def safe(f, x):
    try:
        return f(x)
    except Exception:
        return None


CTRL = {"memo": ctrl_memo, "neqa": ctrl_neqa, "hind": ctrl_hind, "alg": ctrl_alg}


def mmlu(rng, n, old=()):
    rows = [json.loads(l) for l in open(os.path.join(HERE, "data", "mmlu.jsonl")) if l.strip()]
    seen, uniq = set(old), []
    for y in rows:
        if y["question"].strip() not in seen:
            seen.add(y["question"].strip()); uniq.append(y)
    out = []
    for x in rng.sample(uniq, n):
        q = x["question"].strip() + "\n" + "\n".join(f"{c}. {t}" for c, t in zip("ABCD", x["choices"])) + "\nAnswer:"
        out.append(dict(task="mmlu", pop="trap", prompt=q, classes=[" A", " B", " C", " D"], answer_index=x["answer"]))
    return out


def build(S, smoke=False):
    rng = random.Random(S); items = []
    old = set()  # main set never reuses a smoke item (by cluster)
    if not smoke:
        for l in open(os.path.join(HERE, "out_smoke.jsonl")):  # items the smoke run actually used
            x = json.loads(l)
            if "task" in x:
                c = x["cluster"].rsplit("Output: ", 1)[1] if x["task"] == "quote" and "Output: " in x["cluster"] else x["cluster"]
                old.add((x["task"], c))
    for k in MAIN:
        rows = [x for x in load(k) if (k, cluster(k, x["prompt"])) not in old]
        if k in CTRL:  # also never reuse the source of a smoke control
            rows = [x for x in rows if (lambda c: c is None or (k, cluster(k, c["prompt"])) not in old)(safe(CTRL[k], x))]
        if k == "unknown":  # only 114 stories: at most one item per story and answer side; 50 + 50 traps
            rng.shuffle(rows); seen = set()
            rows = [x for x in rows if not ((cluster(k, x["prompt"]), x["answer_index"]) in seen or seen.add((cluster(k, x["prompt"]), x["answer_index"])))]
        if k == "memo":
            rows = [x for x in rows if MEMO.match(x["prompt"]) and
                    MEMO.match(x["prompt"]).group(1) == x["classes"][x["answer_index"]].strip().rstrip(".!?,;")]
        nt = 6 if smoke else (100 if k == "unknown" else N_TRAP)
        pick = [x for c in (0, 1) for x in rng.sample([y for y in rows if y["answer_index"] == c], nt // 2)]
        items += [dict(x, task=k, pop="trap") for x in pick]
        if k in CTRL:  # controls from items NOT used as traps (different clusters)
            used = {cluster(k, x["prompt"]) for x in pick}
            pool = [x for x in rows if cluster(k, x["prompt"]) not in used]
            rng.shuffle(pool); got = []; nc = (4 if smoke else N_CTRL) // 2
            for x in pool:  # balanced: nc controls whose right answer is option 0, nc option 1
                c = CTRL[k](x)
                if c is not None and cluster(k, c["prompt"]) not in {cluster(k, g["prompt"]) for g in got} and \
                        sum(g["answer_index"] == c["answer_index"] for g in got) < nc:
                    got.append(dict(c, task=k, pop="control"))
                if len(got) == 2 * nc:
                    break
            items += got
    for k in REP:
        items += [dict(x, task=k, pop="trap") for x in rng.sample([y for y in load(k) if (k, cluster(k, y["prompt"])) not in old], 4 if smoke else N_REP)]
    items += mmlu(rng, 10 if smoke else N_MMLU, {c.split("\nA. ")[0] for t, c in old if t == "mmlu"})
    out = []
    for i, x in enumerate(items):
        P = x["prompt"]; core = re.sub(r"(^|\s)(Answer|A):\s*$", "", P).rstrip()
        cands = [c.strip() for c in x["classes"]]
        assert len(set(cands)) == len(cands), x
        st = stem(x["task"], P)
        out.append(dict(id=i, task=x["task"], pop=x["pop"], main=x["task"] in MAIN, key=x["answer_index"], cands=cands,
                        stem=st, cluster=cluster(x["task"], P) if x["task"] != "mmlu" else P, raw=P, raw_classes=x["classes"],
                        direct=P, instr=INSTR + "\n\n" + P, think=core + THINK, thinki=INSTR + "\n\n" + core + THINK))
    return out


if __name__ == "__main__":
    S = int(sys.argv[1]); smoke = "--smoke" in sys.argv
    items = build(S, smoke)
    fn = os.path.join(HERE, "sets_smoke2.json" if smoke else "sets_ti.json")  # sets_smoke.json (run as out_smoke.jsonl) was overwritten once; its items are in out_smoke.jsonl
    json.dump(dict(seed=S, instr=INSTR, think=THINK, items=items), open(fn, "w"), ensure_ascii=False)
    print(fn, len(items), hashlib.sha256(open(fn, "rb").read()).hexdigest())
    print(Counter((x["task"], x["pop"]) for x in items))
    print("key balance:", {k: Counter(x["key"] for x in items if x["task"] == k and x["pop"] == "trap") for k in MAIN})
