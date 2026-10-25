"""Outside test items for conflict-train (PLAN.md).  python3 -I make_ext.py SEED  -> ext_items.jsonl
Sets (label = is there a conflict):
  X_CICM  Berkeley CICM stale binding, 120 conflicts (24 per overwrite count) + 120 matched negatives (stale values replaced by the
          current one). CC BY-NC: never committed.
  X_MC    MultiChallenge (Scale AI), all 273 items, all labelled yes, report-only. Licence unstated: never committed.
  X_HIND  hindsight-neglect in the published 10-shot template: 100 traps (outcome disagrees with expected value) + 100 controls.
  X_ODD   Alex's text, sections 1-6, with the planted false sentence (yes) or the clean control (no): 12 items.
  X_MMLU  200 MMLU questions: no conflict (false alarms).
  X_WIKI  WikiContradict, deduped: both passages (yes) or one passage + an off-topic passage (no; easy, so report-only).
WildChat items are built by make_wc.py."""
import json, os, random, re, sys, hashlib, importlib.util
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__)); V3 = os.path.join(HERE, "..")
ODD = os.path.join(HERE, "..", "..", "..", "odd-phrase", "alex-text")


def mod(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(V3, path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def trim(t, n=1500):
    return t if len(t) <= n else t[:n] + " [...]"


def sections(fn):
    t = open(fn).read(); parts = re.split(r"\n(?=## \d)", t)
    return {p.split(".")[0].replace("## ", "").strip(): p.strip() for p in parts if p.startswith("## ")}


def build(S):
    r = random.Random(S); out = []
    cc = [json.loads(l) for l in open(os.path.join(V3, "cicm-lamp", "data", "cicm_natural_factorial_otherdist_l0.jsonl"))]
    for k in sorted({x["k_overwrites"] for x in cc}):
        for x in r.sample([y for y in cc if y["k_overwrites"] == k], 24):
            # drop the vague filler "I used to have a different approach, but I've changed recently." (+ its reply) from both
            # versions: it names no slot, so in the matched negative it would hint at a change that is not there
            ms, keep = x["messages"], []
            for i, m in enumerate(ms):
                if (m["role"] == "user" and "I used to have a different approach" in m["content"]) or \
                   (m["role"] == "assistant" and i > 0 and "I used to have a different approach" in ms[i - 1]["content"]):
                    continue
                keep.append(m)
            x = dict(x, messages=keep)
            out.append(dict(set="X_CICM", label=True, messages=x["messages"], meta=dict(id=x["id"], k=k, current=x["current_value"], stale=x["stale_values"])))
            # matched negative: the same conversation with every stale value replaced by the current one, so all mentions agree
            pat = re.compile(r"\b(" + "|".join(re.escape(v) for v in x["stale_values"]) + r")\b")
            neg = [dict(role=m["role"], content=pat.sub(x["current_value"], m["content"])) for m in x["messages"]]
            out.append(dict(set="X_CICM", label=False, messages=neg, meta=dict(id=x["id"], k=k, current=x["current_value"], stale=[])))
    for x in (json.loads(l) for l in open(os.path.join(HERE, "ext", "multichallenge.jsonl"))):
        msgs = [dict(role=m["role"], content=trim(m["content"]) if m["role"] == "assistant" else m["content"]) for m in x["CONVERSATION"]]
        out.append(dict(set="X_MC", label=True, messages=msgs, meta=dict(id=x["QUESTION_ID"], axis=x["AXIS"])))
    ct = mod("make_ct", "confident-traps/make_ct.py"); ct.N_TRAP, ct.N_CTRL = 100, 100
    for x in ct.build_hind(r):
        out.append(dict(set="X_HIND", label=x["pop"] == "trap", messages=[dict(role="user", content=x["prompt"])], meta=dict(pop=x["pop"])))
    a, c = sections(os.path.join(ODD, "text_en.md")), sections(os.path.join(ODD, "text_en_control.md"))
    ask = "Please summarize this section of my paper in two sentences."
    for n in [str(i) for i in range(1, 7)]:
        out.append(dict(set="X_ODD", label=True, messages=[dict(role="user", content=f"{a[n]}\n\n{ask}")], meta=dict(section=n)))
        out.append(dict(set="X_ODD", label=False, messages=[dict(role="user", content=f"{c[n]}\n\n{ask}")], meta=dict(section=n)))
    mm = [json.loads(l) for l in open(os.path.join(V3, "thinkaloud-indep", "data", "mmlu.jsonl"))]
    for x in r.sample(mm, 200):
        p = x["question"] + "\n" + "\n".join(f"{'ABCD'[i]}. {ch}" for i, ch in enumerate(x["choices"])) + "\nAnswer with the letter."
        out.append(dict(set="X_MMLU", label=False, messages=[dict(role="user", content=p)], meta={}))
    wk, seen = [], set()
    for y in (json.loads(l) for l in open(os.path.join(HERE, "ext", "wikicontradict.jsonl"))):
        if "context1" in y and y["question_ID"] not in seen and y["context1"].strip() != y["context2"].strip():
            seen.add(y["question_ID"]); wk.append(y)
    wk = r.sample(wk, len(wk))
    for j, x in enumerate(wk):
        if j % 2 == 0:      # both passages: a real contradiction
            c1, c2, lab = x["context1"], x["context2"], True
        else:               # matched negative: one passage plus an unrelated passage from another article
            other = wk[(j + 1) % len(wk)]
            c1, c2, lab = x["context1"], other["context2"], False
        p = f"Context:\n{c1}\n{c2}\n\nUsing the context, answer: {x['question']}"
        out.append(dict(set="X_WIKI", label=lab, messages=[dict(role="user", content=p)], meta=dict(kind=x["contradictType"], qid=x["question_ID"])))
    for i, x in enumerate(out):
        x["id"] = f"ext{i}"
    return out


if __name__ == "__main__":
    S = int(sys.argv[1]); items = build(S); fn = os.path.join(HERE, "ext_items.jsonl")
    with open(fn, "w") as fh:
        for x in items:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(fn, len(items), hashlib.sha256(open(fn, "rb").read()).hexdigest())
    print(dict(sorted(Counter((x["set"], x["label"]) for x in items).items())))
