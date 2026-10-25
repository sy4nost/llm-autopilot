"""WildChat items for conflict-train from the hand labels in ext/wc_labels.txt (labels by Claude; S = unclear, dropped).
Split by conversation, fixed before any run: candidates 0-79 + plain 0-39 -> test; candidates 80-124 + plain 40-59 -> train,
every 5th of those -> dev. Training targets are templated; only yes/no is scored.
  python3 make_wc.py 20261025 -> wc_items.jsonl"""
import json, os, sys, random, hashlib
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
T = dict(
    Yr=["Conflict: yes. Between: the earlier request and the user's new requirement. Why: the latest turn asks for something different from what was set before.",
        "Conflict: yes. Between: the earlier answer and the user's changed requirement. Why: the user now wants a different version than the one already given."],
    Yc=["Conflict: yes. Between: the assistant's earlier answer and the user's correction. Why: the user says the earlier answer is wrong or not what was asked.",
        "Conflict: yes. Between: the earlier reply and the user's objection. Why: the user's latest turn rejects what the assistant gave before."],
    N=["Conflict: no. Between: the earlier turns and the latest request. Why: the latest request is new or fits what came before.",
       "Conflict: no. Between: the conversation so far and the final request. Why: nothing earlier calls for a different answer."])


def build(S):
    r = random.Random(S); lab = {}
    for l in open(os.path.join(HERE, "ext", "wc_labels.txt")):
        if l[:4] in ("cand", "plai"):
            k, i, v = l.split(); lab[(k, int(i))] = v
    src = dict(cand=[json.loads(l) for l in open(os.path.join(HERE, "ext", "wildchat_candidates.jsonl"))],
               plain=[json.loads(l) for l in open(os.path.join(HERE, "ext", "wildchat_plain.jsonl"))])
    out, n_tr = [], 0
    for (k, i), v in sorted(lab.items()):
        if v == "S":
            continue
        x = src[k][i]; test = i < (80 if k == "cand" else 40)
        if not test:
            n_tr += 1
        split = "test" if test else ("dev" if n_tr % 5 == 0 else "train")
        it = dict(id=f"wc_{k}{i}", set="X_WC", family="WC", label=v != "N", kind=v, split=split, messages=x["messages"], meta=dict(hash=x["hash"]))
        if split != "test":
            it["target"] = r.choice(T[v])
        out.append(it)
    first = lambda x: next(m["content"] for m in x["messages"] if m["role"] == "user")
    trained = {first(x) for x in out if x["split"] != "test"}
    return [x for x in out if x["split"] != "test" or first(x) not in trained]   # same opening as a training chat -> drop from test


if __name__ == "__main__":
    items = build(int(sys.argv[1])); fn = os.path.join(HERE, "wc_items.jsonl")
    with open(fn, "w") as fh:
        for x in items:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(fn, len(items), hashlib.sha256(open(fn, "rb").read()).hexdigest())
    print(dict(sorted(Counter((x["split"], x["label"]) for x in items).items())))
