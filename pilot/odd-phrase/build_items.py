"""Build odd-phrase items from generated texts (no GPU).

  python build_items.py split   # texts.json -> splits.json (sentences, hosts, donors, word-swap targets)
  python build_items.py build   # splits.json + swaps.json -> items.json

Hosts = rep 0 paragraph of each topic (60). Donor sentences come only from rep 1 paragraphs, so an inserted
sentence never appears elsewhere in a host. Each test text has 3 odd parts at spread positions (never first):
  far   = a sentence from a rep-1 paragraph of a different domain          (gross)
  near  = a sentence from a rep-1 paragraph of the same domain, other topic (medium)
  word  = a host sentence with one noun replaced by a noun from another domain (subtle; written by the author model)
Controls = the same 60 host paragraphs with nothing inserted.
"""

import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 7
BAD_START = re.compile(r"^(It|Its|This|That|These|Those|They|Their|He|She|His|Her|However|Additionally|Moreover|"
                       r"Also|Furthermore|In addition|As a result|Therefore|Thus|Finally|Overall|Lastly|Instead|"
                       r"Similarly|Conversely|Meanwhile|Consequently|Such)\b")


def split_sentences(text):
    text = re.sub(r"\s+", " ", text.replace("*", "")).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if len(p.strip().split()) >= 5]


def rd(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


def wr(name, obj):
    with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def usable(s):
    return not BAD_START.match(s) and 8 <= len(s.split()) <= 40


def split():
    rng = random.Random(SEED)
    texts = rd("texts.json")
    paras = {}
    for t in texts:
        sents = split_sentences(t["text"])
        if len(sents) >= 7:
            paras[(t["domain"], t["sub"], t["rep"])] = sents[:8]
    topics = sorted({(d, s) for d, s, r in paras})
    hosts = [(d, s) for d, s in topics if (d, s, 0) in paras]
    domains = sorted({d for d, s in topics})
    plan, swap_targets = [], []
    for hi, (d, s) in enumerate(hosts):
        host = paras[(d, s, 0)]
        far_pool = [(dd, ss) for dd, ss in topics if dd != d and (dd, ss, 1) in paras]
        near_pool = [(dd, ss) for dd, ss in topics if dd == d and ss != s and (dd, ss, 1) in paras]
        if not far_pool or not near_pool:
            continue
        pick = {}
        for kind, pool in (("far", far_pool), ("near", near_pool)):
            for _ in range(50):
                key = rng.choice(pool)
                donor = paras[key + (1,)]
                cand = [j for j in range(1, len(donor) - 1) if usable(donor[j])]
                if cand:
                    pick[kind] = {"text": f"{key[0]}|{key[1]}|1", "idx": rng.choice(cand)}
                    break
        cand = [j for j in range(1, len(host)) if usable(host[j])]
        if len(pick) < 2 or not cand:
            continue
        wj = rng.choice(cand)
        far_dom = rng.choice([x for x in domains if x != d])
        plan.append({"host": f"{d}|{s}|0", "far": pick["far"], "near": pick["near"], "word_idx": wj})
        swap_targets.append({"host": f"{d}|{s}|0", "idx": wj, "s": host[wj], "far": far_dom})
    native = {f"{d}|{s}|{r}": v for (d, s, r), v in paras.items()}
    wr("splits.json", {"native_texts": native, "plan": plan, "swap_targets": swap_targets})
    print(f"paragraphs kept {len(paras)} of {len(texts)}, hosts planned {len(plan)}")


def parse_swap(orig, reply):
    w = re.search(r"WORD:\s*(.+)", reply)
    s = re.search(r"SENTENCE:\s*(.+)", reply)
    if not w or not s:
        return None
    word, new = w.group(1).strip().strip(".\"'"), s.group(1).strip().strip("\"")
    a, b = orig.split(), new.split()
    if len(a) != len(b):  # allow a one-word noun replaced by a two-word noun
        if abs(len(a) - len(b)) > 1:
            return None
    diff = sum(x.strip(".,;:").lower() != y.strip(".,;:").lower() for x, y in zip(a, b)) if len(a) == len(b) else 1
    if diff < 1 or diff > 2 or word.lower() not in new.lower() or word.lower() in orig.lower():
        return None
    return {"word": word, "sentence": new}


def build():
    rng = random.Random(SEED + 1)
    sp = rd("splits.json")
    swaps = rd("swaps.json")
    native = sp["native_texts"]
    items, dropped = [], 0
    for pl, tgt, rep in zip(sp["plan"], sp["swap_targets"], swaps):
        sw = parse_swap(tgt["s"], rep)
        if sw is None:
            dropped += 1
            continue
        host = native[pl["host"]]
        sents = [[s, [pl["host"], j], None] for j, s in enumerate(host)]
        sents[pl["word_idx"]] = [sw["sentence"], [pl["host"], pl["word_idx"]], {"kind": "word", "word": sw["word"]}]
        # insertion slots between host sentences, never before the first; keep odd parts apart
        for kind in ("far", "near"):
            dn = pl[kind]
            odd_pos = [k for k, x in enumerate(sents) if x[2]]
            slots = [k for k in range(1, len(sents) + 1) if all(abs(k - p) >= 2 for p in odd_pos)]
            k = rng.choice(slots or list(range(1, len(sents) + 1)))
            sents.insert(k, [native[dn["text"]][dn["idx"]], [dn["text"], dn["idx"]], {"kind": kind}])
        odd = [dict(x[2], idx=k, sentence=x[0]) for k, x in enumerate(sents) if x[2]]
        base = pl["host"]
        items.append({"id": f"odd:{base}", "host": base, "cond": "odd", "sentences": [x[0] for x in sents],
                      "source": [x[1] for x in sents], "odd": odd})
        items.append({"id": f"ctl:{base}", "host": base, "cond": "control", "sentences": list(host),
                      "source": [[base, j] for j in range(len(host))], "odd": []})
    wr("items.json", {"native_texts": native, "items": items})
    print(f"items {len(items)} ({len(items) // 2} odd + {len(items) // 2} control), swaps dropped {dropped}")


if __name__ == "__main__":
    {"split": split, "build": build}[sys.argv[1]]()
