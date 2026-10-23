"""SENSITIVITY runner (post-run, report-only; not part of the frozen PREREG). Copy of modal_gl.py::analyze.
   modal run sens_gl.py --tag main --src analyze_sens_w.py [--neqa-not-only 1] --out NAME
General lamp test (PLAN.md v2.1): one plain forward pass per prompt on Qwen2.5-14B-Instruct, no interventions.
Saves per prompt: last-prompt-token state at layers 8..44 (step 4) + layer-0 mean over the prompt tokens, and the summed
log-prob of every answer option. Then the frozen analysis (analyze_gl.py) runs on a Modal CPU box next to the states.
  modal run modal_gl.py --mode devformat                       answer format chosen on the DEV parts (seed-12345 split)
  modal run modal_gl.py --mode smoke --fmt F? --sets sets_smoke.json --tag smoke
  modal run modal_gl.py --mode main  --fmt F? --sets sets_gl.json --seed S --tag main
Public families: inverse-scaling redefine-math / NeQA / quote-repetition, cais/mmlu. Dev/test split fixed with 12345."""
import json, os, re, random
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate", "datasets", "numpy", "scikit-learn")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-general-lamp-sens", image=image)
MODEL = "Qwen/Qwen2.5-14B-Instruct"
LAYERS = [8, 12, 16, 20, 24, 28, 32, 36, 40, 44]
SPLIT_SEED = 12345
NEG = {"can't": "can", "won't": "will", "isn't": "is", "aren't": "are", "doesn't": "does", "don't": "do", "didn't": "did",
       "wasn't": "was", "weren't": "were", "hasn't": "has", "haven't": "have", "couldn't": "could", "wouldn't": "would",
       "shouldn't": "should", "hadn't": "had", "mustn't": "must"}


def unnegate(p):
    """NeQA: remove the single negation of the question line; None unless there is exactly one"""
    lines = p.split("\n")
    qi = [i for i, l in enumerate(lines) if l.startswith("Question: ")]
    if len(qi) != 1:
        return None
    q = lines[qi[0]]
    hits = [("not", m) for m in re.finditer(r"\bnot\b", q, flags=re.I)] + [("nt", m) for m in re.finditer(r"\b\w+n't\b", q, flags=re.I)]
    if len(hits) != 1:
        return None
    kind, m = hits[0]
    if kind == "not":
        new = q[:m.start()].rstrip() + " " + q[m.end():].lstrip()
    else:
        w = m.group(0); base = NEG.get(w.lower())
        if base is None:
            return None
        base = base.capitalize() if w[0].isupper() else base
        new = q[:m.start()] + base + q[m.end():]
    lines[qi[0]] = new
    return "\n".join(lines)


def public_items(S, part):
    """part: 'dev' | 'test'.  Returns prompts for redefine / neqa / quote (+ aux validity prompts) and, for test, 1500 MMLU."""
    from datasets import load_dataset
    out = []
    for fam, src in (("P-redefine", "redefine-math"), ("P-neqa", "NeQA"), ("P-quote", "quote-repetition")):
        ds = load_dataset("inverse-scaling/" + src, split="train")
        n = len(ds)
        ix = list(range(n)); random.Random(0).shuffle(ix); seen = set(ix[:200])           # scored in earlier runs
        if fam == "P-redefine":
            dev = set()
            for t in range(n // 100):
                dev |= set(random.Random(SPLIT_SEED + t).sample(range(t * 100, t * 100 + 100), 10))
        else:
            ix2 = list(range(n)); random.Random(SPLIT_SEED).shuffle(ix2); dev = set(ix2[:max(1, n // 10)])
        for i in range(n):
            if (i in dev) != (part == "dev"):
                continue
            r = ds[i]
            p = r["prompt"].replace("\r\n", "\n").replace("﻿", "")
            base = dict(family=fam, src_i=i, gold=r["answer_index"], classes=list(r["classes"]), seen_before=i in seen,
                        cell=("t%d" % (i // 100)) if fam == "P-redefine" else "all", grp=i)
            if fam == "P-neqa":
                u = unnegate(p)
                if u is None:
                    continue
                out.append(dict(base, prompt=p))
                out.append(dict(base, prompt=u, aux=True))
            elif fam == "P-quote":
                start = p[p.rindex("Output: ") + len("Output: "):]
                out.append(dict(base, prompt=p))
                out.append(dict(base, prompt="Complete the famous quote: " + start, aux=True))
            else:
                out.append(dict(base, prompt=p))
    if part == "test":
        test = load_dataset("cais/mmlu", "all", split="test")
        used, per = set(), {}
        for i, r in enumerate(test):                                              # blockex: first 10 per subject
            if len(per.setdefault(r["subject"], [])) < 10:
                per[r["subject"]].append(i); used.add(i)
        ix = list(range(len(test))); random.Random(185).shuffle(ix); used |= set(ix[:500])   # watch_public
        ix = list(range(len(test))); random.Random(0).shuffle(ix); used |= set(ix[:200])     # conflict
        rest = [i for i in range(len(test)) if i not in used]; random.Random(S).shuffle(rest)
        for k, i in enumerate(rest[:1500]):
            r = test[i]
            q = (f"The following are multiple choice questions (with answers) about {r['subject'].replace('_', ' ')}.\n\n"
                 f"Question: {r['question'].strip()}\n" + "".join(f"{L}. {c}\n" for L, c in zip("ABCD", r["choices"])) + "Answer:")
            out.append(dict(family="mmlu", src_i=i, gold=r["answer"], classes=[" A", " B", " C", " D"], prompt=q, cell="all",
                            grp=i, subject=r["subject"], split="train" if k < 1000 else "heldout"))
    return out


def render(tok, it, fmt):
    """(prefix text, option strings) for an item under an answer format"""
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    if "cands" in it:                                                             # generated families: own format
        return chat(it["prompt"]), list(it["cands"])
    if fmt == "F1":
        return chat(it["prompt"] + "\n\nReply with only the answer, nothing else."), [c.strip() for c in it["classes"]]
    if fmt == "F2":
        return chat(it["prompt"] + "\n\nReply with only the answer, nothing else.") + "The answer is", [" " + c.strip() for c in it["classes"]]
    return it["prompt"], list(it["classes"])


@app.function(cpu=8, memory=32768, volumes={"/cache": cache}, timeout=3 * 3600)
def analyze(tag: str, src: str, meta: list, n_boot: int):
    import glob, io, contextlib, numpy as np
    cache.reload()
    S = np.concatenate([np.load(f) for f in sorted(glob.glob(f"/cache/gl/{tag}/states_*.npy"))])
    g = {}
    exec(compile(src, "analyze_gl.py", "exec"), g)
    feats = lambda ix, L: S[np.asarray(ix), g["LAYERS"].index(L)].astype(np.float32)
    l0 = lambda ix: S[np.asarray(ix), -1].astype(np.float32)
    lines = []
    per = g["run"](meta, feats, l0, B_=n_boot, log=lambda s: (lines.append(s), print(s, flush=True)))
    return "\n".join(lines), per



@app.local_entrypoint()
def main(tag: str = "main", src: str = "analyze_gl.py", neqa_not_only: int = 0, out: str = "sens", n_boot: int = 2000):
    rows = [json.loads(l) for l in open(os.path.join(HERE, f"meta_{tag}.jsonl"))][:-1]
    if neqa_not_only:   # PLAN.md v2.1 rule: NeQA question with exactly one ' not ' (n't forms excluded)
        L = [json.loads(l) for l in open(os.path.join(HERE, "data", "NeQA.jsonl")) if l.strip()]
        def ok(i):
            q = [l for l in L[i]["prompt"].replace("\r\n", "\n").split("\n") if l.startswith("Question: ")][0]
            return len(re.findall(r"\bnot\b", q, flags=re.I)) == 1 and not re.search(r"\b\w+n't\b", q, flags=re.I)
        rows = [r for r in rows if r["family"] != "P-neqa" or ok(r["src_i"])]
    rep, per = analyze.remote(tag, open(os.path.join(HERE, src)).read(), rows, n_boot)
    open(os.path.join(HERE, f"report_{out}.txt"), "w").write(rep + "\n")
    print("wrote", f"report_{out}.txt")
