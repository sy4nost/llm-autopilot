"""General lamp test (PLAN.md v2.1): one plain forward pass per prompt on Qwen2.5-14B-Instruct, no interventions.
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
app = modal.App("silo-general-lamp", image=image)
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


@app.function(gpu="H100", volumes={"/cache": cache}, timeout=6 * 3600, memory=65536)
def collect(mode: str, fmt: str, synth: list, S: int, tag: str, n_small: int):
    import numpy as np, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def score(prefix, opts, want_states):
        """summed log-prob of each option after the prefix: one prefix pass, options continued from a copy of its cache"""
        import copy
        pid = tok(prefix, add_special_tokens=False)["input_ids"]
        n = len(pid)
        with torch.no_grad():
            o = model(input_ids=torch.tensor([pid]).cuda(), use_cache=True, output_hidden_states=want_states, logits_to_keep=1)
        first = torch.log_softmax(o.logits[0, -1].float(), -1)
        sc = []
        for opt in opts:
            t = tok(opt, add_special_tokens=False)["input_ids"]
            s = first[t[0]].item()
            if len(t) > 1:
                with torch.no_grad():
                    o2 = model(input_ids=torch.tensor([t[:-1]]).cuda(), past_key_values=copy.deepcopy(o.past_key_values), use_cache=True)
                lp = torch.log_softmax(o2.logits[0].float(), -1)
                s += sum(lp[i, t[i + 1]].item() for i in range(len(t) - 1))
            sc.append(round(s, 4))
        st = None
        if want_states:
            hs = o.hidden_states
            st = np.stack([hs[l][0, n - 1].float().cpu().numpy() for l in LAYERS] + [hs[0][0, :n].float().mean(0).cpu().numpy()]).astype(np.float16)
        return sc, st, n

    def score_full(prefix, opt):
        """check: the same option scored on the full sequence, no cache"""
        pid = tok(prefix, add_special_tokens=False)["input_ids"]; t = tok(opt, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            lg = model(input_ids=torch.tensor([pid + t]).cuda()).logits[0, len(pid) - 1:-1].float()
        lp = torch.log_softmax(lg, -1)
        return sum(lp[i, t[i]].item() for i in range(len(t)))

    if mode == "devformat":
        items = public_items(S, "dev")
        res = {}
        for fm in ("F1", "F2", "F3"):
            for it in items:
                if it.get("aux"):
                    continue
                pre, opts = render(tok, it, fm)
                sc, _, _ = score(pre, opts, False)
                res.setdefault(fm, {}).setdefault(it["family"], []).append(float(np.exp(np.array(sc)).sum()))
            print("devformat", fm, "done", flush=True)
        med = {fm: {f: float(np.median(v)) for f, v in d.items()} for fm, d in res.items()}
        share = {fm: {f: float(np.mean(np.array(v) >= 0.5)) for f, v in d.items()} for fm, d in res.items()}
        best = max(med, key=lambda fm: min(med[fm].values()))
        yield json.dumps(dict(devformat=True, median_mass=med, share_ge_05=share, chosen=best,
                              n={f: len(v) for f, v in res["F1"].items()}))
        return

    items = [dict(x, uid=f"s{k}") for k, x in enumerate(synth)]
    pub = public_items(S, "test")
    if n_small:
        keep, cnt = [], {}
        for x in pub:
            key = (x["family"], x.get("split", ""))
            base_ok = cnt.get(key, 0) < n_small
            if not x.get("aux"):
                if not base_ok:
                    continue
                cnt[key] = cnt.get(key, 0) + 1
                keep.append(x)
            elif keep and keep[-1]["family"] == x["family"] and keep[-1]["src_i"] == x["src_i"]:
                keep.append(x)
        pub = keep
    items += [dict(x, uid=f"p{k}") for k, x in enumerate(pub)]
    out_dir = f"/cache/gl/{tag}"; os.makedirs(out_dir, exist_ok=True)
    buf, chunk, row = [], 0, 0
    checks = dict(cache_check_max_diff_top=0.0, multi_token_options_checked=0)
    for k, it in enumerate(items):
        pre, opts = render(tok, it, fmt)
        want = not it.get("aux")
        sc, st, n = score(pre, opts, want)
        if k < 6 or (k < 400 and k % 40 == 0):                                     # cached scoring == full-sequence scoring
            j = max(range(len(opts)), key=lambda j: sc[j])
            checks["cache_check_max_diff_top"] = max(checks["cache_check_max_diff_top"], abs(score_full(pre, opts[j]) - sc[j]))
            checks["multi_token_options_checked"] += int(any(len(tok(o, add_special_tokens=False)["input_ids"]) > 1 for o in opts))
        meta = {kk: v for kk, v in it.items() if kk != "prompt"}
        meta.update(scores=sc, n_tokens=n, fmt=fmt if "cands" not in it else "own")
        if want:
            meta["row_index"] = row; row += 1; buf.append(st)
            if len(buf) == 500:
                np.save(f"{out_dir}/states_{chunk:03d}.npy", np.stack(buf)); buf, chunk = [], chunk + 1
        yield json.dumps(meta)
        if (k + 1) % 100 == 0:
            print("collect", k + 1, "/", len(items), flush=True)
    if buf:
        np.save(f"{out_dir}/states_{chunk:03d}.npy", np.stack(buf))
    cache.commit()
    yield json.dumps(dict(checks=checks, n_rows=row, n_items=len(items)))


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
def main(mode: str = "smoke", fmt: str = "F3", sets: str = "sets_smoke.json", seed: int = 12345, tag: str = "smoke",
         n_small: int = 0, n_boot: int = 2000, analyze_only: int = 0):
    if mode == "devformat":
        for line in collect.remote_gen("devformat", "", [], seed, "dev", 0):
            open(os.path.join(HERE, "devformat_out.json"), "w").write(line + "\n"); print(line)
        return
    meta_path = os.path.join(HERE, f"meta_{tag}.jsonl")
    if not analyze_only:
        synth = json.load(open(os.path.join(HERE, sets)))["items"]
        with open(meta_path, "w") as f:
            for line in collect.remote_gen(mode, fmt, synth, seed, tag, n_small):
                f.write(line + "\n"); f.flush()
    rows = [json.loads(l) for l in open(meta_path)]
    print("checks:", rows[-1])
    src = open(os.path.join(HERE, "analyze_gl.py")).read()
    rep, per = analyze.remote(tag, src, rows[:-1], n_boot)
    open(os.path.join(HERE, f"report_{tag}.txt"), "w").write(rep + "\n")
    json.dump(per, open(os.path.join(HERE, f"scores_{tag}.json"), "w"))
    print("wrote", f"report_{tag}.txt", f"scores_{tag}.json")
