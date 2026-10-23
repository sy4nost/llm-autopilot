"""Fast vs thought answers on confident traps (conftrap-lamp/PLAN.md); copy of fastthink/modal_ft.py, changes marked CL.
1) THINK reasoning: batched greedy generation (left padding, batches sorted by length), max 512 new tokens.
2) Forced choice for DIRECT and after-THINK: each option its own full forward pass (no cache); per-token log-probs saved.
State lives on the volume (/cache/cl_runs/<tag>/<model>/), committed every batch; a rerun skips what is saved.
  modal run --detach modal_cl.py --sets sets_smoke.json --tag smoke     (also: batched vs single greedy check on 8 prompts)
  modal run --detach modal_cl.py --sets sets_cl.json --tag main --models qwen14   (one launch per model: detach keeps only the last call)
  modal run modal_cl.py --tag main --get                                (writes out_<tag>_<model>.jsonl)
Hard cap: function timeout 3 h per model."""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-conftrap", image=image)
MODELS = {"qwen14": "Qwen/Qwen2.5-14B-Instruct", "phi4": "microsoft/phi-4", "olmo13": "allenai/OLMo-2-1124-13B-Instruct"}
R_TAIL = "\n\nReply with only the answer, nothing else."
BS = 32
TOKBUDGET = 40000  # max (sequences x tokens) per batch; OLMo-2 has no shared KV heads (code review C3)
NSAMP, SAMPNEW = 5, 24  # weight-free rival (PLAN v2): 5 sampled short fast answers, temperature 1


@app.function(gpu="H100", volumes={"/cache": cache}, timeout=3 * 3600)
def run(items: list, check_batch: bool, tag: str, mk: str, MAXNEW: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = MODELS[mk]
    tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    ge = model.generation_config.eos_token_id
    eos = {tok.eos_token_id} | set(ge if isinstance(ge, list) else [ge] if ge is not None else [])
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tok.unk_token_id:
            eos.add(i)
    eos.discard(None)

    def batches(prompts, per_seq_new, nseq=1):
        """index batches sorted by token length, at most BS prompts and TOKBUDGET sequence-tokens each"""
        L = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
        order = sorted(range(len(prompts)), key=lambda i: L[i]); out, cur = [], []
        for i in order:
            if cur and (len(cur) + 1 > BS or (len(cur) + 1) * nseq * (L[i] + per_seq_new) > TOKBUDGET):
                out.append(cur); cur = []
            cur.append(i)
        return out + ([cur] if cur else [])

    GREEDY = dict(do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)  # plain greedy for every model

    def gen_iter(prompts):
        """yields (index, text, truncated) batch by batch"""
        bl = batches(prompts, MAXNEW); done = 0
        for idx in bl:
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=MAXNEW, pad_token_id=tok.pad_token_id, eos_token_id=sorted(eos), **GREEDY)
            for j, i in enumerate(idx):
                new = g[j, enc.input_ids.shape[1]:].tolist()
                yield i, tok.decode(new, skip_special_tokens=True), not any(t in eos for t in new)
            done += len(idx); print(mk, "gen", done, "/", len(prompts), flush=True)

    def score(prefix, opts):
        pid = tok(prefix, add_special_tokens=False)["input_ids"]; res = []
        for o in opts:
            t = tok(o, add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                lg = model(input_ids=torch.tensor([pid + t]).cuda()).logits[0, len(pid) - 1:-1].float()
            lp = torch.log_softmax(lg, -1)
            res.append([round(lp[i, t[i]].item(), 4) for i in range(len(t))])
        return res

    def pre(it):  # CL: prompts that already end with "Return only the answer." are read straight after the chat template
        if it["mode"] == "label":
            return chat(it["prompt"])
        return chat(it["prompt"] + R_TAIL) + (it["stem"].strip() if it["stem"] else "The answer is")

    def direct(it):
        return score(pre(it), list(it["cands"]) if it["mode"] == "label" else [" " + c for c in it["cands"]])

    def after(it, g):
        base = chat(it["think"])
        pre = base + g[:g.rindex("Answer:")] + "Answer:" if "Answer:" in g else base + g.rstrip() + "\nAnswer:"
        return score(pre + (" " + it["stem"].strip() if it["stem"] else ""), [" " + c for c in it["cands"]])

    d = f"/cache/cl_runs/{tag}/{mk}"; os.makedirs(d, exist_ok=True)
    gfn, ofn = f"{d}/gen.jsonl", f"{d}/out.jsonl"
    import time; t0 = time.time()  # CL: wall-clock stamps (review N8)
    checks = dict(model=name, eos=sorted(eos), maxnew=MAXNEW, gen_config=str(model.generation_config.to_diff_dict())[:400], template_head=chat("Hi")[:120], template_tail=chat("Hi")[-120:],
                  head_ids=tok(chat("Hi"), add_special_tokens=False)["input_ids"][:6])
    if check_batch:  # smoke: first 8 items (mixed tasks); main: 16 spread items -> batched vs single greedy (report-only)
        ps = [chat(it["think"]) for it in (items[:8] if tag == "smoke" else items[::max(1, len(items) // 16)][:16])]
        bat = [None] * len(ps)
        for i, t, tr in gen_iter(ps):
            bat[i] = t
        single = []
        for p in ps:
            enc = tok(p, return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=MAXNEW, pad_token_id=tok.pad_token_id, eos_token_id=sorted(eos), **GREEDY)
            single.append(tok.decode(g[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
        checks["batch_vs_single_identical"] = sum(a == b for a, b in zip(bat, single))
        checks["batch_vs_single_same_answer_line"] = sum(a.strip().split("\n")[-1] == b.strip().split("\n")[-1] for a, b in zip(bat, single))
        checks["n_checked"] = len(ps)
    import hashlib
    items_by = {it["id"]: it for it in items}
    H = lambda it: hashlib.sha256((it["think"] + it["prompt"] + it["stem"] + json.dumps(it["cands"]) + str(MAXNEW)).encode()).hexdigest()[:16]
    gens = {}
    if os.path.exists(gfn):
        for l in open(gfn):
            try:
                g = json.loads(l)
                if g.get("h") == H(items_by[g["id"]]) if g["id"] in items_by else False:
                    gens[g["id"]] = (g["text"], g["trunc"])
            except ValueError:  # a line cut by a crash
                pass
    good = []
    if os.path.exists(ofn):  # keep only whole scored lines
        for l in open(ofn):
            try:
                x = json.loads(l)
                if "res" in x and x["id"] in items_by and x.get("h") == H(items_by[x["id"]]):
                    good.append(json.dumps(x) + "\n")
            except ValueError:
                pass
        open(ofn, "w").writelines(good)
    done = {json.loads(l)["id"] for l in good}
    gf = open(gfn, "a")
    todo = [it for it in items if it["id"] not in gens]
    print(mk, "already", len(gens), "to generate", len(todo), flush=True)
    for k, (i, t, tr) in enumerate(gen_iter([chat(it["think"]) for it in todo])):
        gens[todo[i]["id"]] = (t, tr)
        gf.write(json.dumps(dict(id=todo[i]["id"], text=t, trunc=tr, h=H(todo[i]))) + "\n")
        if (k + 1) % BS == 0 or k + 1 == len(todo):
            gf.flush(); cache.commit()
    gf.close()
    # weight-free rival: sampled short fast answers, same prefix as DIRECT, batched, fixed torch seed per batch
    sfn = f"{d}/samp.jsonl"; samp = {}
    if os.path.exists(sfn):
        for l in open(sfn):
            try:
                g = json.loads(l)
                if g["id"] in items_by and g.get("h") == H(items_by[g["id"]]):
                    samp[g["id"]] = g["s"]
            except ValueError:
                pass
    todo = [it for it in items if it["id"] not in samp]
    sf = open(sfn, "a"); nd = 0
    for idx in batches([pre(it) for it in todo], SAMPNEW, NSAMP):
        b = todo[idx[0]]["id"]
        enc = tok([pre(todo[i]) for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        torch.manual_seed(b)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=SAMPNEW, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                               num_return_sequences=NSAMP, pad_token_id=tok.pad_token_id, eos_token_id=sorted(eos), repetition_penalty=1.0)  # CL: review S3
        for j, i in enumerate(idx):
            ss = [tok.decode(g[j * NSAMP + q, enc.input_ids.shape[1]:], skip_special_tokens=True) for q in range(NSAMP)]
            samp[todo[i]["id"]] = ss; sf.write(json.dumps(dict(id=todo[i]["id"], s=ss, h=H(todo[i]))) + "\n")
        nd += len(idx); sf.flush(); cache.commit(); print(mk, "samp", nd, "/", len(todo), flush=True)
    sf.close()
    of = open(ofn, "a")
    for n, it in enumerate(items):
        if it["id"] in done:
            continue
        g, tr = gens[it["id"]]
        res = {"direct": direct(it), "think": after(it, g)}
        meta = {k: v for k, v in it.items() if k not in ("prompt", "core", "think")}
        of.write(json.dumps(dict(meta, model=mk, res=res, gen_think=g, trunc_think=tr, samples=samp[it["id"]], h=H(it))) + "\n")
        if (n + 1) % 100 == 0:
            of.flush(); cache.commit(); print(mk, "score", n + 1, "/", len(items), flush=True)
    checks["seconds"] = round(time.time() - t0)
    of.write(json.dumps(dict(checks=checks)) + "\n"); of.close(); cache.commit()
    print(mk, "DONE", ofn, flush=True)


@app.function(volumes={"/cache": cache}, timeout=900)
def fetch(tag: str, mk: str):
    """returns out.jsonl from the volume in <1 MiB chunks (the proxy blocks Modal's large-blob host)"""
    import gzip, base64
    cache.reload()
    blob = base64.b64encode(gzip.compress(open(f"/cache/cl_runs/{tag}/{mk}/out.jsonl", "rb").read())).decode()
    for i in range(0, len(blob), 900000):
        yield blob[i:i + 900000]


@app.local_entrypoint()
def main(sets: str = "sets_smoke.json", tag: str = "smoke", get: bool = False, models: str = "qwen14,phi4,olmo13", maxnew: int = 512):
    mks = models.split(",")
    if get:
        import gzip, base64
        for mk in mks:
            out = os.path.join(HERE, f"out_{tag}_{mk}.jsonl")
            try:
                data = gzip.decompress(base64.b64decode("".join(fetch.remote_gen(tag, mk))))
            except Exception as e:
                print(mk, "not fetched:", repr(e)[:200]); continue
            open(out, "wb").write(data); print("wrote", out, data.count(b"\n"), "lines")
        return
    items = json.load(open(os.path.join(HERE, sets)))["items"]
    calls = [run.spawn(items, True, tag, mk, maxnew) for mk in mks]
    for mk, c in zip(mks, calls):
        try:
            c.get(); print(mk, "finished")
        except Exception as e:
            print(mk, "FAILED:", repr(e)[:300])
