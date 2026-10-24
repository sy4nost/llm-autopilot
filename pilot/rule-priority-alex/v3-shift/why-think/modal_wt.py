"""why-think (PLAN.md): re-read the final answer after edited reasonings / with attention blocks. No generation.
  modal run --detach modal_wt.py --tag main --models qwen14        (one launch per model; --n 6 for a smoke)
  modal run modal_wt.py --tag main --get                           (writes out_<tag>_<model>.jsonl)"""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-whythink", image=image)
MODELS = {"qwen14": "Qwen/Qwen2.5-14B-Instruct", "phi4": "microsoft/phi-4", "olmo13": "allenai/OLMo-2-1124-13B-Instruct"}


@app.function(gpu="L40S", volumes={"/cache": cache}, timeout=3 * 3600)
def run(blob: str, tag: str, mk: str):
    import torch, time, gzip, base64
    from transformers import AutoModelForCausalLM, AutoTokenizer
    jobs = json.loads(gzip.decompress(base64.b64decode(blob)))
    name = MODELS[mk]; t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    neg = torch.finfo(model.dtype).min
    d = f"/cache/wt_runs/{tag}/{mk}"; os.makedirs(d, exist_ok=True); ofn = f"{d}/out.jsonl"
    done = set()
    if os.path.exists(ofn):
        keep = []
        for l in open(ofn):
            try:
                x = json.loads(l); keep.append(json.dumps(x) + "\n"); done.add(x["id"])
            except (ValueError, KeyError):
                pass
        open(ofn, "w").writelines(keep)

    def forward(ids, rows_from, cols, opts):
        """summed log-prob of each option " "+c, options batched; rows from the answer on (incl. option tokens) may not see cols"""
        L = [len(ids) + len(o) for o in opts]; T = max(L)
        x = torch.full((len(opts), T), tok.pad_token_id if tok.pad_token_id is not None else 0, dtype=torch.long)
        for k, o in enumerate(opts):
            x[k, :L[k]] = torch.tensor(ids + o)
        m = torch.full((T, T), neg, device="cuda", dtype=model.dtype).triu(1)
        if cols:
            c = torch.zeros(T, dtype=torch.bool, device="cuda"); c[cols] = True
            m[rows_from:, c] = neg
        with torch.no_grad():
            lg = model(input_ids=x.cuda(), attention_mask=m[None, None].expand(len(opts), 1, T, T)).logits.float()
        out = []
        for k, o in enumerate(opts):
            lp = lg[k, len(ids) - 1:len(ids) - 1 + len(o)].log_softmax(-1)
            out.append(round(sum(lp[i, t].item() for i, t in enumerate(o)), 4))
        return out

    of = open(ofn, "a"); checks = []
    for n, j in enumerate(jobs):
        if j["id"] in done:
            continue
        U = j["user"]; s = chat(U); ua = s.find(U); assert ua >= 0
        opts = [tok(" " + c, add_special_tokens=False)["input_ids"] for c in j["cands"]]
        res = {}
        for arm, spec in j["arms"].items():
            P = spec["prefix"]
            if P is None:   # FILLER / FILLTXT: as many filler tokens as the reasoning had
                n_t = len(tok(spec["n_of"], add_special_tokens=False)["input_ids"])
                unit = spec["fill"]; k = max(1, len(tok(unit, add_special_tokens=False)["input_ids"]))
                P = (unit * (n_t // k + 1)).strip()
            text = s + (P + "\n\nAnswer:" if P else "Answer:")
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True); ids, offs = enc["input_ids"], enc["offset_mapping"]
            ans_char = len(s) + len(P)
            rows_from = next(i for i, (a, b) in enumerate(offs) if b > ans_char)
            cols = None
            if spec.get("ko"):
                rng = [((ua if w == "u" else len(s)) + a, (ua if w == "u" else len(s)) + b) for w, a, b in spec["ko"]]
                cols = [i for i, (a, b) in enumerate(offs) if i < rows_from and b > a and any(a < B and b > A for A, B in rng)]
            sc = forward(ids, rows_from, cols, opts)
            res[arm] = dict(s=sc, nblock=len(cols) if cols is not None else 0, T=len(ids))
            if arm == "FULL" and len(checks) < 5:   # an empty block must equal the plain run
                plain = forward(ids, len(ids), None, opts)
                with torch.no_grad():
                    ref = []
                    for o in opts:
                        lg = model(input_ids=torch.tensor([ids + o]).cuda()).logits[0].float()[len(ids) - 1:-1].log_softmax(-1)
                        ref.append(sum(lg[i, t].item() for i, t in enumerate(o)))
                checks.append(round(max(abs(a - b) for a, b in zip(plain, ref)), 4))
        of.write(json.dumps(dict(id=j["id"], family=j["family"], group=j["group"], key=j["key"], habit=j["habit"], cands=j["cands"],
                                 res=res)) + "\n")
        if (n + 1) % 10 == 0:
            of.flush(); cache.commit(); print(mk, n + 1, "/", len(jobs), flush=True)
    of.write(json.dumps(dict(id=-1, checks=dict(model=name, mask_check=checks, seconds=round(time.time() - t0)))) + "\n")
    of.close(); cache.commit(); print(mk, "DONE", flush=True)


@app.function(volumes={"/cache": cache}, timeout=900)
def fetch(tag: str, mk: str):
    import gzip, base64
    cache.reload()
    blob = base64.b64encode(gzip.compress(open(f"/cache/wt_runs/{tag}/{mk}/out.jsonl", "rb").read())).decode()
    for i in range(0, len(blob), 900000):
        yield blob[i:i + 900000]


@app.local_entrypoint()
def main(tag: str = "smoke", get: bool = False, models: str = "qwen14,phi4,olmo13", n: int = 0):
    import gzip, base64
    mks = models.split(",")
    if get:
        for mk in mks:
            try:
                data = gzip.decompress(base64.b64decode("".join(fetch.remote_gen(tag, mk))))
            except Exception as e:
                print(mk, "not fetched:", repr(e)[:200]); continue
            out = os.path.join(HERE, f"out_{tag}_{mk}.jsonl"); open(out, "wb").write(data); print("wrote", out, data.count(b"\n"))
        return
    J = json.load(open(os.path.join(HERE, "jobs_wt.json")))
    calls = []
    for mk in mks:
        jobs = J[mk]
        if n:   # smoke: n jobs spread over families/groups
            jobs = jobs[::max(1, len(jobs) // n)][:n]
        calls.append(run.spawn(base64.b64encode(gzip.compress(json.dumps(jobs).encode())).decode(), tag, mk))
    for mk, c in zip(mks, calls):
        try:
            c.get(); print(mk, "finished")
        except Exception as e:
            print(mk, "FAILED:", repr(e)[:300])
