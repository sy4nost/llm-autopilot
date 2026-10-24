"""Logit lens at the DIRECT answer position (lateflip/PLAN.md). One forward pass per item, no generation.
Saves, for every layer, the lens log-prob of each option's distinguishing token.
  modal run --detach modal_lf.py --tag main --models qwen14     (one launch per model; --n 12 for a smoke)
  modal run modal_lf.py --tag main --get                        (writes lf_<tag>_<model>.jsonl)"""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-lateflip", image=image)
MODELS = {"qwen14": "Qwen/Qwen2.5-14B-Instruct", "phi4": "microsoft/phi-4", "olmo13": "allenai/OLMo-2-1124-13B-Instruct"}
R_TAIL = "\n\nReply with only the answer, nothing else."   # as conftrap-lamp/modal_cl.py


@app.function(gpu="L40S", volumes={"/cache": cache}, timeout=2 * 3600)
def run(blob: str, tag: str, mk: str):
    import torch, time, gzip, base64
    from transformers import AutoModelForCausalLM, AutoTokenizer
    items = json.loads(gzip.decompress(base64.b64decode(blob)))
    name = MODELS[mk]; t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    norm, head = model.model.norm, model.lm_head
    d = f"/cache/lf_runs/{tag}/{mk}"; os.makedirs(d, exist_ok=True); ofn = f"{d}/out.jsonl"
    done = set()
    if os.path.exists(ofn):
        keep = []
        for l in open(ofn):
            try:
                x = json.loads(l); keep.append(json.dumps(x) + "\n"); done.add(x["id"])
            except (ValueError, KeyError):
                pass
        open(ofn, "w").writelines(keep)
    of = open(ofn, "a"); checks = dict(model=name, dropped=0, appended=0, final_vs_logits=[])
    for n, it in enumerate(items):
        if it["id"] in done:
            continue
        # exact DIRECT prefix and option strings of conftrap-lamp
        pre = chat(it["prompt"]) if it["mode"] == "label" else chat(it["prompt"] + R_TAIL) + (it["stem"].strip() if it["stem"] else "The answer is")
        opts = list(it["cands"]) if it["mode"] == "label" else [" " + c for c in it["cands"]]
        ids = tok(pre, add_special_tokens=False)["input_ids"]
        ot = [tok(o, add_special_tokens=False)["input_ids"] for o in opts]
        pos = 0
        while len({o[pos] if len(o) > pos else -1 for o in ot}) == 1 and all(len(o) > pos + 1 for o in ot):
            ids = ids + [ot[0][pos]]; pos += 1   # shared leading token(s), e.g. a bare " " before digits
        dist = [o[pos] if len(o) > pos else None for o in ot]
        if None in dist or len(set(dist)) < len(dist):
            checks["dropped"] += 1; of.write(json.dumps(dict(id=it["id"], drop=True)) + "\n"); continue
        checks["appended"] += pos > 0
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids]).cuda(), output_hidden_states=True)
            hs = out.hidden_states   # embeddings, layer 1 .. layer L (HF: last entry already passed the final norm)
            L = len(hs) - 1; rows = []
            for l in range(1, L + 1):
                h = hs[l][0, -1]
                lg = head(h if l == L else norm(h)).float()
                rows.append([round(v, 3) for v in lg.log_softmax(-1)[dist].tolist()])
            real = out.logits[0, -1].float().log_softmax(-1)[dist].tolist()
        if len(checks["final_vs_logits"]) < 5:
            checks["final_vs_logits"].append(round(max(abs(a - b) for a, b in zip(rows[-1], real)), 4))
        of.write(json.dumps(dict(id=it["id"], pos=pos, tokens=[tok.decode([t]) for t in dist], lens=rows)) + "\n")
        if (n + 1) % 50 == 0:
            of.flush(); cache.commit(); print(mk, n + 1, "/", len(items), flush=True)
    checks["seconds"] = round(time.time() - t0)
    of.write(json.dumps(dict(id=-1, checks=checks)) + "\n"); of.close(); cache.commit(); print(mk, "DONE", flush=True)


@app.function(volumes={"/cache": cache}, timeout=900)
def fetch(tag: str, mk: str):
    import gzip, base64
    cache.reload()
    blob = base64.b64encode(gzip.compress(open(f"/cache/lf_runs/{tag}/{mk}/out.jsonl", "rb").read())).decode()
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
            out = os.path.join(HERE, f"lf_{tag}_{mk}.jsonl"); open(out, "wb").write(data); print("wrote", out, data.count(b"\n"))
        return
    items = [x for x in json.load(open(os.path.join(HERE, "..", "conftrap-lamp", "sets_cl.json")))["items"] if x["family"] != "mmlu"]
    if n:
        items = items[::max(1, len(items) // n)][:n]
    blob = base64.b64encode(gzip.compress(json.dumps(items).encode())).decode()
    calls = [run.spawn(blob, tag, mk) for mk in mks]
    for mk, c in zip(mks, calls):
        try:
            c.get(); print(mk, "finished")
        except Exception as e:
            print(mk, "FAILED:", repr(e)[:300])
