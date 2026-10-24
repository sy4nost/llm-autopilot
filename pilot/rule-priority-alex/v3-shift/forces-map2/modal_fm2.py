"""forces-map2 generation: chat template, greedy, repetition penalty off, batched (left padding). One container per model
runs all steps: d (DIRECT, 48 tokens), mreq/mex/mkn (three map questions, 40 each), th (THINK, 512), slow (SLOW, 512).
  modal run modal_fm2.py --tag smoke --n 30 ;  modal run modal_fm2.py --tag smoke --get"""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-forces-map2", image=image)
MODELS = {"qwen14": "Qwen/Qwen2.5-14B-Instruct", "phi4": "microsoft/phi-4", "olmo13": "allenai/OLMo-2-1124-13B-Instruct"}
STEPS = {"d": 48, "mreq": 40, "mex": 40, "mkn": 40, "th": 512, "slow": 512}   # v1.2 (review M1, M2)
BS, TOKBUDGET = 32, 12000


@app.function(gpu="L40S", volumes={"/cache": cache}, timeout=3600)
def run(blob: str, mk: str, tag: str):
    import torch, gzip, base64
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sets = json.loads(gzip.decompress(base64.b64decode(blob)))
    tok = AutoTokenizer.from_pretrained(MODELS[mk]); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODELS[mk], dtype=torch.bfloat16).to("cuda").eval()
    d = "/cache/forces-map2"; os.makedirs(d, exist_ok=True)
    for step, items in sets.items():
        if os.path.exists(f"{d}/{tag}_{step}_{mk}.json") and len(json.load(open(f"{d}/{tag}_{step}_{mk}.json"))) == len(items):
            print(mk, step, "already done, skipped", flush=True); continue   # resume after a stopped run (spend limit)
        NEW = STEPS[step]
        texts = [tok.apply_chat_template(it["messages"], tokenize=False, add_generation_prompt=True) for it in items]
        L = [len(tok(t, add_special_tokens=False)["input_ids"]) + NEW for t in texts]
        order = sorted(range(len(items)), key=lambda i: L[i]); out = {}; batches, cur = [], []
        for i in order:
            if cur and (len(cur) + 1 > BS or (len(cur) + 1) * L[i] > TOKBUDGET):
                batches.append(cur); cur = []
            cur.append(i)
        batches += [cur] if cur else []
        for idx in batches:
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=NEW, do_sample=False, temperature=None, top_p=None, top_k=None,
                                   repetition_penalty=1.0, pad_token_id=tok.pad_token_id)
            for j, i in enumerate(idx):
                out[items[i]["id"]] = tok.decode(g[j, enc.input_ids.shape[1]:], skip_special_tokens=True)
            print(mk, step, len(out), "/", len(items), flush=True)
        open(f"{d}/{tag}_{step}_{mk}.json", "w").write(json.dumps(out)); cache.commit()
    open(f"{d}/{tag}_{mk}.DONE", "w").write("1"); cache.commit()
    return "saved"


@app.function(volumes={"/cache": cache}, timeout=600)
def fetch(tag: str, mk: str):
    cache.reload(); d = "/cache/forces-map2"
    return json.dumps({s: open(f"{d}/{tag}_{s}_{mk}.json").read() for s in STEPS if os.path.exists(f"{d}/{tag}_{s}_{mk}.json")})


@app.local_entrypoint()
def main(tag: str = "smoke", n: int = 0, get: bool = False, only: str = ""):
    import gzip, base64, random
    mks = only.split(",") if only else list(MODELS)
    if get:
        for mk in mks:
            try:
                r = json.loads(fetch.remote(tag, mk))
                for s, txt in r.items():
                    open(os.path.join(HERE, f"out_{tag}_{s}_{mk}.json"), "w").write(txt)
                print(mk, "ok", {s: len(json.loads(t)) for s, t in r.items()})
            except Exception as e:
                print(mk, "not fetched", repr(e)[:150])
        return
    sets = {s: json.load(open(os.path.join(HERE, f"sets_{s}.json")))["items"] for s in STEPS}
    if n:   # smoke: n dialogues, balanced over the 10 group/sub groups
        ids = sorted(it["id"] for it in sets["d"]); groups = {}
        for i in ids:
            groups.setdefault(i.rsplit("|", 1)[0], []).append(i)
        keep = {i for g in groups.values() for i in random.Random(2).sample(g, n // len(groups))}
        sets = {s: [it for it in v if it["id"] in keep] for s, v in sets.items()}
    blob = base64.b64encode(gzip.compress(json.dumps({s: [dict(id=it["id"], messages=it["messages"]) for it in v]
                                                      for s, v in sets.items()}).encode())).decode()
    calls = {mk: run.spawn(blob, mk, tag) for mk in mks}
    for mk, c in calls.items():
        try:
            print(mk, c.get())
        except Exception as e:
            print(mk, "FAILED", repr(e)[:300])
