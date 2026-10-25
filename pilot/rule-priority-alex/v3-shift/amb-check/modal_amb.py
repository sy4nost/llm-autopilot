"""amb-check: parcels/bonus DIRECT forced choice exactly as conftrap-lamp label mode (chat(prompt), options c, summed log-prob).
  modal run --detach modal_amb.py --tag main ;  modal run modal_amb.py --tag main --get"""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-ambcheck", image=image)
MODELS = {"qwen14": "Qwen/Qwen2.5-14B-Instruct", "phi4": "microsoft/phi-4", "olmo13": "allenai/OLMo-2-1124-13B-Instruct"}


@app.function(gpu="L40S", volumes={"/cache": cache}, timeout=3600)
def run(blob: str, mk: str, tag: str = "x"):
    import torch, gzip, base64
    items = json.loads(gzip.decompress(base64.b64decode(blob)))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[mk])
    model = AutoModelForCausalLM.from_pretrained(MODELS[mk], dtype=torch.bfloat16).to("cuda").eval()
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    out = []
    for n, it in enumerate(items):
        pid = tok(chat(it["prompt"]), add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            o = model(input_ids=torch.tensor([pid]).cuda(), use_cache=True)
        last = o.logits[0, -1].float().log_softmax(-1); sc = []
        for c in it["cands"]:
            t = tok(c, add_special_tokens=False)["input_ids"]; s = last[t[0]].item()
            if len(t) > 1:
                import copy
                kv = copy.deepcopy(o.past_key_values)
                with torch.no_grad():
                    lg = model(input_ids=torch.tensor([t[:-1]]).cuda(), past_key_values=kv, use_cache=True).logits[0].float().log_softmax(-1)
                s += sum(lg[i, t[i + 1]].item() for i in range(len(t) - 1))
            sc.append(round(s, 4))
        out.append(dict(id=it["id"], s=sc))
        if n % 100 == 0:
            print(mk, n, flush=True)
    os.makedirs("/cache/ambcheck", exist_ok=True)
    open(f"/cache/ambcheck/{tag}_{mk}.json", "w").write(json.dumps(out)); cache.commit()
    return "saved"


@app.function(volumes={"/cache": cache}, timeout=600)
def fetch(tag: str, mk: str):
    import gzip, base64
    cache.reload()
    return base64.b64encode(gzip.compress(open(f"/cache/ambcheck/{tag}_{mk}.json", "rb").read())).decode()


@app.local_entrypoint()
def main(tag: str = "smoke", n: int = 0, get: bool = False):
    import gzip, base64
    if get:
        for mk in MODELS:
            try:
                r = gzip.decompress(base64.b64decode(fetch.remote(tag, mk))).decode(); open(os.path.join(HERE, f"out_{tag}_{mk}.json"), "w").write(r); print(mk, "ok")
            except Exception as e:
                print(mk, "not fetched", repr(e)[:200])
        return
    items = json.load(open(os.path.join(HERE, "sets_amb.json")))["items"]
    if n:
        items = items[::max(1, len(items) // n)][:n]
    blob = base64.b64encode(gzip.compress(json.dumps(items).encode())).decode()
    calls = {mk: run.spawn(blob, mk, tag) for mk in MODELS}
    for mk, c in calls.items():
        try:
            print(mk, c.get())
        except Exception as e:
            print(mk, "FAILED", repr(e)[:300])
