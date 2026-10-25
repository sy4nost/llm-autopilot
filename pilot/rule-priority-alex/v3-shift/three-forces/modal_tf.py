"""three-forces generation: greedy, 40 new tokens, batched (left padding).
  modal run --detach modal_tf.py --tag smoke --n 64 ;  modal run modal_tf.py --tag smoke --get"""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-threeforces", image=image)
MODELS = {"qwen14": "Qwen/Qwen2.5-14B-Instruct", "phi4": "microsoft/phi-4", "olmo13": "allenai/OLMo-2-1124-13B-Instruct"}
BS, TOKBUDGET = 32, 12000  # max prompts per batch and max (prompt tokens x batch); OLMo has no GQA


@app.function(gpu="L40S", volumes={"/cache": cache}, timeout=3600)
def run(blob: str, mk: str, tag: str):
    import torch, gzip, base64
    from transformers import AutoModelForCausalLM, AutoTokenizer
    items = json.loads(gzip.decompress(base64.b64decode(blob)))
    tok = AutoTokenizer.from_pretrained(MODELS[mk]); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODELS[mk], dtype=torch.bfloat16).to("cuda").eval()
    texts = [tok.apply_chat_template(it["messages"], tokenize=False, add_generation_prompt=True) for it in items]
    L = [len(tok(t, add_special_tokens=False)["input_ids"]) + 40 for t in texts]
    order = sorted(range(len(items)), key=lambda i: L[i]); out = {}; batches, cur = [], []
    for i in order:
        if cur and (len(cur) + 1 > BS or (len(cur) + 1) * L[i] > TOKBUDGET):
            batches.append(cur); cur = []
        cur.append(i)
    batches += [cur] if cur else []
    os.makedirs("/cache/threeforces", exist_ok=True); done = 0
    for bn, idx in enumerate(batches):
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=40, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            out[items[i]["id"]] = tok.decode(g[j, enc.input_ids.shape[1]:], skip_special_tokens=True)
        done += len(idx); print(mk, done, "/", len(items), flush=True)
        if bn % 10 == 0:
            open(f"/cache/threeforces/{tag}_{mk}.json", "w").write(json.dumps(out)); cache.commit()
    open(f"/cache/threeforces/{tag}_{mk}.json", "w").write(json.dumps(out)); cache.commit()
    return "saved"


@app.function(volumes={"/cache": cache}, timeout=600)
def fetch(tag: str, mk: str):
    cache.reload()
    return open(f"/cache/threeforces/{tag}_{mk}.json").read()


@app.local_entrypoint()
def main(tag: str = "smoke", n: int = 0, get: bool = False):
    import gzip, base64
    if get:
        for mk in MODELS:
            try:
                open(os.path.join(HERE, f"out_{tag}_{mk}.json"), "w").write(fetch.remote(tag, mk)); print(mk, "ok")
            except Exception as e:
                print(mk, "not fetched", repr(e)[:150])
        return
    items = json.load(open(os.path.join(HERE, "sets_tf.json")))["items"]
    if n:
        import random
        items = random.Random(0).sample(items, n)
    blob = base64.b64encode(gzip.compress(json.dumps(items).encode())).decode()
    calls = {mk: run.spawn(blob, mk, tag) for mk in MODELS}
    for mk, c in calls.items():
        try:
            print(mk, c.get())
        except Exception as e:
            print(mk, "FAILED", repr(e)[:300])
