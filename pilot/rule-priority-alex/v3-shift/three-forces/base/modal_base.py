"""three-forces/base: base vs instruct on identical plain-text transcripts. Greedy, 40 new tokens, repetition penalty forced off, batched (left padding).
  modal run --detach modal_base.py --tag smoke --n 48 ;  modal run modal_base.py --tag smoke --get"""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-threeforces-base", image=image)
MODELS = {"qwen14_base": "Qwen/Qwen2.5-14B", "qwen14_inst": "Qwen/Qwen2.5-14B-Instruct",
          "olmo13_base": "allenai/OLMo-2-1124-13B", "olmo13_inst": "allenai/OLMo-2-1124-13B-Instruct",
          "qwen14_chat_nopen": "Qwen/Qwen2.5-14B-Instruct"}  # chat template, repetition penalty off: re-check of the main run
BS, NEW = 32, 40
TOKBUDGET = {"qwen": 12000, "olmo": 8000}
BOS = {"qwen": "", "olmo": "<|endoftext|>"}  # one explicit rule per family, identical for base and instruct; no automatic special tokens


@app.function(gpu="L40S", volumes={"/cache": cache}, timeout=3600)
def run(blob: str, mk: str, tag: str):
    import torch, gzip, base64
    from transformers import AutoModelForCausalLM, AutoTokenizer
    items = json.loads(gzip.decompress(base64.b64decode(blob)))
    tok = AutoTokenizer.from_pretrained(MODELS[mk]); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODELS[mk], dtype=torch.bfloat16).to("cuda").eval()
    fam = mk.split("1")[0]; print(mk, "generation_config:", model.generation_config.to_dict(), flush=True)
    if mk.endswith("chat_nopen"):
        texts = [tok.apply_chat_template(it["messages"], tokenize=False, add_generation_prompt=True) for it in items]
    else:
        texts = [BOS[fam] + it["text"] for it in items]
    for t in texts[:3]:
        print(mk, "ids head:", tok(t, add_special_tokens=False)["input_ids"][:8], flush=True)
    L = [len(tok(t, add_special_tokens=False)["input_ids"]) + NEW for t in texts]
    order = sorted(range(len(items)), key=lambda i: L[i]); out = {}; batches, cur = [], []
    for i in order:
        if cur and (len(cur) + 1 > BS or (len(cur) + 1) * L[i] > TOKBUDGET[fam]):
            batches.append(cur); cur = []
        cur.append(i)
    batches += [cur] if cur else []
    d = "/cache/threeforces_base"; os.makedirs(d, exist_ok=True); done = 0
    for bn, idx in enumerate(batches):
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=NEW, do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            out[items[i]["id"]] = tok.decode(g[j, enc.input_ids.shape[1]:], skip_special_tokens=True)
        done += len(idx); print(mk, done, "/", len(items), flush=True)
        if bn % 10 == 0:
            open(f"{d}/{tag}_{mk}.json", "w").write(json.dumps(out)); cache.commit()
    open(f"{d}/{tag}_{mk}.json", "w").write(json.dumps(out)); open(f"{d}/{tag}_{mk}.DONE", "w").write("1"); cache.commit()
    return "saved"


@app.function(volumes={"/cache": cache}, timeout=600)
def fetch(tag: str, mk: str):
    cache.reload(); d = "/cache/threeforces_base"
    return json.dumps(dict(out=open(f"{d}/{tag}_{mk}.json").read(), done=os.path.exists(f"{d}/{tag}_{mk}.DONE")))


@app.local_entrypoint()
def main(tag: str = "smoke", n: int = 0, get: bool = False, only: str = ""):
    import gzip, base64
    mks = only.split(",") if only else list(MODELS)
    if get:
        for mk in mks:
            try:
                r = json.loads(fetch.remote(tag, mk)); open(os.path.join(HERE, f"out_{tag}_{mk}.json"), "w").write(r["out"])
                print(mk, "ok", "done" if r["done"] else "PARTIAL", len(json.loads(r["out"])))
            except Exception as e:
                print(mk, "not fetched", repr(e)[:150])
        return
    items = json.load(open(os.path.join(HERE, "sets_base.json")))["items"]
    if n:
        import random
        items = random.Random(1).sample(items, n)
    items = [dict(id=it["id"], text=it["text"], messages=it["messages"]) for it in items]
    blob = base64.b64encode(gzip.compress(json.dumps(items).encode())).decode()
    calls = {mk: run.spawn(blob, mk, tag) for mk in mks}
    for mk, c in calls.items():
        try:
            print(mk, c.get())
        except Exception as e:
            print(mk, "FAILED", repr(e)[:300])
