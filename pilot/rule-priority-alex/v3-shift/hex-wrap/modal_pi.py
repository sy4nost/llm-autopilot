"""pi-wrap: fast forced choice exactly as conftrap-lamp DIRECT (chat(prompt + R_TAIL) + "The answer is", options " "+c, summed log-prob).
  modal run modal_pi.py --tag main      (all 3 models in parallel; small, results returned directly)"""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-hexwrap", image=image)
MODELS = {"qwen14": "Qwen/Qwen2.5-14B-Instruct", "phi4": "microsoft/phi-4", "olmo13": "allenai/OLMo-2-1124-13B-Instruct"}
R_TAIL = "\n\nReply with only the answer, nothing else."


@app.function(gpu="L40S", volumes={"/cache": cache}, timeout=3600)
def run(blob: str, mk: str):
    import torch, gzip, base64
    items = json.loads(gzip.decompress(base64.b64decode(blob)))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[mk])
    model = AutoModelForCausalLM.from_pretrained(MODELS[mk], dtype=torch.bfloat16).to("cuda").eval()
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    out = []
    for it in items:
        pid = tok(chat(it["prompt"] + R_TAIL) + "The answer is", add_special_tokens=False)["input_ids"]; sc = []
        for c in it["cands"]:
            t = tok(" " + c, add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                lp = model(input_ids=torch.tensor([pid + t]).cuda()).logits[0, len(pid) - 1:-1].float().log_softmax(-1)
            sc.append(round(sum(lp[i, t[i]].item() for i in range(len(t))), 4))
        out.append(dict(id=it["id"], s=sc))
    import gzip as g2, base64 as b2
    return b2.b64encode(g2.compress(json.dumps(out).encode())).decode()


@app.local_entrypoint()
def main(tag: str = "smoke", n: int = 0):
    items = json.load(open(os.path.join(HERE, "sets_pi.json")))["items"]
    if n:
        items = items[:n]
    import gzip, base64
    blob = base64.b64encode(gzip.compress(json.dumps(items).encode())).decode()
    calls = {mk: run.spawn(blob, mk) for mk in MODELS}
    for mk, c in calls.items():
        try:
            r = gzip.decompress(base64.b64decode(c.get())).decode(); open(os.path.join(HERE, f"out_{tag}_{mk}.json"), "w").write(r); print(mk, "ok")
        except Exception as e:
            print(mk, "FAILED", repr(e)[:300])
