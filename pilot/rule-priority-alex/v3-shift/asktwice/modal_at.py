"""Ask-twice checker (PLAN.md v2.1) on Qwen2.5-14B-Instruct. Every option scored by its own full forward pass (no cache,
no batching); per-token log-probs saved.  Format F2: chat(text + R_TAIL) + "The answer is", options " " + c.
  modal run modal_at.py --mode dev  --sets sets_dev.json --tag dev     (normal only: variant choice, M threshold, H prior)
  modal run modal_at.py --mode main --sets sets_at.json  --tag main    (normal, disg, para, instr, truth)
"""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-asktwice", image=image)
MODEL = "Qwen/Qwen2.5-14B-Instruct"
R_TAIL = "\n\nReply with only the answer, nothing else."


@app.function(gpu="H100", volumes={"/cache": cache}, timeout=6 * 3600)
def run(mode: str, items: list):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)

    def score(prefix, opts):
        pid = tok(prefix, add_special_tokens=False)["input_ids"]; out = []
        for o in opts:
            t = tok(o, add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                lg = model(input_ids=torch.tensor([pid + t]).cuda()).logits[0, len(pid) - 1:-1].float()
            lp = torch.log_softmax(lg, -1)
            out.append([round(lp[i, t[i]].item(), 4) for i in range(len(t))])
        return out

    ans = lambda text, cands: score(chat(text + R_TAIL) + "The answer is", [" " + c for c in cands])
    rep = 0.0
    for n, it in enumerate(items):
        res = {"normal": ans(it["text"], it["cands"])}
        if n < 5:
            r2 = ans(it["text"], it["cands"])
            rep = max(rep, max(abs(x - y) for u, v in zip(r2, res["normal"]) for x, y in zip(u, v)))
        if mode == "main":
            res["disg"] = res["normal"] if it["disg"] == it["text"] else ans(it["disg"], it["cands"])
            res["para"] = ans(it["para"], it["cands"])
            res["instr"] = ans(it["instr"], it["cands"])
            if it["truth"]:
                res["truth"] = score(chat(it["truth"]) + "The answer is", [" yes", " no"])
        meta = {k: v for k, v in it.items() if k not in ("text", "disg", "para", "instr", "truth")}
        yield json.dumps(dict(meta, res=res))
        if (n + 1) % 100 == 0:
            print("at", n + 1, "/", len(items), flush=True)
    yield json.dumps(dict(checks=dict(repeat_max_diff=rep)))


@app.local_entrypoint()
def main(mode: str = "dev", sets: str = "sets_dev.json", tag: str = "dev"):
    items = json.load(open(os.path.join(HERE, sets)))["items"]
    out = os.path.join(HERE, f"out_{mode}_{tag}.jsonl")
    with open(out, "w") as f:
        for line in run.remote_gen(mode, items):
            f.write(line + "\n"); f.flush()
    print("wrote", out)
