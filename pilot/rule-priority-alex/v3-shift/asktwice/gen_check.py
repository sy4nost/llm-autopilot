"""Diagnostic only: greedy continuations after 'The answer is' for low-option-mass items (no scoring role)."""
import json, os, math
import modal
MODEL = "Qwen/Qwen2.5-14B-Instruct"
R_TAIL = "\n\nReply with only the answer, nothing else."
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-asktwice-gen", image=image)

HERE = os.path.dirname(os.path.abspath(__file__))


@app.function(gpu="H100", volumes={"/cache": cache}, timeout=1800)
def gen(texts: list):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    out = []
    for t in texts:
        p = tok.apply_chat_template([{"role": "user", "content": t + R_TAIL}], tokenize=False, add_generation_prompt=True) + "The answer is"
        ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        g = model.generate(ids, max_new_tokens=12, do_sample=False)
        out.append(repr(tok.decode(g[0, ids.shape[1]:])))
    return out


@app.local_entrypoint()
def main():
    items = {x["uid"]: x for x in json.load(open(os.path.join(HERE, "sets_smoke.json")))["items"]}
    rows = [json.loads(l) for l in open(os.path.join(HERE, "out_main_smoke.jsonl"))]
    m = lambda sc: sum(math.exp(sum(s)) for s in sc)
    sel = [items[r["uid"]] for r in rows if "res" in r and m(r["res"]["normal"]) < 0.5][:14]
    for it, o in zip(sel, gen.remote([x["text"] for x in sel])):
        print(it["pop"], it["cands"], "|", it["text"][-70:], "->", o)
