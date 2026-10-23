"""Run the v3 shift sets (sets_shift.json) on Qwen2.5 models.
Every prompt is its own one-message conversation. Per prompt: 1 greedy answer + 4 sampled (T=0.7, fixed seed).
  modal run modal_rules.py   -> results/<model>.json
"""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
          "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct"]
N_SAMPLES = 4
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-rule-shift", image=image)


@app.function(gpu="H100", volumes={"/cache": cache}, timeout=4 * 3600)
def run(model_name: str, items: list) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(torch.bfloat16).to("cuda").eval()
    items = sorted(items, key=lambda x: (x.get("max_new", 80), len(x["prompt"])))  # similar lengths share a batch
    prompts = [tok.apply_chat_template([{"role": "user", "content": x["prompt"]}], tokenize=False,
                                       add_generation_prompt=True) for x in items]
    out = [dict(level=x["level"], set=x["set"], arm=x["arm"], fmt=x.get("fmt", "six"), probe_ids=[q["id"] for q in x["probes"]],
                greedy=None, samples=[]) for x in items]
    bs = 8
    for i in range(0, len(prompts), bs):
        b = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to("cuda")
        L = b["input_ids"].shape[1]
        mx = max(x.get("max_new", 80) for x in items[i:i + bs])
        with torch.no_grad():
            g = model.generate(**b, max_new_tokens=mx, do_sample=False)
            torch.manual_seed(1000 + i)
            s = model.generate(**b, max_new_tokens=mx, do_sample=True, temperature=0.7, top_p=1.0, top_k=0,
                               num_return_sequences=N_SAMPLES)
        for j in range(g.shape[0]):
            out[i + j]["greedy"] = tok.decode(g[j][L:], skip_special_tokens=True).strip()
            out[i + j]["samples"] = [tok.decode(s[j * N_SAMPLES + k][L:], skip_special_tokens=True).strip()
                                     for k in range(N_SAMPLES)]
        print(model_name, i + bs, "/", len(prompts), flush=True)
    return {"model": model_name, "answers": out}


@app.local_entrypoint()
def main(models: str = ",".join(MODELS), sets_file: str = "sets.json", out: str = "results"):
    items = json.load(open(os.path.join(HERE, sets_file)))
    os.makedirs(os.path.join(HERE, out), exist_ok=True)
    names = [m for m in models.split(",") if m]
    for name, res in zip(names, run.map(names, kwargs={"items": items})):
        json.dump(res, open(os.path.join(HERE, out, name.split("/")[-1] + ".json"), "w"), indent=1)
        print("done", name, flush=True)
