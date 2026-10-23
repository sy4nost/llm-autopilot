"""Run Alex's 10 minimal pairs on Qwen2.5 models: every version (A, B) in its own fresh conversation, greedy, bf16.
  modal run modal_bench.py   -> results/<model>.json
"""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
          "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct"]
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-autopilot-bench", image=image)


@app.function(gpu="L40S", volumes={"/cache": cache}, timeout=3600)
def run(model_name: str, tests: list) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(torch.bfloat16).to("cuda").eval()
    jobs = [(t["test"], v, t[v]) for t in tests for v in ("A", "B")]
    # each prompt is its own one-message conversation; batching only shares the GPU, not the context
    prompts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
               for _, _, p in jobs]
    b = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        g = model.generate(**b, max_new_tokens=800, do_sample=False)
    outs = [tok.decode(r[b["input_ids"].shape[1]:], skip_special_tokens=True).strip() for r in g]
    return {"model": model_name, "answers": [{"test": n, "version": v, "answer": o} for (n, v, _), o in zip(jobs, outs)]}


@app.local_entrypoint()
def main(models: str = ",".join(MODELS), tests_file: str = "tests.json", out: str = "results"):
    tests = json.load(open(os.path.join(HERE, tests_file)))
    os.makedirs(os.path.join(HERE, out), exist_ok=True)
    names = [m for m in models.split(",") if m]
    for name, res in zip(names, run.map(names, kwargs={"tests": tests})):
        json.dump(res, open(os.path.join(HERE, out, name.split("/")[-1] + ".json"), "w"), indent=1)
        print("done", name, flush=True)
