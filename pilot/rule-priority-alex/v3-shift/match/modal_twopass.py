"""Two-pass run (sets_twopass.json) on Qwen2.5-14B, H100.  modal run modal_twopass.py -> results_twopass/<model>.json
Stage 1: greedy answer to every distinct pass-1 prompt (no examples) = the draft label.
Stage 2: pass-2 prompt with {draft} filled in; 1 greedy + 4 samples (T=0.7), as in the other runs."""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-rule-twopass", image=image)
N_SAMPLES = 4


@app.function(gpu="H100", volumes={"/cache": cache}, timeout=4 * 3600)
def run(model_name: str, items: list) -> dict:
    import re, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(torch.bfloat16).to("cuda").eval()
    chat = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)

    def gen(prompts, sample, tag, mx=20):
        order = sorted(range(len(prompts)), key=lambda k: len(prompts[k]))
        res = [None] * len(prompts)
        for i in range(0, len(order), 8):
            idx = order[i:i + 8]
            b = tok([chat(prompts[k]) for k in idx], return_tensors="pt", padding=True).to("cuda")
            L = b["input_ids"].shape[1]
            with torch.no_grad():
                g = model.generate(**b, max_new_tokens=mx, do_sample=False)
                if sample:
                    torch.manual_seed(1000 + i)
                    s = model.generate(**b, max_new_tokens=mx, do_sample=True, temperature=0.7, top_p=1.0, top_k=0,
                                       num_return_sequences=N_SAMPLES)
            for j, k in enumerate(idx):
                res[k] = dict(greedy=tok.decode(g[j][L:], skip_special_tokens=True).strip(),
                              samples=[tok.decode(s[j * N_SAMPLES + m][L:], skip_special_tokens=True).strip()
                                       for m in range(N_SAMPLES)] if sample else [])
            print(tag, min(i + 8, len(order)), "/", len(order), flush=True)
        return res

    p1 = sorted({x["prompt1"] for x in items})
    d = dict(zip(p1, gen(p1, False, "stage1")))
    drafts = {}
    for x in items:
        t = d[x["prompt1"]]["greedy"]
        w = [a.upper() for a in re.findall(r"[A-Za-z]+", t) if a.upper() in x["labels"]]
        drafts[x["prompt1"]] = w[0] if w else t
    r2 = gen([x["prompt"].replace("{draft}", drafts[x["prompt1"]]) for x in items], True, "stage2", max(x.get("max_new", 20) for x in items))
    out = [dict(set=x["set"], arm=x["arm"], probe_ids=[x["probes"][0]["id"]], draft=drafts[x["prompt1"]],
                draft_raw=d[x["prompt1"]]["greedy"], **r) for x, r in zip(items, r2)]
    return {"model": model_name, "answers": out}


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-14B-Instruct", sets: str = "sets_twopass.json", out: str = "results_twopass"):
    items = json.load(open(os.path.join(HERE, sets)))
    os.makedirs(os.path.join(HERE, out), exist_ok=True)
    res = run.remote(model, items)
    json.dump(res, open(os.path.join(HERE, out, model.split("/")[-1] + ".json"), "w"), indent=1)
    print("done", model, flush=True)
