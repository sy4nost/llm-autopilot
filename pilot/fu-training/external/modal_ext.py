"""External, public tests that catch models on habits / memorised patterns. Base vs fu-v1 vs fu-v2 (Qwen2.5-7B).
  inverse-scaling/redefine-math     a symbol is redefined in the prompt; memorised meaning pulls the wrong way
  inverse-scaling/quote-repetition  a famous quote is altered; must repeat the altered text, not the memorised one
  inverse-scaling/NeQA              negated questions ('not'); pattern ignores the negation
  inverse-scaling/hindsight-neglect-10shot  examples where outcome = expected value, then a case where they differ
  apple/GSM-Symbolic main / p1 / p2 GSM8K problems with changed numbers and 1-2 extra clauses (generation)
Multiple-choice tasks: summed log-prob of each class as the assistant reply (chat template).
  modal run modal_ext.py --n-mc 300 --n-gsm 100
"""
import json, os, re
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate", "datasets", "peft")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-fu-external", image=image)
MC = ["redefine-math", "quote-repetition", "NeQA", "hindsight-neglect-10shot"]


@app.function(gpu="L40S", volumes={"/cache": cache}, timeout=4 * 3600)
def run(model_name: str, variants: list, n_mc: int, n_gsm: int):
    import random
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    mc = {}
    for name in MC:
        ds = load_dataset("inverse-scaling/" + name, split="train")
        idx = list(range(len(ds)))
        random.Random(0).shuffle(idx)
        mc[name] = [ds[i] for i in sorted(idx[:n_mc])]
    gtrain = load_dataset("openai/gsm8k", "main", split="train").select(range(4))
    intro = "Solve the math problem. Think step by step and end with a line '#### <number>'.\n\n"
    shots = "".join(f"Question: {x['question']}\nAnswer: {x['answer']}\n\n" for x in gtrain)
    gsm = {}
    for cfg in ("main", "p1", "p2"):
        ds = load_dataset("apple/GSM-Symbolic", cfg, split="test")
        seen, rows = set(), []
        for r in ds:  # one instance per template first, then a second one, until n_gsm
            if r["original_id"] not in seen:
                seen.add(r["original_id"])
                rows.append(r)
        gsm[cfg] = rows[:n_gsm]
    num = lambda s: (re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", s) or re.findall(r"-?[\d,]*\.?\d+", s) or [None])[-1]
    clean = lambda v: None if v is None else v.replace(",", "").rstrip(".").rstrip("0").rstrip(".") if "." in v else v.replace(",", "")
    for var, adapter in variants:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda").eval()
        if adapter:
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
        for name, rows in mc.items():
            for n, r in enumerate(rows):
                pid = tok(chat(r["prompt"].replace("\r\n", "\n")), add_special_tokens=False)["input_ids"]
                sc = []
                for c in r["classes"]:
                    ids_c = tok(c.strip(), add_special_tokens=False)["input_ids"]
                    ids = torch.tensor([pid + ids_c]).cuda()
                    with torch.no_grad():
                        lg = model(input_ids=ids, logits_to_keep=len(ids_c) + 1).logits[0, :-1].float()
                    sc.append(torch.log_softmax(lg, -1).gather(-1, torch.tensor(ids_c).cuda().unsqueeze(-1)).sum().item())
                yield json.dumps(dict(model=var, task=name, i=n, gold=r["answer_index"], scores=[round(x, 3) for x in sc]))
            print(var, name, len(rows), flush=True)
        for cfg, rows in gsm.items():
            texts = [chat(intro + shots + "NEW PROBLEM:\n" + f"Question: {r['question']}\nAnswer:") for r in rows]
            for i in range(0, len(texts), 16):
                b = tok(texts[i:i + 16], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    g = model.generate(**b, max_new_tokens=450, do_sample=False)
                for j, x in enumerate(g):
                    out = tok.decode(x[b["input_ids"].shape[1]:], skip_special_tokens=True)
                    gold = clean(rows[i + j]["answer"].split("####")[-1].strip())
                    yield json.dumps(dict(model=var, task="gsm-" + cfg, i=i + j, gold=gold, pred=clean(num(out)), text=out[-200:]))
            print(var, "gsm", cfg, len(rows), flush=True)
        del model
        torch.cuda.empty_cache()


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-7B-Instruct", n_mc: int = 300, n_gsm: int = 100, tag: str = "",
         variants: str = "base:,v1:/cache/fu/adapter,v2:/cache/fu/adapter_v2"):
    variants = [v.split(":", 1) for v in variants.split(",")]
    out = os.path.join(HERE, "ext_" + model.split("/")[-1] + tag + ".jsonl")
    with open(out, "w") as f:
        for line in run.remote_gen(model, variants, n_mc, n_gsm):
            f.write(line + "\n")
            f.flush()
    print("wrote", out, flush=True)
