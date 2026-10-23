"""Public tests on Qwen2.5-14B-Instruct: normal vs the 20 'copy' heads switched off (top20.json, layers 29-38).
  MC (inverse-scaling redefine-math, quote-repetition, NeQA, hindsight-neglect-10shot): summed log-prob of each class.
    normal | off_all (heads zeroed everywhere) | off_ans (zeroed only at the answer positions, from the last prompt token)
  GSM-Symbolic main / p1: 4-shot, greedy, 450 tokens.  normal | off_all
  modal run modal_public.py --n-mc 200 --n-gsm 50   -> public_<model>.jsonl
"""
import json, os, re
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate", "datasets")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-public14", image=image)
MC = ["redefine-math", "quote-repetition", "NeQA", "hindsight-neglect-10shot"]


@app.function(gpu="H100", volumes={"/cache": cache}, timeout=3 * 3600)
def run(model_name: str, heads: list, n_mc: int, n_gsm: int):
    import random, torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda").eval()
    H = model.config.num_attention_heads
    D = model.config.hidden_size // H
    by = {}
    for l, h in heads:
        by.setdefault(l, []).append(h)
    cur = {"mode": None, "t0": 0}

    def mk(li):
        def p(mod, args):
            if cur["mode"] is None or li not in by:
                return None
            x = args[0].clone()
            start = 0 if (cur["mode"] == "all" or x.shape[1] == 1) else cur["t0"]
            for h in by[li]:
                x[:, start:, h * D:(h + 1) * D] = 0
            return (x,) + args[1:]
        return p
    hooks = [layer.self_attn.o_proj.register_forward_pre_hook(mk(li)) for li, layer in enumerate(model.model.layers)]
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    for name in MC:
        ds = load_dataset("inverse-scaling/" + name, split="train")
        idx = list(range(len(ds)))
        random.Random(0).shuffle(idx)
        rows = [ds[i] for i in sorted(idx[:n_mc])]
        for n, r in enumerate(rows):
            pid = tok(chat(r["prompt"].replace("\r\n", "\n")), add_special_tokens=False)["input_ids"]
            out = {}
            for mode, key in ((None, "normal"), ("all", "off_all"), ("ans", "off_ans")):
                cur["mode"], cur["t0"] = mode, len(pid) - 1
                sc = []
                for c in r["classes"]:
                    ids_c = tok(c.strip(), add_special_tokens=False)["input_ids"]
                    ids = torch.tensor([pid + ids_c]).cuda()
                    with torch.no_grad():
                        lg = model(input_ids=ids, logits_to_keep=len(ids_c) + 1).logits[0, :-1].float()
                    sc.append(round(torch.log_softmax(lg, -1).gather(-1, torch.tensor(ids_c).cuda().unsqueeze(-1)).sum().item(), 3))
                out[key] = sc
            cur["mode"] = None
            yield json.dumps(dict(task=name, i=n, gold=r["answer_index"], **out))
            if (n + 1) % 50 == 0:
                print(name, n + 1, "/", len(rows), flush=True)
    gtrain = load_dataset("openai/gsm8k", "main", split="train").select(range(4))
    intro = "Solve the math problem. Think step by step and end with a line '#### <number>'.\n\n"
    shots = "".join(f"Question: {x['question']}\nAnswer: {x['answer']}\n\n" for x in gtrain)
    num = lambda s: (re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", s) or re.findall(r"-?[\d,]*\.?\d+", s) or [None])[-1]
    clean = lambda v: None if v is None else v.replace(",", "").rstrip(".").rstrip("0").rstrip(".") if "." in v else v.replace(",", "")
    for cfg in ("main", "p1"):
        ds = load_dataset("apple/GSM-Symbolic", cfg, split="test")
        seen, rows = set(), []
        for r in ds:
            if r["original_id"] not in seen:
                seen.add(r["original_id"])
                rows.append(r)
        rows = rows[:n_gsm]
        texts = [chat(intro + shots + "NEW PROBLEM:\n" + f"Question: {r['question']}\nAnswer:") for r in rows]
        for mode, key in ((None, "normal"), ("all", "off_all")):
            cur["mode"] = mode
            for i in range(0, len(texts), 16):
                b = tok(texts[i:i + 16], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    g = model.generate(**b, max_new_tokens=450, do_sample=False)
                for j, x in enumerate(g):
                    o = tok.decode(x[b["input_ids"].shape[1]:], skip_special_tokens=True)
                    yield json.dumps(dict(task="gsm-" + cfg, mode=key, i=i + j, gold=clean(rows[i + j]["answer"].split("####")[-1].strip()), pred=clean(num(o)), text=o[-200:]))
            print("gsm", cfg, key, len(rows), flush=True)
        cur["mode"] = None
    for h in hooks:
        h.remove()


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-14B-Instruct", n_mc: int = 200, n_gsm: int = 50, tag: str = ""):
    heads = json.load(open(os.path.join(HERE, "top20.json")))
    out = os.path.join(HERE, "public_" + model.split("/")[-1] + tag + ".jsonl")
    with open(out, "w") as f:
        for line in run.remote_gen(model, heads, n_mc, n_gsm):
            f.write(line + "\n")
            f.flush()
    print("wrote", out, flush=True)
