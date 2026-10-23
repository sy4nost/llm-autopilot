"""No-regression check: Qwen2.5-14B-Instruct normal vs the 20 'copy' heads switched off (top20.json, from heads/).
MMLU: 10 test questions per subject (57 subjects, 570), 5-shot from the dev split, answer = highest log-prob of the
  first token of A/B/C/D. Settings: normal; off_all (heads zeroed at every position); off_q (zeroed from the test
  question onward, like in our task).
GSM8K: first N test problems, 4 fixed train examples in the prompt, greedy, up to 400 new tokens, answer after '####'.
  Settings: normal; off_all.
  modal run modal_bench.py --mmlu-per 10 --gsm 250   -> bench_<model>.jsonl
"""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate", "datasets")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-rule-bench", image=image)
L4 = "ABCD"


def mmlu_q(r, ans=True):
    s = r["question"].strip() + "\n" + "\n".join(f"{L4[i]}. {c}" for i, c in enumerate(r["choices"])) + "\nAnswer:"
    return s + (f" {L4[r['answer']]}\n\n" if ans else "")


@app.function(gpu="H100", volumes={"/cache": cache}, timeout=4 * 3600)
def run(model_name: str, heads: list, mmlu_per: int, gsm_n: int):
    import re, torch
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
    cur = {"mode": None}

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
    # ---- MMLU
    test = load_dataset("cais/mmlu", "all", split="test")
    dev = load_dataset("cais/mmlu", "all", split="dev")
    shots, picked = {}, {}
    for r in dev:
        shots.setdefault(r["subject"], []).append(r)
    for r in test:
        if len(picked.setdefault(r["subject"], [])) < mmlu_per:
            picked[r["subject"]].append(r)
    qs = [r for s in sorted(picked) for r in picked[s]]
    letters = [tok.encode(c, add_special_tokens=False)[0] for c in L4]
    for n, r in enumerate(qs):
        subj = r["subject"].replace("_", " ")
        head_txt = (f"The following are multiple choice questions about {subj}. Answer with the letter only.\n\n"
                    + "".join(mmlu_q(x) for x in shots[r["subject"]][:5]))
        text = chat(head_txt + mmlu_q(r, ans=False))
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
        offs = enc["offset_mapping"][0].tolist()
        qpos = text.find(r["question"].strip(), text.find(head_txt[-50:]) if len(head_txt) > 50 else 0)
        t0 = next(k for k, o in enumerate(offs) if o[0] >= qpos)
        ids = enc["input_ids"].to("cuda")
        res = {}
        for name, mode in (("normal", None), ("off_all", "all"), ("off_q", "q")):
            cur.update(mode=mode, t0=t0)
            with torch.no_grad():
                lg = model(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
            res[name] = L4[int(torch.argmax(lg[letters]))]
        cur["mode"] = None
        yield json.dumps(dict(bench="mmlu", subject=r["subject"], gold=L4[r["answer"]], pred=res))
        print("mmlu", n + 1, "/", len(qs), flush=True)
    # ---- GSM8K
    gtest = load_dataset("openai/gsm8k", "main", split="test").select(range(gsm_n))
    gtrain = load_dataset("openai/gsm8k", "main", split="train").select(range(4))
    pre = ("Solve the math problem. Think step by step and end with a line '#### <number>'.\n\n"
           + "".join(f"Question: {x['question']}\nAnswer: {x['answer']}\n\n" for x in gtrain))
    num = lambda s: (re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", s) or re.findall(r"-?[\d,]*\.?\d+", s) or [None])[-1]
    clean = lambda v: None if v is None else v.replace(",", "").rstrip(".")
    prompts = [chat(pre + f"Question: {x['question']}\nAnswer:") for x in gtest]
    golds = [clean(x["answer"].split("####")[-1].strip()) for x in gtest]
    outs = {}
    for name, mode in (("normal", None), ("off_all", "all")):
        outs[name] = []
        for i in range(0, len(prompts), 16):
            b = tok(prompts[i:i + 16], return_tensors="pt", padding=True).to("cuda")
            cur.update(mode=mode, t0=0)
            with torch.no_grad():
                g = model.generate(**b, max_new_tokens=400, do_sample=False)
            outs[name] += [tok.decode(x[b["input_ids"].shape[1]:], skip_special_tokens=True) for x in g]
            print("gsm8k", name, min(i + 16, len(prompts)), "/", len(prompts), flush=True)
        cur["mode"] = None
    for k in range(len(prompts)):
        yield json.dumps(dict(bench="gsm8k", gold=golds[k], pred={n: clean(num(outs[n][k])) for n in outs},
                              text={n: outs[n][k][-300:] for n in outs}))
    for h in hooks:
        h.remove()


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-14B-Instruct", mmlu_per: int = 10, gsm: int = 250, tag: str = ""):
    heads = json.load(open(os.path.join(HERE, "top20.json")))
    out = os.path.join(HERE, "bench_" + model.split("/")[-1] + tag + ".jsonl")
    with open(out, "w") as f:
        for line in run.remote_gen(model, heads, mmlu_per, gsm):
            f.write(line + "\n")
            f.flush()
    print("wrote", out, flush=True)
