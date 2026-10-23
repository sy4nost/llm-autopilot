"""Selective version: the 20 'copy' heads (top20.json) may not attend to the block of PAST SOLVED EXAMPLES, from the
current task onward; they still see the rules, the current task and the model's own reasoning. All other heads
see everything. Done with a per-head attention mask in the copy heads' layers (eager attention).
Benchmarks, each with settings normal and blockex:
  ours:  match/sets_match.json held-out sets 15-29 (A/B/C change D1-D4, no-change D1/D3), 270 prompts, label log-probs
  mmlu:  570 questions (10 per subject), 5-shot; examples block = the 5 solved questions
  gsm8k: 250 problems, 4 solved problems in the prompt (= examples block), greedy CoT up to 400 tokens
  modal run modal_blockex.py --ours 0 --mmlu-per 10 --gsm 250     (small numbers for a check)
"""
import json, os, re
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate", "datasets")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-rule-blockex", image=image)
L4 = "ABCD"


def mmlu_q(r, ans=True):
    s = r["question"].strip() + "\n" + "\n".join(f"{L4[i]}. {c}" for i, c in enumerate(r["choices"])) + "\nAnswer:"
    return s + (f" {L4[r['answer']]}\n\n" if ans else "")


@app.function(gpu="H100", volumes={"/cache": cache}, timeout=5 * 3600)
def run(model_name: str, heads: list, ours: list, mmlu_per: int, gsm_n: int):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager",
                                                 dtype=torch.bfloat16).to("cuda").eval()
    H = model.config.num_attention_heads
    by = {}
    for l, h in heads:
        by.setdefault(l, []).append(h)
    cur = {"on": False}  # rows: list of (q_start, ex_a, ex_b) in padded coordinates

    def mk(li):
        def pre(mod, args, kwargs):
            if not cur["on"] or li not in by:
                return args, kwargs
            m = kwargs.get("attention_mask")
            if m is None:
                raise RuntimeError("no attention mask passed to the layer")
            b, _, q, k = m.shape
            m = m.expand(b, H, q, k).clone()
            hs = torch.tensor(by[li], device=m.device)
            for r, (qs, a, e) in enumerate(cur["rows"]):
                q0 = 0 if q == 1 else qs
                if m.dtype == torch.bool:
                    m[r, hs, q0:, a:e] = False
                else:
                    m[r, hs, q0:, a:e] = torch.finfo(m.dtype).min
            kwargs["attention_mask"] = m
            return args, kwargs
        return pre
    hooks = [layer.register_forward_pre_hook(mk(li), with_kwargs=True) for li, layer in enumerate(model.model.layers)]
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)

    def tok_pos(offs, c):
        return next(k for k, o in enumerate(offs) if o[0] >= c and o[1] > o[0])

    def one(text, ex_start_str, ex_end_str, q_str, lab_ids):
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
        offs = enc["offset_mapping"][0].tolist()
        a = text.find(ex_start_str)
        e = text.find(ex_end_str, a)
        qc = text.find(q_str, e)
        row = (tok_pos(offs, qc), tok_pos(offs, a), tok_pos(offs, e))
        ids = enc["input_ids"].to("cuda")
        res = {}
        for name, on in (("normal", False), ("blockex", True)):
            cur.update(on=on, rows=[row])
            with torch.no_grad():
                lg = model(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
            res[name] = torch.log_softmax(lg, -1)[lab_ids].tolist()
        cur["on"] = False
        return res
    # ---- ours
    for n, it in enumerate(ours):
        text = chat(it["prompt"])
        lab = [tok.encode(l, add_special_tokens=False)[0] for l in it["labels"]]
        end = "IMPORTANT UPDATE" if "IMPORTANT UPDATE" in text else "Classify the TEST"
        res = one(text, "Below are", end, "Classify the TEST", lab)
        pr = it["probes"][0]
        yield json.dumps(dict(bench="ours", set=it["set"], arm=it["arm"], probe=pr["id"], labels=it["labels"],
                              key=pr["key"], old=pr["old"], scores={k: [round(x, 3) for x in v] for k, v in res.items()}))
        print("ours", n + 1, "/", len(ours), flush=True)
    # ---- MMLU
    if mmlu_per:
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
            intro = f"The following are multiple choice questions about {subj}. Answer with the letter only.\n\n"
            text = chat(intro + "".join(mmlu_q(x) for x in shots[r["subject"]][:5]) + "QUESTION:\n" + mmlu_q(r, ans=False))
            ex0 = shots[r["subject"]][0]["question"].strip()[:40]
            res = one(text, ex0, "QUESTION:\n", "QUESTION:\n", letters)
            yield json.dumps(dict(bench="mmlu", subject=r["subject"], gold=L4[r["answer"]],
                                  pred={k: L4[max(range(4), key=lambda i: v[i])] for k, v in res.items()}))
            print("mmlu", n + 1, "/", len(qs), flush=True)
    # ---- GSM8K
    if gsm_n:
        gtest = load_dataset("openai/gsm8k", "main", split="test").select(range(gsm_n))
        gtrain = load_dataset("openai/gsm8k", "main", split="train").select(range(4))
        intro = "Solve the math problem. Think step by step and end with a line '#### <number>'.\n\n"
        shots_txt = "".join(f"Question: {x['question']}\nAnswer: {x['answer']}\n\n" for x in gtrain)
        num = lambda s: (re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", s) or re.findall(r"-?[\d,]*\.?\d+", s) or [None])[-1]
        clean = lambda v: None if v is None else v.replace(",", "").rstrip(".")
        texts = [chat(intro + shots_txt + "NEW PROBLEM:\n" + f"Question: {x['question']}\nAnswer:") for x in gtest]
        golds = [clean(x["answer"].split("####")[-1].strip()) for x in gtest]
        outs = {}
        for name, on in (("normal", False), ("blockex", True)):
            outs[name] = []
            for i in range(0, len(texts), 16):
                bt = texts[i:i + 16]
                b = tok(bt, return_tensors="pt", padding=True, return_offsets_mapping=True)
                Tm = b["input_ids"].shape[1]
                rows = []
                for j, t in enumerate(bt):
                    offs = b["offset_mapping"][j].tolist()
                    pad = int((b["attention_mask"][j] == 0).sum())
                    a = t.find("Question:")
                    e = t.find("NEW PROBLEM:")
                    find = lambda c: next(k for k in range(pad, Tm) if offs[k][0] >= c and offs[k][1] > offs[k][0])
                    rows.append((find(e), find(a), find(e)))
                cur.update(on=on, rows=rows)
                inp = {k: v.to("cuda") for k, v in b.items() if k != "offset_mapping"}
                with torch.no_grad():
                    g = model.generate(**inp, max_new_tokens=400, do_sample=False)
                cur["on"] = False
                outs[name] += [tok.decode(x[Tm:], skip_special_tokens=True) for x in g]
                print("gsm8k", name, min(i + 16, len(texts)), "/", len(texts), flush=True)
        for k in range(len(texts)):
            yield json.dumps(dict(bench="gsm8k", gold=golds[k], pred={n: clean(num(outs[n][k])) for n in outs},
                                  text={n: outs[n][k][-300:] for n in outs}))
    for h in hooks:
        h.remove()


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-14B-Instruct", ours: int = 0, mmlu_per: int = 10, gsm: int = 250, tag: str = ""):
    heads = json.load(open(os.path.join(HERE, "top20.json")))
    items = [x for x in json.load(open(os.path.join(HERE, "..", "match", "sets_match.json"))) if x["set"] >= 15 and (
        x["arm"] in ("A_change_60", "B_change_60", "C_change_60")
        or (x["arm"] in ("A_nochange_60", "B_nochange_60", "C_nochange_60") and x["probes"][0]["id"] in ("D1", "D3")))]
    if ours:
        items = items[:ours]
    out = os.path.join(HERE, "blockex_" + model.split("/")[-1] + tag + ".jsonl")
    with open(out, "w") as f:
        for line in run.remote_gen(model, heads, items, mmlu_per, gsm):
            f.write(line + "\n")
            f.flush()
    print("wrote", out, flush=True)
