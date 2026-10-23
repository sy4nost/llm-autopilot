"""'Fu / good' preference training (DPO with LoRA) on Qwen2.5-7B-Instruct, plus before/after tests.
  test:  test.json candidates scored by summed log-prob of each candidate answer (base, then tuned)
  train: train.json pairs good=key-answer vs fu-answer, DPO loss (beta 0.1), LoRA r16 on all linear layers
  bench: MMLU (10 per subject, 5-shot, letter log-prob) and GSM8K (first N, 4-shot CoT greedy 400 tokens)
  modal run modal_fu.py --train-n 0 (all) --mmlu-per 10 --gsm 250 --tag ""
"""
import json, os, re
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate", "datasets", "peft")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-fu-training", image=image)
L4 = "ABCD"


def mmlu_q(r, ans=True):
    s = r["question"].strip() + "\n" + "\n".join(f"{L4[i]}. {c}" for i, c in enumerate(r["choices"])) + "\nAnswer:"
    return s + (f" {L4[r['answer']]}\n\n" if ans else "")


@app.function(gpu="L40S", volumes={"/cache": cache}, timeout=4 * 3600)
def run(model_name: str, train: list, test: list, mmlu_per: int, gsm_n: int, tag: str, lr: float, beta: float,
        base_bench: bool = True):
    import math, random, time
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda").eval()
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    END = "<|im_end|>"

    def seq_logp(m, prompt_ids, resp_ids_list, grad=False):
        """summed log-prob of each response after the same prompt (right-padded batch)"""
        seqs = [prompt_ids + r for r in resp_ids_list]
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        ids, att = ids.cuda(), att.cuda()
        P = len(prompt_ids)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            lg = m(input_ids=ids, attention_mask=att, logits_to_keep=T - P + 1).logits  # positions P-1 .. T-1
            lp = torch.log_softmax(lg[:, :-1].float(), -1)
            tgt = ids[:, P:]
            g = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            mask = torch.zeros_like(g)
            for i, r in enumerate(resp_ids_list):
                mask[i, :len(r)] = 1
            return (g * mask).sum(-1)

    def eval_test(m, name):
        for n, it in enumerate(test):
            pid = tok(chat(it["prompt"]), add_special_tokens=False)["input_ids"]
            rs = [tok(c, add_special_tokens=False)["input_ids"] for c in it["cands"]]
            s = seq_logp(m, pid, rs).tolist()
            yield json.dumps(dict(stage="test", model=name, i=n, fam=it["fam"], kind=it["kind"], key=it["key"],
                                  old=it["old"], cands=it["cands"], scores=[round(x, 3) for x in s]))
            if (n + 1) % 200 == 0:
                print("test", name, n + 1, "/", len(test), flush=True)

    def bench(m, name):
        if mmlu_per:
            tst = load_dataset("cais/mmlu", "all", split="test")
            dev = load_dataset("cais/mmlu", "all", split="dev")
            shots, picked = {}, {}
            for r in dev:
                shots.setdefault(r["subject"], []).append(r)
            for r in tst:
                if len(picked.setdefault(r["subject"], [])) < mmlu_per:
                    picked[r["subject"]].append(r)
            qs = [r for s in sorted(picked) for r in picked[s]]
            letters = [tok.encode(c, add_special_tokens=False)[0] for c in L4]
            for n, r in enumerate(qs):
                subj = r["subject"].replace("_", " ")
                intro = f"The following are multiple choice questions about {subj}. Answer with the letter only.\n\n"
                text = chat(intro + "".join(mmlu_q(x) for x in shots[r["subject"]][:5]) + "QUESTION:\n" + mmlu_q(r, ans=False))
                ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
                with torch.no_grad():
                    lg = m(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
                v = lg[letters].tolist()
                yield json.dumps(dict(stage="mmlu", model=name, subject=r["subject"], gold=L4[r["answer"]],
                                      pred=L4[max(range(4), key=lambda i: v[i])]))
                if (n + 1) % 190 == 0:
                    print("mmlu", name, n + 1, "/", len(qs), flush=True)
        if gsm_n:
            gtest = load_dataset("openai/gsm8k", "main", split="test").select(range(gsm_n))
            gtrain = load_dataset("openai/gsm8k", "main", split="train").select(range(4))
            intro = "Solve the math problem. Think step by step and end with a line '#### <number>'.\n\n"
            shots_txt = "".join(f"Question: {x['question']}\nAnswer: {x['answer']}\n\n" for x in gtrain)
            num = lambda s: (re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", s) or re.findall(r"-?[\d,]*\.?\d+", s) or [None])[-1]
            clean = lambda v: None if v is None else v.replace(",", "").rstrip(".")
            texts = [chat(intro + shots_txt + "NEW PROBLEM:\n" + f"Question: {x['question']}\nAnswer:") for x in gtest]
            for i in range(0, len(texts), 16):
                b = tok(texts[i:i + 16], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    g = m.generate(**b, max_new_tokens=400, do_sample=False)
                for j, x in enumerate(g):
                    out = tok.decode(x[b["input_ids"].shape[1]:], skip_special_tokens=True)
                    gold = clean(gtest[i + j]["answer"].split("####")[-1].strip())
                    yield json.dumps(dict(stage="gsm8k", model=name, gold=gold, pred=clean(num(out)), text=out[-200:]))
                print("gsm8k", name, min(i + 16, len(texts)), "/", len(texts), flush=True)

    # ---- base
    yield from eval_test(model, "base")
    if base_bench:
        yield from bench(model, "base")
    if not train:
        return
    # ---- data
    pairs = []
    for it in train:
        pid = tok(chat(it["prompt"]), add_special_tokens=False)["input_ids"]
        good = tok(it["key"] + END, add_special_tokens=False)["input_ids"]
        bad = tok(it["fu"] + END, add_special_tokens=False)["input_ids"]
        pairs.append((pid, good, bad))
    ref = []
    for n, (pid, good, bad) in enumerate(pairs):
        ref.append(seq_logp(model, pid, [good, bad]).tolist())
    print("ref log-probs done", len(ref), flush=True)
    # ---- LoRA DPO
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    pm = get_peft_model(model, cfg)
    pm.train()
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    ACC = 8
    steps = math.ceil(len(pairs) / ACC)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / steps))
    order = list(range(len(pairs)))
    random.Random(0).shuffle(order)
    run_l, run_a, t0 = [], [], time.time()
    for k, idx in enumerate(order):
        pid, good, bad = pairs[idx]
        lp = seq_logp(pm, pid, [good, bad], grad=True)
        rg, rb = ref[idx]
        z = beta * ((lp[0] - rg) - (lp[1] - rb))
        loss = -F.logsigmoid(z) / ACC
        loss.backward()
        run_l.append(loss.item() * ACC)
        run_a.append(float(z.item() > 0))
        if (k + 1) % ACC == 0 or k + 1 == len(order):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        if (k + 1) % 100 == 0:
            print(f"train {k + 1} / {len(order)} loss {sum(run_l[-100:]) / 100:.3f} "
                  f"pref-acc {sum(run_a[-100:]) / 100:.2f} {time.time() - t0:.0f}s", flush=True)
            yield json.dumps(dict(stage="train", step=k + 1, loss=sum(run_l[-100:]) / 100, acc=sum(run_a[-100:]) / 100))
    pm.save_pretrained(f"/cache/fu/adapter{tag}")
    cache.commit()
    pm.eval()
    m2 = pm.merge_and_unload()
    m2.config.use_cache = True
    yield from eval_test(m2, "tuned")
    yield from bench(m2, "tuned")


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-7B-Instruct", train_n: int = 0, test_n: int = 0, mmlu_per: int = 10,
         gsm: int = 250, tag: str = "", lr: float = 5e-5, beta: float = 0.1, train_file: str = "train.json",
         test_file: str = "test.json", base_bench: bool = True):
    train = json.load(open(os.path.join(HERE, train_file)))
    test = json.load(open(os.path.join(HERE, test_file)))
    if train_n:
        train = train[:train_n] if train_n > 0 else []  # -1 = test only, no training
    if test_n:
        test = test[::max(1, len(test) // test_n)]
    out = os.path.join(HERE, "fu_" + model.split("/")[-1] + tag + ".jsonl")
    with open(out, "w") as f:
        for line in run.remote_gen(model, train, test, mmlu_per, gsm, tag, lr, beta, base_bench):
            f.write(line + "\n")
            f.flush()
    print("wrote", out, flush=True)
