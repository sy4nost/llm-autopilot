"""Watchman (frozen, as in ../watch/modal_watch.py) on public tests made by others (PREREG.md).
  hindsight-neglect-10shot  (inverse-scaling, all 315)       10 solved examples, then a case where outcome and expected value disagree
  repetitive-algebra        (inverse-scaling round 2, all 1000, HF mirror Albertmade/repetitive-algebra)
  mmlu                      (cais/mmlu, 500 test questions drawn with seed 185, 5 dev examples of the same subject)
Each prompt is split into: header, solved examples (question part + answer), test question. Re-assembly must equal the original.
Watchman: for each example, the model answers 'header + that example's question' with no other examples, choosing among the
test classes and all recorded example answers by summed log-prob (scored like the test); mismatch with the record = stale. Brake: stale example blocks silenced in the 40 copy heads b40 from the test question on.
Scores: summed log-prob of each class after the chat-formatted prompt (as in ../public/modal_public.py). normal | watch_b40 | watch_all
  modal run modal_pub_watch.py [--n 5] [--tag _smoke]   -> pubwatch_<model><tag>.jsonl
"""
import json, os, re
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate", "datasets")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-pubwatch14", image=image)


def split_hindsight(p):
    blocks = re.findall(r"Question: .*?\nAnswer: \S+\n", p, flags=re.S)
    test = p[sum(len(b) for b in blocks):]
    assert "".join(blocks) + test == p and test.startswith("Question: ") and test.rstrip().endswith("Answer:")
    ex = [(b[:b.rindex("\nAnswer: ") + len("\nAnswer:")], b[b.rindex("\nAnswer: ") + len("\nAnswer: "):-1]) for b in blocks]
    return "", blocks, ex, test


def split_algebra(p):
    lines = p.split("\n")
    header = lines[0] + "\n"
    blocks = [l + "\n" for l in lines[1:-1]]
    test = lines[-1]
    assert header + "".join(blocks) + test == p and all(" A: " in b for b in blocks) and test.endswith(" A:")
    ex = [(b[:b.rindex(" A: ") + len(" A:")], b[b.rindex(" A: ") + len(" A: "):-1]) for b in blocks]
    return header, blocks, ex, test


def mmlu_items(n, seed):
    import random
    from datasets import load_dataset
    test = load_dataset("cais/mmlu", "all", split="test")
    dev = load_dataset("cais/mmlu", "all", split="dev")
    shots = {}
    for r in dev:
        shots.setdefault(r["subject"], []).append(r)
    fmt = lambda r: r["question"].strip() + "\n" + "".join(f"{L}. {c}\n" for L, c in zip("ABCD", r["choices"])) + "Answer:"
    idx = list(range(len(test)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx[:n]:
        r = test[i]
        header = f"The following are multiple choice questions (with answers) about {r['subject'].replace('_', ' ')}.\n\n"
        ex = [(fmt(s), "ABCD"[s["answer"]]) for s in shots[r["subject"]][:5]]
        blocks = [q + " " + a + "\n\n" for q, a in ex]
        tq = fmt(r)
        out.append(dict(task="mmlu", prompt=header + "".join(blocks) + tq, header=header, blocks=blocks, ex=ex, test=tq,
                        classes=[" A", " B", " C", " D"], gold=r["answer"]))
    return out


@app.function(gpu="H100", volumes={"/cache": cache}, timeout=4 * 3600)
def run(model_name: str, b40: list, n: int):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    items = []
    for name, src, split in (("hindsight", "inverse-scaling/hindsight-neglect-10shot", split_hindsight),
                             ("algebra", "Albertmade/repetitive-algebra", split_algebra)):
        ds = load_dataset(src, split="train")
        for r in ds:
            p = r["prompt"].replace("\r\n", "\n")
            h, bl, ex, t = split(p)
            items.append(dict(task=name, prompt=p, header=h, blocks=bl, ex=ex, test=t, classes=r["classes"], gold=r["answer_index"]))
    items += mmlu_items(500, 185)
    if n:
        items = [x for t in ("hindsight", "algebra", "mmlu") for x in [y for y in items if y["task"] == t][:n]]
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", dtype=torch.bfloat16).to("cuda").eval()
    H, NL = model.config.num_attention_heads, model.config.num_hidden_layers
    heads = {"b40": {}, "all": {l: list(range(H)) for l in range(NL)}}
    for l, h in b40:
        heads["b40"].setdefault(l, []).append(h)
    cur = {"hs": None, "spans": [], "q": 0}

    def mk(li):
        def pre(mod, args, kwargs):
            if not cur["spans"] or li not in heads[cur["hs"]]:
                return args, kwargs
            m = kwargs["attention_mask"]
            b, _, q, k = m.shape
            if m.dtype == torch.bool:
                m = torch.zeros(m.shape, dtype=model.dtype, device=m.device).masked_fill(~m, torch.finfo(model.dtype).min)
            m = m.expand(b, H, q, k).clone()
            hs = torch.tensor(heads[cur["hs"]][li], device=m.device)
            for a, e in cur["spans"]:
                m[0, hs, cur["q"]:, a:e] = torch.finfo(m.dtype).min
            kwargs["attention_mask"] = m
            return args, kwargs
        return pre
    for li, layer in enumerate(model.model.layers):
        layer.register_forward_pre_hook(mk(li), with_kwargs=True)
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    memo = {}

    def fresh(text, cands):
        """the candidate answer with the highest summed log-prob after the history-free prompt (scored like the test)"""
        key = (text, tuple(cands))
        if key not in memo:
            pid0 = tok(chat(text), add_special_tokens=False)["input_ids"]
            sc = []
            for c in cands:
                ids_c = tok(c, add_special_tokens=False)["input_ids"]
                ids = torch.tensor([pid0 + ids_c]).cuda()
                with torch.no_grad():
                    lg = model(input_ids=ids, logits_to_keep=len(ids_c) + 1).logits[0, :-1].float()
                sc.append(torch.log_softmax(lg, -1).gather(-1, torch.tensor(ids_c).cuda().unsqueeze(-1)).sum().item())
            memo[key] = cands[max(range(len(cands)), key=lambda j: sc[j])]
        return memo[key]
    checks = dict(n_zero_flag=0, max_zero_flag_diff=0.0, shown={})
    for nn, it in enumerate(items):
        flags = []
        cands = sorted({c.strip() for c in it["classes"]} | {a.strip() for _, a in it["ex"]})
        for q, a in it["ex"]:
            flags.append(fresh(it["header"] + q, cands) != a.strip())
        text = chat(it["prompt"])
        off = text.index(it["prompt"])
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        offs = enc["offset_mapping"]
        pos = lambda c: next(j for j, o in enumerate(offs) if o[0] >= c and o[1] > o[0])
        starts, c0 = [], off + len(it["header"])
        for b in it["blocks"]:
            starts.append((c0, c0 + len(b))); c0 += len(b)
        qpos = pos(c0)
        spans = [(pos(a), pos(e)) for (a, e), f in zip(starts, flags) if f]
        if it["task"] not in checks["shown"]:
            ids0 = enc["input_ids"]
            q0, a0 = it["ex"][0]
            checks["shown"][it["task"]] = dict(recheck=it["header"] + q0, ans=a0, cands=cands, got=fresh(it["header"] + q0, cands),
                                               block0=tok.decode(ids0[pos(starts[0][0]):pos(starts[0][1])]), query=tok.decode(ids0[qpos:qpos + 6]))
        pid = enc["input_ids"]
        res = {}
        for name, hs, sp in (("normal", None, []), ("watch_b40", "b40", spans), ("watch_all", "all", spans)):
            cur.update(hs=hs, spans=sp, q=qpos)
            sc = []
            for c in it["classes"]:
                ids_c = tok(c.strip(), add_special_tokens=False)["input_ids"]
                ids = torch.tensor([pid + ids_c]).cuda()
                with torch.no_grad():
                    lg = model(input_ids=ids, logits_to_keep=len(ids_c) + 1).logits[0, :-1].float()
                sc.append(round(torch.log_softmax(lg, -1).gather(-1, torch.tensor(ids_c).cuda().unsqueeze(-1)).sum().item(), 3))
            res[name] = sc
        cur["spans"] = []
        if not spans:
            checks["n_zero_flag"] += 1
            checks["max_zero_flag_diff"] = max(checks["max_zero_flag_diff"], max(abs(x - y) for x, y in zip(res["watch_b40"], res["normal"])))
        yield json.dumps(dict(task=it["task"], i=nn, gold=it["gold"], flags=flags, scores=res))
        if (nn + 1) % 50 == 0:
            print("pubwatch", nn + 1, "/", len(items), flush=True)
    yield json.dumps(dict(checks=checks))


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-14B-Instruct", n: int = 0, tag: str = ""):
    b40 = json.load(open(os.path.join(HERE, "..", "eras", "settings2.json")))["b40"]
    out = os.path.join(HERE, "pubwatch_" + model.split("/")[-1] + tag + ".jsonl")
    with open(out, "w") as f:
        for line in run.remote_gen(model, b40, n):
            f.write(line + "\n")
            f.flush()
    print("wrote", out, flush=True)
