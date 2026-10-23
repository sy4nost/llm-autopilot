"""Find the 'copying' attention heads in 14B and switch off only them.
Prompts: match/sets_match.json, N=60: {A,B,C}_change_60 D1-D4, {A,B,C}_nochange_60 D1, D3 (540).
Stage 1 (all prompts, normal run): for every head (48 layers x 40), attention from the answer position (last token)
  to (a) the label tokens of examples IDENTICAL to the test case ('exact'), (b) the label tokens of all examples
  ('labels'). Plus final first-label-token log-probs.
Stage 2: rank heads by mean 'exact' attention on the DISCOVERY half (sets 0-14, A_change_60, D1/D3). On the
  HELD-OUT half (sets 15-29, all 270 prompts) switch off the top 10 / 20 / 40 heads (their output zeroed from the
  test case onward, at the o_proj input), and 20 random heads as a control. Records label log-probs per setting.
  modal run modal_heads.py --limit 0   -> heads_<model>.jsonl   (--limit 6: quick check, discovery = all)
"""
import json, os, random, re
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-rule-heads", image=image)


def prep(tok, item):
    text = tok.apply_chat_template([{"role": "user", "content": item["prompt"]}], tokenize=False,
                                   add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    offs = enc["offset_mapping"][0].tolist()
    test = re.search(r"TEST CASE: (.*)\n", text).group(1)
    first_tok = lambda c: next(k for k, o in enumerate(offs) if o[0] >= c)
    exact, labels = [], []
    for m in re.finditer(r"Example \d+: (.*?) -> (\w+)", text):
        k = first_tok(m.start(2))
        labels.append(k)
        if m.group(1) == test:
            exact.append(k)
    t_start = first_tok(text.find("Classify the TEST"))
    return enc["input_ids"], exact, labels, t_start


@app.function(gpu="H100", volumes={"/cache": cache}, timeout=4 * 3600)
def probe(model_name: str, items: list, disc_sets: list):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager",
                                                 dtype=torch.bfloat16).to("cuda").eval()
    layers = model.model.layers
    L, H = len(layers), model.config.num_attention_heads
    D = model.config.hidden_size // H
    lab_of = lambda it: [tok.encode(l, add_special_tokens=False)[0] for l in it["labels"]]
    pre = {}
    # ---- stage 1
    stats = {}
    for n, it in enumerate(items):
        ids, exact, labels, t0 = prep(tok, it)
        ids = ids.to("cuda")
        ex_t = torch.tensor(exact, dtype=torch.long, device="cuda")
        lb_t = torch.tensor(labels, dtype=torch.long, device="cuda")
        em = torch.zeros(L, H)
        lm = torch.zeros(L, H)
        hooks = []
        for li, layer in enumerate(layers):
            def hk(mod, inp, out, li=li):
                row = out[1][0, :, -1, :].float()
                if len(exact):
                    em[li] = row[:, ex_t].sum(-1).cpu()
                if len(labels):
                    lm[li] = row[:, lb_t].sum(-1).cpu()
            hooks.append(layer.self_attn.register_forward_hook(hk))
        with torch.no_grad():
            o = model(input_ids=ids, logits_to_keep=1)
        for h in hooks:
            h.remove()
        lp = torch.log_softmax(o.logits[0, -1].float(), -1)[lab_of(it)].tolist()
        pr = it["probes"][0]
        stats[n] = (em, lm)
        pre[n] = (ids, t0)
        yield json.dumps(dict(stage=1, set=it["set"], arm=it["arm"], probe=pr["id"], labels=it["labels"], key=pr["key"],
                              old=pr["old"], n_exact=len(exact), final=[round(x, 3) for x in lp],
                              exact=[[round(float(x), 5) for x in r] for r in em],
                              lab=[[round(float(x), 5) for x in r] for r in lm]))
        print("stage1", n + 1, "/", len(items), flush=True)
        pre[n] = None
    # ---- rank heads on discovery prompts
    disc = [n for n, it in enumerate(items) if it["set"] in disc_sets and it["arm"] == "A_change_60"
            and it["probes"][0]["id"] in ("D1", "D3")]
    score = torch.stack([stats[n][0] for n in disc]).mean(0)
    order = torch.argsort(score.flatten(), descending=True).tolist()
    rank = [(i // H, i % H) for i in order]
    rnd = random.Random(0).sample(rank[40:], 20)
    sets_ = {"top10": rank[:10], "top20": rank[:20], "top40": rank[:40], "rand20": rnd}
    yield json.dumps(dict(stage="rank", top=[(l, h, round(float(score[l, h]), 4)) for l, h in rank[:60]], rand20=rnd,
                          disc_prompts=len(disc)))
    # ---- stage 2: held-out
    held = [n for n, it in enumerate(items) if it["set"] not in disc_sets] or list(range(len(items)))
    cur = {}

    def mk(li):
        def p(mod, args):
            heads = cur.get(li)
            if not heads:
                return None
            x = args[0].clone()
            for h in heads:
                x[:, cur["t0"]:, h * D:(h + 1) * D] = 0
            return (x,) + args[1:]
        return p
    hooks = [layer.self_attn.o_proj.register_forward_pre_hook(mk(li)) for li, layer in enumerate(layers)]
    for k, n in enumerate(held):
        it = items[n]
        ids, exact, labels, t0 = prep(tok, it)
        ids = ids.to("cuda")
        res = {}
        for name, hs in sets_.items():
            cur.clear()
            cur["t0"] = t0
            for l, h in hs:
                cur.setdefault(l, []).append(h)
            with torch.no_grad():
                o = model(input_ids=ids, logits_to_keep=1)
            res[name] = [round(x, 3) for x in torch.log_softmax(o.logits[0, -1].float(), -1)[lab_of(it)].tolist()]
        cur.clear()
        pr = it["probes"][0]
        yield json.dumps(dict(stage=2, set=it["set"], arm=it["arm"], probe=pr["id"], labels=it["labels"], key=pr["key"],
                              old=pr["old"], ko=res))
        print("stage2", k + 1, "/", len(held), flush=True)
    for h in hooks:
        h.remove()


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-14B-Instruct", limit: int = 6):
    items = [x for x in json.load(open(os.path.join(HERE, "..", "match", "sets_match.json")))
             if x["arm"] in ("A_change_60", "B_change_60", "C_change_60")
             or (x["arm"] in ("A_nochange_60", "B_nochange_60", "C_nochange_60") and x["probes"][0]["id"] in ("D1", "D3"))]
    disc = list(range(15))
    if limit:
        items = [x for x in items if x["set"] == 0 and x["probes"][0]["id"] in ("D1", "D3")][:limit]
        disc = [0]
    out = os.path.join(HERE, "heads_" + model.split("/")[-1] + ("" if not limit else "_test") + ".jsonl")
    with open(out, "w") as f:
        for line in probe.remote_gen(model, items, disc):
            f.write(line + "\n")
            f.flush()
    print("wrote", out, flush=True)
