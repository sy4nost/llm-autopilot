"""'Pull dial' inside 14B (Alex: replace the pull, keep the knowledge). One forward pass per setting, no generation.
From the test case onward, attention to the solved examples is scaled by alpha (log(alpha) added to those attention
logits, i.e. the unnormalised weight is multiplied by alpha) ONLY in the late layers LATE (36..47 of 48), where the
logit lens showed the old answer taking over (late_0), only in 24..35 (mid_0), or in all layers with
alpha 0.5 / 0.25 / 0.1 / 0 (all_*); "1.0" is the dial at 100% (must equal the normal run). Records first-label-token log-probs per alpha, the logit-lens
label scores at layer 32 (what the model 'computed by the rule' before the takeover), and mask_check
(alpha=1 through the hook vs a plain run; must be ~0).
Prompts: match/sets_match.json, N=60: {A,B,C}_change_60 D1-D4 and {A,B,C}_nochange_60 D1, D3 (540).
  modal run modal_damp.py --limit 0   -> damp_<model>.jsonl     (--limit 8 for a quick check)
"""
import json, math, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-rule-damp", image=image)
ALPHAS = [1.0, 0.5, 0.25, 0.1, 0.0]
# setting name -> (layers where the dial applies, alpha); "1.0" = dial at 100% through the hook (= normal run)
SETTINGS = {"1.0": ("all", 1.0), "late_0": ("late", 0.0), "mid_0": ("mid", 0.0),
            "all_0.5": ("all", 0.5), "all_0.25": ("all", 0.25), "all_0.1": ("all", 0.1), "all_0": ("all", 0.0)}
LENS_LAYER = 32


def ex_span(text):
    i = text.find("Below are")
    if i < 0:
        return None
    ends = [j for j in (text.find("IMPORTANT UPDATE", i), text.find("Classify the TEST", i)) if j >= 0]
    return i, min(ends)


def run_one(model, tok, item, torch):
    text = tok.apply_chat_template([{"role": "user", "content": item["prompt"]}], tokenize=False,
                                   add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    ids = enc["input_ids"].to("cuda")
    offs = enc["offset_mapping"][0].tolist()
    T = ids.shape[1]
    a, b = ex_span(text)
    t0 = text.find("Classify the TEST")
    ex_cols = torch.tensor([a <= o[0] < b for o in offs], device="cuda")
    test_start = next(k for k, o in enumerate(offs) if o[0] >= t0)
    lab_ids = [tok.encode(l, add_special_tokens=False)[0] for l in item["labels"]]
    neg = torch.finfo(model.dtype).min
    base = torch.full((T, T), neg, device="cuda", dtype=model.dtype).triu(1)[None, None]
    L = len(model.model.layers)
    groups = {"late": set(range(36, L)), "mid": set(range(24, 36)), "all": set(range(L))}
    cur = {}
    hooks = []
    for li, layer in enumerate(model.model.layers):
        def pre(mod, args, kwargs, li=li):
            if "mask" in cur:
                if "attention_mask" not in kwargs:
                    raise RuntimeError("attention_mask not passed as kwarg")
                kwargs["attention_mask"] = cur["mask"] if li in cur["layers"] else base
            return args, kwargs
        hooks.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
    out = {}
    try:
        with torch.no_grad():
            plain = model(input_ids=ids, output_hidden_states=True, logits_to_keep=1)
            h = plain.hidden_states[LENS_LAYER][0, -1]
            lens = torch.log_softmax(model.get_output_embeddings()(model.model.norm(h)).float(), -1)[lab_ids]
            for name, (grp, al) in SETTINGS.items():
                m = base.clone()
                add = neg if al == 0 else math.log(al)
                if al != 1.0:
                    blk = m[0, 0, test_start:, :]
                    blk[:, ex_cols] = blk[:, ex_cols] + add if al != 0 else neg
                cur["mask"] = m
                cur["layers"] = groups[grp]
                o = model(input_ids=ids, attention_mask=base, logits_to_keep=1)
                out[name] = torch.log_softmax(o.logits[0, -1].float(), -1)
                cur.clear()
    finally:
        for hk in hooks:
            hk.remove()
    mask_check = float((out["1.0"] - torch.log_softmax(plain.logits[0, -1].float(), -1)).abs().max())
    pr = item["probes"][0]
    return dict(set=item["set"], arm=item["arm"], probe=pr["id"], labels=item["labels"], key=pr["key"], old=pr["old"],
                label_tok=[tok.decode([i]) for i in lab_ids], lens32=[round(float(x), 3) for x in lens],
                damp={k: [round(float(v[i]), 3) for i in lab_ids] for k, v in out.items()},
                mask_check=round(mask_check, 4), T=T, n_ex_tok=int(ex_cols.sum()))


@app.function(gpu="H100", volumes={"/cache": cache}, timeout=3 * 3600)
def probe(model_name: str, items: list):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager",
                                                 dtype=torch.bfloat16).to("cuda").eval()
    for k, item in enumerate(items):
        r = run_one(model, tok, item, torch)
        print(model_name, k + 1, "/", len(items), flush=True)
        yield json.dumps(r)


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-14B-Instruct", limit: int = 8):
    items = [x for x in json.load(open(os.path.join(HERE, "..", "match", "sets_match.json")))
             if x["arm"] in ("A_change_60", "B_change_60", "C_change_60")
             or (x["arm"] in ("A_nochange_60", "B_nochange_60", "C_nochange_60") and x["probes"][0]["id"] in ("D1", "D3"))]
    if limit:
        items = items[:limit // 2] + items[-(limit - limit // 2):]
    out = os.path.join(HERE, "damp_" + model.split("/")[-1] + ("" if not limit else "_test") + ".jsonl")
    with open(out, "w") as f:
        for line in probe.remote_gen(model, items):
            f.write(line + "\n")
            f.flush()
    print("wrote", out, len(items), flush=True)
