"""Look inside the model on the v3 shift prompts (change_0/20/60, probes D1-D4), one prompt at a time.
For the answer position (last prompt token) it records:
  - label scores: log-prob of the first token of each of the 3 labels (final layer)
  - logit lens: the same at every layer (final norm + unembedding on the residual stream)
  - attention: share of attention (mean over heads) from the last position to each part of the prompt, per layer:
      rules, examples, update (the rule-change line), test (the test case and instructions), other
  - knockouts (causal check): final label scores when the positions from the test case onward cannot attend to
      K_ex = the solved examples, K_up = the update line. Done with a custom 4D attention mask; mask_check is the
      max |logit diff| of a plain causal 4D mask vs the normal run (must be ~0, otherwise knockouts are invalid).
  modal run modal_inside.py --model Qwen/Qwen2.5-14B-Instruct --gpu H100 --limit 0   -> inside_<model>.jsonl
"""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-rule-inside", image=image)
SEGS = ["rules", "examples", "update", "test", "other"]


def spans(text):
    """char spans of the prompt parts inside the chat-formatted text"""
    def find(a, b=None, start=0):
        i = text.find(a, start)
        if i < 0:
            return None
        j = text.find(b, i) if b else len(text)
        return (i, j if j >= 0 else len(text))
    s = {"rules": find("RULE 1", "Every case has")}
    s["examples"] = find("Below are", "IMPORTANT UPDATE") if "Below are" in text else None
    u = find("IMPORTANT UPDATE", "stay the same.")
    s["update"] = (u[0], u[1] + len("stay the same.")) if u else None
    s["test"] = find("Classify the TEST case below")
    return s


def run_one(model, tok, item, n_layers, torch):
    text = tok.apply_chat_template([{"role": "user", "content": item["prompt"]}], tokenize=False,
                                   add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    ids = enc["input_ids"].to("cuda")
    offs = enc["offset_mapping"][0].tolist()
    T = ids.shape[1]
    sp = spans(text)
    seg_of = []
    for a, _ in offs:
        name = "other"
        for k, v in sp.items():
            if v and v[0] <= a < v[1]:
                name = k
        seg_of.append(SEGS.index(name))
    seg_t = torch.tensor(seg_of, device="cuda")
    lab_ids = [tok.encode(l, add_special_tokens=False)[0] for l in item["labels"]]

    attn_rows = [None] * n_layers
    hooks = []
    for li, layer in enumerate(model.model.layers):
        def hook(mod, inp, out, li=li):
            w = out[1]
            if w is None:
                return
            row = w[0, :, -1, :].float().mean(0)  # mean over heads, attention from the last position
            attn_rows[li] = [float(row[seg_t == k].sum()) for k in range(len(SEGS))]
        hooks.append(layer.self_attn.register_forward_hook(hook))
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, logits_to_keep=1)
    for h in hooks:
        h.remove()
    final = torch.log_softmax(out.logits[0, -1].float(), -1)
    lens = []
    norm, head = model.model.norm, model.get_output_embeddings()
    with torch.no_grad():
        mid = torch.stack([h[0, -1, :] for h in out.hidden_states[1:-1]])
        lp = torch.log_softmax(head(norm(mid)).float(), -1)[:, lab_ids]
    lens = [[round(float(x), 3) for x in r] for r in lp.tolist()] + [[round(float(final[i]), 3) for i in lab_ids]]

    # knockouts with an additive 4D mask
    neg = torch.finfo(model.dtype).min
    base = torch.full((T, T), neg, device="cuda", dtype=model.dtype).triu(1)
    test_start = next((i for i, s in enumerate(seg_of) if s == SEGS.index("test")), T - 1)

    def masked_final(block_seg):
        m = base.clone()
        if block_seg is not None:
            cols = (seg_t == SEGS.index(block_seg))
            if not bool(cols.any()):
                return None
            m[test_start:, cols] = neg
        with torch.no_grad():
            o = model(input_ids=ids, attention_mask=m[None, None], logits_to_keep=1)
        return o.logits[0, -1].float()

    plain = masked_final(None)
    mask_check = float((plain - out.logits[0, -1].float()).abs().max())
    ko = {}
    for name, seg in (("K_ex", "examples"), ("K_up", "update")):
        lg = masked_final(seg)
        ko[name] = None if lg is None else [round(float(x), 3) for x in torch.log_softmax(lg, -1)[lab_ids]]
    pr = item["probes"][0]
    return dict(set=item["set"], arm=item["arm"], probe=pr["id"], labels=item["labels"], key=pr["key"], old=pr["old"],
                T=T, label_tok=[tok.decode([i]) for i in lab_ids], final=lens[-1], lens=lens,
                attn=[[round(x, 4) for x in r] if r else None for r in attn_rows], knockout=ko,
                mask_check=round(mask_check, 4), seg_tokens=[seg_of.count(k) for k in range(len(SEGS))])


@app.function(gpu="L40S", volumes={"/cache": cache}, timeout=3 * 3600)
def probe(model_name: str, items: list):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager",
                                                 dtype=torch.bfloat16).to("cuda").eval()
    n_layers = len(model.model.layers)
    for k, item in enumerate(items):
        r = run_one(model, tok, item, n_layers, torch)
        print(model_name, k + 1, "/", len(items), flush=True)
        yield json.dumps(r)


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-0.5B-Instruct", gpu: str = "L40S", limit: int = 8):
    items = [x for x in json.load(open(os.path.join(HERE, "..", "sets_shift.json")))
             if x["arm"] in ("change_0", "change_20", "change_60") and x["probes"][0]["id"].startswith("D")]
    if limit:
        items = items[:limit // 2] + items[-(limit - limit // 2):]
    out = os.path.join(HERE, "inside_" + model.split("/")[-1] + ("" if not limit else "_test") + ".jsonl")
    with open(out, "w") as f:
        for line in probe.with_options(gpu=gpu).remote_gen(model, items):
            f.write(line + "\n")
            f.flush()
    print("wrote", out, flush=True)
