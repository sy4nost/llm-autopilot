"""'Lamp' = conflict between the past and the current rule, measured inside 14B at the answer position.
Copy heads: the top-20 heads from heads/ (ranked on sets 0-14), file top20.json.
Per prompt, normal run:
  rule reading: logit lens (final norm + unembedding) on the residual stream BEFORE the copy heads' layers
                (after 20, 24, 28 and 29 layers; the heads start at layer 29), scores of the 3 labels;
  copy push:   what the 20 heads add to the residual stream at the answer position (their o_proj slice),
               read straight through the unembedding (scaled by the final-norm weight), scores of the 3 labels;
  final:       normal label log-probs.
Second run: the 20 heads switched off from the test case onward (as in heads/), label log-probs.
lamp(L) = argmax copy push != argmax rule reading at L. Gated answer = switched-off run if the lamp is on.
  modal run modal_lamp.py --limit 0 -> lamp_<model>.jsonl     (--limit 6 quick check)
"""
import json, os, re
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-rule-lamp", image=image)
LENS = [20, 24, 28, 29]


@app.function(gpu="H100", volumes={"/cache": cache}, timeout=4 * 3600)
def probe(model_name: str, items: list, heads: list):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda").eval()
    layers = model.model.layers
    H = model.config.num_attention_heads
    D = model.config.hidden_size // H
    norm, head = model.model.norm, model.get_output_embeddings()
    by_layer = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)
    cur = {}

    def mk(li):
        def p(mod, args):
            if li not in by_layer:
                return None
            x = args[0]
            if cur.get("capture"):
                W = mod.weight  # (hidden, H*D)
                for h in by_layer[li]:
                    v = x[0, -1, h * D:(h + 1) * D]
                    c = W[:, h * D:(h + 1) * D] @ v
                    cur["push"] = cur.get("push", 0) + c.float()
            if cur.get("off"):
                x = x.clone()
                for h in by_layer[li]:
                    x[:, cur["t0"]:, h * D:(h + 1) * D] = 0
                return (x,) + args[1:]
            return None
        return p
    hooks = [layer.self_attn.o_proj.register_forward_pre_hook(mk(li)) for li, layer in enumerate(layers)]
    for n, it in enumerate(items):
        text = tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False,
                                       add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
        offs = enc["offset_mapping"][0].tolist()
        t0 = next(k for k, o in enumerate(offs) if o[0] >= text.find("Classify the TEST"))
        ids = enc["input_ids"].to("cuda")
        lab = [tok.encode(l, add_special_tokens=False)[0] for l in it["labels"]]
        cur.clear()
        cur["capture"] = True
        with torch.no_grad():
            o = model(input_ids=ids, output_hidden_states=True, logits_to_keep=1)
            final = torch.log_softmax(o.logits[0, -1].float(), -1)[lab].tolist()
            lens = {}
            for L in LENS:
                lens[L] = torch.log_softmax(head(norm(o.hidden_states[L][0, -1])).float(), -1)[lab].tolist()
            push = (head.weight[lab].float() @ (cur["push"] * norm.weight.float())).tolist()
            cur.clear()
            cur.update(off=True, t0=t0)
            o2 = model(input_ids=ids, logits_to_keep=1)
            off = torch.log_softmax(o2.logits[0, -1].float(), -1)[lab].tolist()
        cur.clear()
        pr = it["probes"][0]
        yield json.dumps(dict(set=it["set"], arm=it["arm"], probe=pr["id"], labels=it["labels"], key=pr["key"],
                              old=pr["old"], final=[round(x, 3) for x in final], off=[round(x, 3) for x in off],
                              push=[round(x, 3) for x in push], lens={str(k): [round(x, 3) for x in v] for k, v in lens.items()}))
        print("lamp", n + 1, "/", len(items), flush=True)
    for h in hooks:
        h.remove()


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-14B-Instruct", limit: int = 6):
    items = [x for x in json.load(open(os.path.join(HERE, "..", "match", "sets_match.json")))
             if x["arm"] in ("A_change_60", "B_change_60", "C_change_60")
             or (x["arm"] in ("A_nochange_60", "B_nochange_60", "C_nochange_60") and x["probes"][0]["id"] in ("D1", "D3"))]
    if limit:
        items = [x for x in items if x["set"] == 20][:limit]
    heads = json.load(open(os.path.join(HERE, "top20.json")))
    out = os.path.join(HERE, "lamp_" + model.split("/")[-1] + ("" if not limit else "_test") + ".jsonl")
    with open(out, "w") as f:
        for line in probe.remote_gen(model, items, heads):
            f.write(line + "\n")
            f.flush()
    print("wrote", out, flush=True)
