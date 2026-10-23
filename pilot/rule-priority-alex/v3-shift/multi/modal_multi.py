"""Watchman on other models (PREREG.md). For each model, everything model-specific is found on OLD data only:
  1. copy heads: attention from the answer position to the label tokens of examples IDENTICAL to the test case,
     on the discovery prompts of match/sets_match.json (sets 0-14, A_change_60, probes D1/D3), as in heads/modal_heads.py;
     top 40 heads = this model's 'b40'.
  2. watchman exactly as watch/modal_watch.py, except that every answer (rechecks and final) is chosen among candidate
     strings by log-prob, so it does not depend on the tokenizer or on the model's habit of explaining first:
       candidates of a recheck = the item's labels + every answer seen in its examples;
       score = log-prob of the first token if the candidates' first tokens are all different, else summed log-prob of the whole string.
Settings: normal | watch_b40 (own top-40 heads) | watch_all | oracle_b40 | nohist (same prompt with ALL examples removed: the trivial baseline the audit asked for).
  modal run modal_multi.py --model M [--sets sets_multi.json] [--n 0] [--tag ""]   -> multi_<model><tag>.jsonl
  modal run modal_multi.py::fetch --model M     (download weights to the cache volume on a CPU box)
"""
import json, os, re
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.join(HERE, "..")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate", "huggingface_hub")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-multi", image=image)
EX = re.compile(r"Example \d+: (.+?) -> (\S+)\n?")


def recheck_prompt(p, case):
    """current statement without the history: drop solved-example lines and the 'Below are ...' intro lines"""
    lines = [l for l in p.split("\n") if not re.match(r"Example \d+: ", l) and not l.startswith("Below are ")]
    t = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return re.sub(r"TEST CASE: .+\n", "TEST CASE: " + case + "\n", t)


@app.function(cpu=2, memory=8192, volumes={"/cache": cache}, timeout=2 * 3600)
def download(model_name: str):
    from huggingface_hub import snapshot_download
    p = snapshot_download(model_name, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "*.py"])
    cache.commit()
    return p


@app.function(gpu="H100", volumes={"/cache": cache}, timeout=4 * 3600)
def run(model_name: str, disc: list, items: list):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", dtype=torch.bfloat16).to("cuda").eval()
    layers = model.model.layers
    H, NL = model.config.num_attention_heads, len(layers)
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    enc = lambda text: tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    checks = dict(model=model_name, layers=NL, heads=H, mask_kinds=set(), max_zero_flag_diff=0.0, n_zero_flag=0,
                  first_token_scoring=0, full_string_scoring=0)

    # ---- 1. copy heads on old discovery prompts
    att = torch.zeros(NL, H)
    for it in disc:
        text = chat(it["prompt"])
        e = enc(text)
        offs = e["offset_mapping"][0].tolist()
        test = re.search(r"TEST CASE: (.*)\n", text).group(1)
        ft = lambda c: next(k for k, o in enumerate(offs) if o[0] >= c and o[1] > o[0])
        exact = [ft(m.start(2)) for m in re.finditer(r"Example \d+: (.*?) -> (\w+)", text) if m.group(1) == test]
        ex_t = torch.tensor(exact, dtype=torch.long, device="cuda")
        hooks = []
        for li, layer in enumerate(layers):
            def hk(mod, inp, out, li=li):
                att[li] += out[1][0, :, -1, :].float()[:, ex_t].sum(-1).cpu()
            hooks.append(layer.self_attn.register_forward_hook(hk))
        with torch.no_grad():
            model(input_ids=e["input_ids"].to("cuda"), logits_to_keep=1)
        for h in hooks:
            h.remove()
    order = torch.argsort((att / len(disc)).flatten(), descending=True).tolist()
    top = [(i // H, i % H) for i in order[:40]]
    yield json.dumps(dict(heads=top, disc_prompts=len(disc), top_scores=[round(float(att.flatten()[i] / len(disc)), 4) for i in order[:40]]))

    # ---- 2. watchman
    heads = {"b40": {}, "all": {l: list(range(H)) for l in range(NL)}}
    for l, h in top:
        heads["b40"].setdefault(l, []).append(h)
    cur = {"hs": None, "spans": [], "q": 0}

    def mk(li):
        def pre(mod, args, kwargs):
            if not cur["spans"] or li not in heads[cur["hs"]]:
                return args, kwargs
            m = kwargs.get("attention_mask")
            assert m is not None and m.dim() == 4, "no 4-d attention mask at layer %d" % li
            checks["mask_kinds"].add(str(m.dtype))
            b, _, q, k = m.shape
            if m.dtype == torch.bool:
                m = torch.zeros(m.shape, dtype=model.dtype, device=m.device).masked_fill(~m, torch.finfo(model.dtype).min)
            m = m.expand(b, H, q, k).clone()
            hs = torch.tensor(heads[cur["hs"]][li], device=m.device)
            for a, e_ in cur["spans"]:
                m[:, hs, cur["q"]:, a:e_] = torch.finfo(m.dtype).min
            kwargs["attention_mask"] = m
            return args, kwargs
        return pre
    for li, layer in enumerate(layers):
        layer.register_forward_pre_hook(mk(li), with_kwargs=True)

    def score(ids, cands):
        """log-prob of each candidate string after the prompt ids [1, n] (hooks active as set in cur)"""
        toks = [tok.encode(c, add_special_tokens=False) for c in cands]
        if len({t[0] for t in toks}) == len(toks):
            checks["first_token_scoring"] += 1
            with torch.no_grad():
                lg = model(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
            lp = torch.log_softmax(lg, -1)
            return [round(lp[t[0]].item(), 3) for t in toks]
        checks["full_string_scoring"] += 1
        n, L = ids.shape[1], max(len(t) for t in toks)
        batch = torch.full((len(toks), n + L), tok.pad_token_id, dtype=torch.long)
        am = torch.zeros_like(batch)
        for j, t in enumerate(toks):
            batch[j, :n] = ids[0].cpu(); batch[j, n:n + len(t)] = torch.tensor(t); am[j, :n + len(t)] = 1
        with torch.no_grad():
            lg = model(input_ids=batch.cuda(), attention_mask=am.cuda()).logits.float()
        lp = torch.log_softmax(lg[:, n - 1:-1], -1)
        return [round(sum(lp[j, i, t[i]].item() for i in range(len(t))), 3) for j, t in enumerate(toks)]

    memo = {}
    def fresh(text, cands):
        k = (text, tuple(cands))
        if k not in memo:
            cur["spans"] = []
            sc = score(enc(chat(text))["input_ids"].to("cuda"), cands)
            memo[k] = cands[max(range(len(cands)), key=lambda j: sc[j])]
        return memo[k]

    for n, it in enumerate(items):
        p = it["prompt"]
        text = chat(p)
        off = text.index(p)
        e = enc(text)
        offs = e["offset_mapping"][0].tolist()
        pos = lambda c: next(j for j, o in enumerate(offs) if o[0] >= c and o[1] > o[0])
        qpos = pos(text.find("Answer for the TEST"))
        ex = list(EX.finditer(p))
        cands = sorted(set(it["labels"]) | {mm.group(2) for mm in ex})
        flags = [fresh(recheck_prompt(p, mm.group(1)), cands) != mm.group(2) for mm in ex]
        span = lambda mm: (pos(off + mm.start()), pos(off + mm.end()))
        flagged = [span(mm) for mm, f in zip(ex, flags) if f]
        truth = [span(mm) for mm, x in zip(ex, it["examples"]) if x["stale"]]
        assert [x["case"] for x in it["examples"]] == [mm.group(1) for mm in ex]
        ids = e["input_ids"].to("cuda")
        res = {}
        for name, hs, sp in (("normal", None, []), ("watch_b40", "b40", flagged), ("watch_all", "all", flagged), ("oracle_b40", "b40", truth)):
            cur.update(hs=hs, spans=sp, q=qpos)
            res[name] = score(ids, it["labels"])
        cur["spans"] = []
        tc = re.search(r"TEST CASE: (.+)\n", p).group(1)
        res["nohist"] = score(enc(chat(recheck_prompt(p, tc)))["input_ids"].to("cuda"), it["labels"])
        if not flagged:
            checks["n_zero_flag"] += 1
            checks["max_zero_flag_diff"] = max(checks["max_zero_flag_diff"], max(abs(x - y) for x, y in zip(res["watch_b40"], res["normal"])))
        yield json.dumps(dict(task=it["task"], set=it["set"], arm=it["arm"], probe=it["probe"], labels=it["labels"], key=it["key"],
                              old=it["old"], scores=res, flags=flags, truth=[x["stale"] for x in it["examples"]]))
        if (n + 1) % 40 == 0:
            print("multi", n + 1, "/", len(items), "rechecks cached", len(memo), flush=True)
    checks["mask_kinds"] = sorted(checks["mask_kinds"])
    yield json.dumps(dict(checks=checks))


@app.local_entrypoint()
def main(model: str = "microsoft/phi-4", n: int = 0, sets: str = "sets_multi.json", tag: str = "", fetch: int = 0):
    if fetch:
        print(download.remote(model)); return
    disc = [x for x in json.load(open(os.path.join(V3, "match", "sets_match.json")))
            if x["set"] in range(15) and x["arm"] == "A_change_60" and x["probes"][0]["id"] in ("D1", "D3")]
    items = json.load(open(os.path.join(HERE, sets)))["items"]
    if n:
        seen, keep = {}, []
        for x in items:
            k = (x["task"], x["arm"])
            if seen.get(k, 0) < n:
                seen[k] = seen.get(k, 0) + 1; keep.append(x)
        items = keep
    out = os.path.join(HERE, "multi_" + model.split("/")[-1] + tag + ".jsonl")
    with open(out, "w") as f:
        for line in run.remote_gen(model, disc, items):
            f.write(line + "\n")
            f.flush()
    print("wrote", out, flush=True)
