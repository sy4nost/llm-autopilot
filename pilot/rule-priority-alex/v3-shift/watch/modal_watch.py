"""'Watchman' test (PREREG.md). The watchman does not solve the test case. It re-reads the current task statement (the prompt
with all solved examples removed: rules + any notes) and, for every past example, asks the model for that example's answer
under the current statement, without any history. An example whose recorded answer differs from that fresh answer is
'stale' (made under other conditions). If any example is stale, the lamp is on.
Brake: the stale example lines are silenced (attention set to the mask minimum) from the test question onward,
  watch_b40  in the 40 copy heads (b40, eras/settings2.json)        <- primary
  watch_all  in every head of every layer                           <- secondary
  oracle_b40 the truly stale lines (ground truth from the text checkers) in b40: what a perfect watchman would give (report only)
No note boosting, no thresholds. normal = untouched model.
  modal run modal_watch.py [--n 2] [--sets sets_watch.json] [--tag _smoke]   -> watch_<model><tag>.jsonl (first row = checks)
"""
import json, os, re
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-watch14", image=image)
EX = re.compile(r"Example \d+: (.+?) -> (\S+)\n?")


def recheck_prompt(p, case):
    """current statement without the history: drop solved-example lines and the 'Below are ...' intro lines"""
    lines = [l for l in p.split("\n") if not re.match(r"Example \d+: ", l) and not l.startswith("Below are ")]
    t = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return re.sub(r"TEST CASE: .+\n", "TEST CASE: " + case + "\n", t)


@app.function(gpu="H100", volumes={"/cache": cache}, timeout=3 * 3600)
def run(model_name: str, b40: list, items: list):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
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
    first = lambda s: tok.encode(s, add_special_tokens=False)[0]

    memo = {}
    def fresh(text):   # first answer token of the model for a history-free prompt
        if text not in memo:
            ids = tok(chat(text), return_tensors="pt")["input_ids"].to("cuda")
            with torch.no_grad():
                memo[text] = int(model(input_ids=ids, logits_to_keep=1).logits[0, -1].argmax())
        return memo[text]

    checks = dict(max_zero_flag_diff=0.0, n_zero_flag=0, example_recheck_shown=None)
    out = []
    for n, it in enumerate(items):
        p = it["prompt"]
        text = chat(p)
        off = text.index(p)
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
        offs = enc["offset_mapping"][0].tolist()
        pos = lambda c: next(j for j, o in enumerate(offs) if o[0] >= c and o[1] > o[0])
        qpos = pos(text.find("Answer for the TEST"))
        ex = list(EX.finditer(p))
        flags = []
        for mm in ex:
            rp = recheck_prompt(p, mm.group(1))
            if checks["example_recheck_shown"] is None and it["arm"] == "A60":
                checks["example_recheck_shown"] = rp
                checks["example_span_shown"] = tok.decode(enc["input_ids"][0][pos(off + mm.start()):pos(off + mm.end())])
            flags.append(fresh(rp) != first(mm.group(2)))
        span = lambda mm: (pos(off + mm.start()), pos(off + mm.end()))   # whole example line, up to the next line's first token
        flagged = [span(mm) for mm, f in zip(ex, flags) if f]
        truth = [span(mm) for mm, e in zip(ex, it["examples"]) if e["stale"]]
        assert [e["case"] for e in it["examples"]] == [mm.group(1) for mm in ex]
        lab = [first(l) for l in it["labels"]]
        assert len(set(lab)) == len(lab)
        ids = enc["input_ids"].to("cuda")
        res = {}
        for name, hs, sp in (("normal", None, []), ("watch_b40", "b40", flagged), ("watch_all", "all", flagged), ("oracle_b40", "b40", truth)):
            cur.update(hs=hs, spans=sp, q=qpos)
            with torch.no_grad():
                lg = model(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
            res[name] = [round(x, 3) for x in torch.log_softmax(lg, -1)[lab].tolist()]
        cur["spans"] = []
        if not flagged:
            checks["n_zero_flag"] += 1
            checks["max_zero_flag_diff"] = max(checks["max_zero_flag_diff"], max(abs(x - y) for x, y in zip(res["watch_b40"], res["normal"])))
        out.append(json.dumps(dict(task=it["task"], set=it["set"], arm=it["arm"], probe=it["probe"], labels=it["labels"], key=it["key"],
                                   old=it["old"], scores=res, flags=flags, truth=[e["stale"] for e in it["examples"]])))
        if (n + 1) % 40 == 0:
            print("watch", n + 1, "/", len(items), "rechecks cached", len(memo), flush=True)
    yield json.dumps(dict(checks=checks))
    for line in out:
        yield line


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-14B-Instruct", n: int = 0, sets: str = "sets_watch.json", tag: str = ""):
    b40 = json.load(open(os.path.join(HERE, "..", "eras", "settings2.json")))["b40"]
    items = json.load(open(os.path.join(HERE, sets)))["items"]
    if n:
        seen, keep = {}, []
        for x in items:
            k = (x["task"], x["arm"])
            if seen.get(k, 0) < n:
                seen[k] = seen.get(k, 0) + 1; keep.append(x)
        items = keep
    out = os.path.join(HERE, "watch_" + model.split("/")[-1] + tag + ".jsonl")
    with open(out, "w") as f:
        for line in run.remote_gen(model, b40, items):
            f.write(line + "\n")
            f.flush()
    print("wrote", out, flush=True)
