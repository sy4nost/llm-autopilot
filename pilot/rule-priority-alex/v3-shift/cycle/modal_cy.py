"""Life-cycle frame test (PLAN.md) on Qwen2.5-14B-Instruct, bf16. Forced choice only (no generation), same code path as
thinkaloud/modal_ta.py direct(): each option its own full forward pass, per-token log-probs saved.
Arms: death / goal / placebo / facts (BASE = thinkaloud DIRECT, reproduced 78/78 in smoke v1) = system prompt (sets_cy.json) + counter line before the task (none for base).
  modal run --detach modal_cy.py --sets sets_cy_smoke.json --tag smoke ; modal run modal_cy.py --tag smoke --get
State on the volume (/cache/cy_runs/<tag>/out.jsonl), resumable. Hard cap: timeout 2 h (H100 ~ $4/h)."""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-cycle", image=image)
MODEL = "Qwen/Qwen2.5-14B-Instruct"
R_TAIL = "\n\nReply with only the answer, nothing else."
ARMS = ["death", "goal", "placebo", "facts"]


@app.function(gpu="H100", volumes={"/cache": cache}, timeout=2 * 3600)
def run(sets: dict, tag: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    SYS, CNT = sets["system"], sets["counter"]

    def chat(arm, u, k):
        if CNT[arm]:
            u = CNT[arm].format(k=k) + "\n\n" + u
        return tok.apply_chat_template([{"role": "system", "content": SYS[arm]}, {"role": "user", "content": u}],
                                       tokenize=False, add_generation_prompt=True)

    def score(prefix, opts):
        pid = tok(prefix, add_special_tokens=False)["input_ids"]; res = []
        for o in opts:
            t = tok(o, add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                lg = model(input_ids=torch.tensor([pid + t]).cuda()).logits[0, len(pid) - 1:-1].float()
            lp = torch.log_softmax(lg, -1)
            res.append([round(lp[i, t[i]].item(), 4) for i in range(len(t))])
        return res

    def direct(arm, it):
        if it["family"] == "N":
            return score(chat(arm, it["direct"], it["k"]), list(it["cands"]))
        return score(chat(arm, it["direct"] + R_TAIL, it["k"]) + "The answer is", [" " + c for c in it["cands"]])

    # sanity: explicit default system + no counter must equal the template's implicit default (thinkaloud DIRECT prompts)
    u = sets["items"][0]["direct"]
    same = tok.apply_chat_template([{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                                    {"role": "user", "content": u}], tokenize=False, add_generation_prompt=True) == \
        tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    print("default system == thinkaloud prompt:", same, flush=True)
    d = f"/cache/cy_runs/{tag}"; os.makedirs(d, exist_ok=True); ofn = f"{d}/out.jsonl"
    done = {json.loads(l)["uid"] for l in open(ofn) if l.endswith("}\n") and '"res"' in l} if os.path.exists(ofn) else set()
    of = open(ofn, "a")
    if not done:
        of.write(json.dumps(dict(checks=dict(base_equals_thinkaloud_prompt=same))) + "\n")
    for n, it in enumerate(sets["items"]):
        if it["uid"] in done:
            continue
        of.write(json.dumps(dict({k: v for k, v in it.items() if k != "direct"}, res={a: direct(a, it) for a in ARMS})) + "\n")
        if (n + 1) % 50 == 0:
            of.flush(); cache.commit(); print("score", n + 1, "/", len(sets["items"]), flush=True)
    of.close(); cache.commit(); print("DONE", ofn, flush=True)


@app.function(volumes={"/cache": cache}, timeout=900)
def fetch(tag: str):
    """out.jsonl from the volume in <1 MiB chunks (the proxy blocks Modal's large-blob host)"""
    import gzip, base64
    cache.reload()
    blob = base64.b64encode(gzip.compress(open(f"/cache/cy_runs/{tag}/out.jsonl", "rb").read())).decode()
    for i in range(0, len(blob), 900000):
        yield blob[i:i + 900000]


@app.local_entrypoint()
def main(sets: str = "sets_cy_smoke.json", tag: str = "smoke", get: bool = False):
    out = os.path.join(HERE, f"out_{tag}.jsonl")
    if get:
        import gzip, base64
        data = gzip.decompress(base64.b64decode("".join(fetch.remote_gen(tag))))
        open(out, "wb").write(data); print("wrote", out, data.count(b"\n"), "lines"); return
    run.remote(json.load(open(os.path.join(HERE, sets))), tag)
