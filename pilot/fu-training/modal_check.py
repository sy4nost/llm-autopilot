"""Score check.json (or another test file) with the base model and with the saved fu adapter.
  modal run modal_check.py --file check.json --adapter /cache/fu/adapter"""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate", "peft")
         .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"}))
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-fu-check", image=image)


@app.function(gpu="L40S", volumes={"/cache": cache}, timeout=3600)
def run(model_name: str, adapter: str, test: list):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda").eval()
    chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)

    def score(m, name):
        for n, it in enumerate(test):
            pid = tok(chat(it["prompt"]), add_special_tokens=False)["input_ids"]
            out = []
            for c in it["cands"]:
                r = tok(c, add_special_tokens=False)["input_ids"]
                ids = torch.tensor([pid + r]).cuda()
                with torch.no_grad():
                    lg = m(input_ids=ids, logits_to_keep=len(r) + 1).logits[0, :-1].float()
                out.append(torch.log_softmax(lg, -1).gather(-1, torch.tensor(r).cuda().unsqueeze(-1)).sum().item())
            yield json.dumps(dict(stage="test", model=name, i=n, fam=it["fam"], kind=it["kind"], key=it["key"],
                                  old=it["old"], cands=it["cands"], scores=[round(x, 3) for x in out]))
        print("done", name, flush=True)
    yield from score(model, "base")
    m2 = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
    yield from score(m2, "tuned")


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-7B-Instruct", adapter: str = "/cache/fu/adapter", file: str = "check.json", tag: str = "_check"):
    test = json.load(open(os.path.join(HERE, file)))
    out = os.path.join(HERE, "fu_" + model.split("/")[-1] + tag + ".jsonl")
    with open(out, "w") as f:
        for line in run.remote_gen(model, adapter, test):
            f.write(line + "\n")
    print("wrote", out, flush=True)
