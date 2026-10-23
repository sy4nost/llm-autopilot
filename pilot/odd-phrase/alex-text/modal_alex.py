"""Alex's text (6 planted logical inconsistencies) on Qwen2.5 models, greedy decoding, bfloat16.

Conditions per model:
  A  turn 1 = text only (no instruction) -> reply; turn 2 = Alex's instructions about "the text above" -> reply
  B  the same, on the control text (the 6 planted sentences removed) -> false alarms
  C  one message: Alex's instructions + the text (his original format)
  modal run modal_alex.py   -> results/<model>.json
"""

import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
          "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct"]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
    .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"})
)
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-alex-text", image=image)


@app.function(gpu="L40S", volumes={"/cache": cache}, timeout=3600)
def run(model_name: str, text: str, control: str, ask: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(torch.bfloat16).to("cuda").eval()

    def reply(msgs, n):
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = {k: v.to("cuda") for k, v in ids.items()}
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=n, do_sample=False)
        return tok.decode(g[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    ask_turn2 = ask.replace("Read the following research-style text carefully.",
                            "Read the research-style text above carefully again.")
    out = {"model": model_name}
    for cond, t in (("A", text), ("B", control)):
        m = [{"role": "user", "content": t}]
        r1 = reply(m, 700)
        r2 = reply(m + [{"role": "assistant", "content": r1}, {"role": "user", "content": ask_turn2}], 1200)
        out[cond] = {"turn1": r1, "turn2": r2}
    out["C"] = {"turn1": reply([{"role": "user", "content": ask + "\n---\n\n" + text}], 1200)}
    cache.commit()
    return out


@app.local_entrypoint()
def main(models: str = ",".join(MODELS)):
    rd = lambda n: open(os.path.join(HERE, n), encoding="utf-8").read()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    names = [m for m in models.split(",") if m]
    for name, res in zip(names, run.map(names, kwargs={"text": rd("text_en.md"), "control": rd("text_en_control.md"),
                                                       "ask": rd("ask_en.txt")})):
        with open(os.path.join(HERE, "results", name.split("/")[-1] + ".json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print("done", name, flush=True)
