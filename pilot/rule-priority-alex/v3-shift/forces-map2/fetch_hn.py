"""Fetch inverse-scaling/hindsight-neglect-10shot via a Modal CPU container (HF is blocked from this container)."""
import json, os, modal
app = modal.App("silo-fetch-hn", image=modal.Image.debian_slim(python_version="3.12").pip_install("datasets"))


@app.function(timeout=600)
def get():
    from datasets import load_dataset
    return json.dumps([dict(r) for r in load_dataset("inverse-scaling/hindsight-neglect-10shot", split="train")])


@app.local_entrypoint()
def main():
    rows = json.loads(get.remote())
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hindsight.json"), "w").write(json.dumps(rows))
    print(len(rows), list(rows[0]))
