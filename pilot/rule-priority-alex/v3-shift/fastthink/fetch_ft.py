"""CPU-only: fetch public tests not used before (mirrors) as jsonl into data/. Tries each candidate name, keeps the first that loads."""
import json, os, gzip, base64
import modal
HERE = os.path.dirname(os.path.abspath(__file__))
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub", "datasets")
app = modal.App("silo-fetch-ft", image=image)
WANT = {"resisting-correction": [("inverse-scaling/resisting-correction", None), ("Albertmade/resisting-correction", None)],
        "prompt-injection": [("inverse-scaling/prompt-injection", None), ("Albertmade/prompt-injection", None)],
        "hans": [("jhu-cogsci/hans", None), ("hans", None), ("aps/hans", None), ("tasksource/hans", None)],
        "epistemic": [("tasksource/bigbench", "epistemic_reasoning"), ("google/bigbench", "epistemic_reasoning")],
        "truthfulqa": [("truthfulqa/truthful_qa", "multiple_choice"), ("truthful_qa", "multiple_choice")]}


@app.function(timeout=900)
def fetch(keys: list):
    from datasets import load_dataset
    for key in keys:
        for name, cfg in WANT[key]:
            try:
                ds = load_dataset(name, cfg) if cfg else load_dataset(name)
                sp = "validation" if "validation" in ds else list(ds.keys())[0]; rows = [dict(r) for r in ds[sp]]
                blob = base64.b64encode(gzip.compress("\n".join(json.dumps(r) for r in rows).encode())).decode()
                for i in range(0, len(blob), 900000):
                    yield json.dumps({"key": key, "src": name, "part": blob[i:i + 900000], "last": i + 900000 >= len(blob)})
                break
            except Exception as e:
                yield json.dumps({"key": key, "src": name, "error": repr(e)[:200]})


@app.local_entrypoint()
def main(keys: str = "resisting-correction,prompt-injection,truthfulqa"):
    parts = {}
    for line in fetch.remote_gen(keys.split(",")):
        m = json.loads(line)
        if "error" in m:
            print("ERR", m["src"], m["error"]); continue
        parts.setdefault(m["key"], []).append(m["part"])
        if not m["last"]:
            continue
        rows = gzip.decompress(base64.b64decode("".join(parts[m["key"]]))).decode()
        open(os.path.join(HERE, "data", m["key"] + ".jsonl"), "w").write(rows + "\n")
        print("saved", m["key"], "from", m["src"], rows.count("\n") + 1)
