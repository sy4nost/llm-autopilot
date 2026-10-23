"""CPU-only: list inverse-scaling datasets on the HF hub and save the ones we need as jsonl (returned inline, gzip chunks)."""
import json, os, gzip, base64
import modal
HERE = os.path.dirname(os.path.abspath(__file__))
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub", "datasets")
app = modal.App("silo-fetch-indep", image=image)

@app.function(timeout=900)
def fetch():
    from huggingface_hub import HfApi
    api = HfApi()
    ids = ["inverse-scaling/hindsight-neglect-10shot"]
    yield json.dumps({"ids": ids})
    from datasets import load_dataset
    for name in ids:
        try:
            ds = load_dataset(name, split="train")
            rows = [dict(r) for r in ds]
            blob = base64.b64encode(gzip.compress("\n".join(json.dumps(r) for r in rows).encode())).decode()
            for i in range(0, len(blob), 900000):
                yield json.dumps({"name": name, "part": blob[i:i + 900000], "last": i + 900000 >= len(blob)})
        except Exception as e:
            yield json.dumps({"name": name, "error": repr(e)[:300]})

@app.local_entrypoint()
def main():
    parts = {}
    for line in fetch.remote_gen():
        m = json.loads(line)
        if "ids" in m: print("datasets:", m["ids"])
        elif "error" in m: print("ERR", m["name"], m["error"])
        else:
            parts.setdefault(m["name"], []).append(m["part"])
            if m["last"]:
                rows = gzip.decompress(base64.b64decode("".join(parts[m["name"]]))).decode()
                fn = os.path.join(HERE, "data", m["name"].split("/")[-1] + ".jsonl")
                open(fn, "w").write(rows + "\n"); print("saved", fn, rows.count("\n") + 1)
