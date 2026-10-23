"""CPU-only: fetch extra inverse-scaling tasks (mirrors) as jsonl into data/. Tries each candidate name, keeps the first that loads."""
import json, os, gzip, base64
import modal
HERE = os.path.dirname(os.path.abspath(__file__))
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub", "datasets")
app = modal.App("silo-fetch-ti", image=image)
WANT = {"memo-trap": ["inverse-scaling/memo-trap", "Albertmade/memo-trap", "jmichaelov/inverse_scaling_prize-memo_trap"],
        "modus-tollens": ["inverse-scaling/modus-tollens", "Albertmade/modus-tollens", "jmichaelov/inverse_scaling_prize-modus_tollens"],
        "into-the-unknown": ["inverse-scaling/into-the-unknown", "Albertmade/into-the-unknown", "jmichaelov/inverse_scaling_prize-into_the_unknown"],
        "sig-figs": ["inverse-scaling/sig-figs", "Albertmade/sig-figs", "jmichaelov/inverse_scaling_prize-sig_figs"]}
WANT["mmlu"] = ["cais/mmlu"]

@app.function(timeout=900)
def fetch(keys: list):
    from datasets import load_dataset
    for key in keys:
        names = WANT[key]
        for name in names:
            try:
                ds = load_dataset(name, "all", split="test") if key == "mmlu" else load_dataset(name)
                if key == "mmlu": ds = {"test": ds}
                sp = list(ds.keys())[0]; rows = [dict(r) for r in ds[sp]]
                blob = base64.b64encode(gzip.compress("\n".join(json.dumps(r) for r in rows).encode())).decode()
                for i in range(0, len(blob), 900000):
                    yield json.dumps({"key": key, "src": name, "part": blob[i:i + 900000], "last": i + 900000 >= len(blob)})
                break
            except Exception as e:
                yield json.dumps({"key": key, "src": name, "error": repr(e)[:200]})

@app.local_entrypoint()
def main(keys: str = "memo-trap,modus-tollens,into-the-unknown,sig-figs"):
    parts = {}
    for line in fetch.remote_gen(keys.split(",")):
        m = json.loads(line)
        if "error" in m: print("ERR", m["src"], m["error"]); continue
        parts.setdefault(m["key"], []).append(m["part"])
        if not m["last"]: continue
        rows = gzip.decompress(base64.b64decode("".join(parts[m["key"]]))).decode()
        fn = os.path.join(HERE, "data", m["key"] + ".jsonl"); open(fn, "w").write(rows + "\n")
        print("saved", m["key"], "from", m["src"], rows.count("\n") + 1)
