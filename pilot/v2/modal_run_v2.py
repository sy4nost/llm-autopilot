"""Run run_v2.py on a Modal GPU (L4 by default) and copy the results into the shared folder.

For every model: first the N=0 gate (--n 0), then, if the gate passes, the full run. Full runs of models that passed
are started in parallel. run_v2.py and analyze_v2.py are taken from this folder at launch time (not edited here).

Usage (from the cloud container; needs MODAL_TOKEN_ID / MODAL_TOKEN_SECRET and pip install 'modal[api-proxy-support]'):
  modal run modal_run_v2.py                                   # 4 Qwen2.5 models, seed 1, 512 items
  modal run modal_run_v2.py --models Qwen/Qwen2.5-0.5B-Instruct --gate-only
  modal run modal_run_v2.py --extra "--dtype bfloat16"        # extra args passed to run_v2.py as they are
  modal run modal_run_v2.py --force                           # full run even if the gate fails
Results: /mnt/project-files/pilot/run-v2-<date>/<model>-seed<seed>[-gate]/ (trials.csv, layers.csv, run_info.txt,
summary.txt, log.txt).
"""

import os
import re
import shlex
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
GPU = os.environ.get("V2_GPU", "L4")
GPU = None if GPU.lower() == "cpu" else GPU  # V2_GPU=cpu for a test without a GPU
TIMEOUT_S = 3 * 3600  # hard stop per call, keeps a stuck run from burning credits

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.1", "transformers==5.20.0", "accelerate")
    .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"})
    .add_local_file(os.path.join(HERE, "run_v2.py"), "/app/run_v2.py")
    .add_local_file(os.path.join(HERE, "analyze_v2.py"), "/app/analyze_v2.py")
)
cache = modal.Volume.from_name("silo-hf-cache", create_if_missing=True)
app = modal.App("silo-rule-switch-v2", image=image)


CHUNK = 1024 * 1024  # results come back gzipped in chunks under Modal's 2 MiB inline limit, so no blob download is
# needed (the blob host is not reachable from the cloud container)


@app.function(gpu=GPU, volumes={"/cache": cache}, timeout=TIMEOUT_S)
def run_remote(args: list[str]):
    import gzip
    import pickle
    import subprocess
    import sys

    out = "/tmp/out"
    t0 = time.time()
    p = subprocess.Popen([sys.executable, "/app/run_v2.py", *args, "--out", out], cwd="/app",
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = []
    for line in p.stdout:
        print(line, end="")
        log.append(line)
    p.wait()
    cache.commit()
    files = {"log.txt": "".join(log).encode()}
    if os.path.isdir(out):
        for name in os.listdir(out):
            with open(os.path.join(out, name), "rb") as f:
                files[name] = f.read()
    blob = gzip.compress(pickle.dumps({"code": p.returncode, "files": files, "seconds": time.time() - t0}))
    for i in range(0, len(blob), CHUNK):
        yield blob[i:i + CHUNK]


def call(args):
    """Run run_remote and put the chunks back together (blocking; use in a thread for parallel runs)."""
    import gzip
    import pickle
    return pickle.loads(gzip.decompress(b"".join(run_remote.remote_gen(args))))


def tag(model):
    return re.sub(r"[^A-Za-z0-9._-]", "_", model.rstrip("/").split("/")[-1])


def save(res, folder):
    os.makedirs(folder, exist_ok=True)
    for name, data in res["files"].items():
        with open(os.path.join(folder, name), "wb") as f:
            f.write(data)


def gate_verdict(res):
    text = res["files"].get("summary.txt", b"").decode("utf-8", "replace")
    m = re.search(r"->\s*(PASS|FAIL)", text)
    return m.group(1) if m else "NONE"


@app.local_entrypoint()
def main(models: str = "Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-3B-Instruct,"
                       "Qwen/Qwen2.5-7B-Instruct",
         seed: int = 1, trials: int = 512, extra: str = "", date: str = "", gate_only: bool = False,
         force: bool = False, root: str = "/mnt/project-files/pilot"):
    date = date or time.strftime("%Y-%m-%d")
    base = os.path.join(root, f"run-v2-{date}")
    extra_args = shlex.split(extra)
    common = ["--trials", str(trials), "--seed", str(seed), *extra_args]
    model_list = [m.strip() for m in models.split(",") if m.strip()]

    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=2 * len(model_list))
    gates = {m: pool.submit(call, ["--model", m, "--n", "0", *common]) for m in model_list}
    full = {}
    for m, fut in gates.items():
        res = fut.result()
        save(res, os.path.join(base, f"{tag(m)}-seed{seed}-gate"))
        v = gate_verdict(res)
        print(f"[gate] {m}: exit={res['code']} verdict={v} ({res['seconds']:.0f} s)")
        if not gate_only and res["code"] == 0 and (v == "PASS" or force):
            full[m] = pool.submit(call, ["--model", m, *common])
    for m, fut in full.items():
        res = fut.result()
        save(res, os.path.join(base, f"{tag(m)}-seed{seed}"))
        print(f"[full] {m}: exit={res['code']} ({res['seconds'] / 60:.1f} min)")
    print(f"results in {base}")
