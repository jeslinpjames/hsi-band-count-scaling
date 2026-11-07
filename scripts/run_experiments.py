"""Run the experiments of the paper (or a subset) on one or more GPUs.

  python scripts/run_experiments.py --list
  python scripts/run_experiments.py --exp main --gpus 0,1,2,3 --out results/new
  python scripts/run_experiments.py --exp copernicus --python_copernicus ~/envs/cfm/bin/python --gpus 0
  python scripts/run_experiments.py --exp all --dry_run          # print the commands only

Runs already present in their CSV are skipped, so the command can be restarted at any time. Each GPU runs
--per_gpu runs at once; CPU-only runs (classical baselines) use --cpu_jobs slots.
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from experiments import EXPERIMENTS  # noqa: E402
from bcs.data import is_done  # noqa: E402
from bcs.paths import RESULTS_DIR  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--exp", default="main", help="comma-separated names, or 'all'")
p.add_argument("--out", default=os.path.join(ROOT, "results", "new"))
p.add_argument("--gpus", default="0")
p.add_argument("--per_gpu", type=int, default=1)
p.add_argument("--cpu_jobs", type=int, default=1)
p.add_argument("--python_dofa", default=sys.executable, help="interpreter of the DOFA environment")
p.add_argument("--python_copernicus", default=sys.executable, help="interpreter of the Copernicus-FM environment")
p.add_argument("--logs", default=None, help="log folder (default <out>/logs)")
p.add_argument("--dry_run", action="store_true")
p.add_argument("--list", action="store_true")
args = p.parse_args()
names = list(EXPERIMENTS) if args.exp == "all" else args.exp.split(",")


def build(name):
    fn = EXPERIMENTS[name]
    return fn(args.out, RESULTS_DIR) if name == "supplementary" else fn(args.out)


if args.list:
    for n in EXPERIMENTS:
        print("%-14s %5d runs   %s" % (n, len(build(n)), (EXPERIMENTS[n].__doc__ or "").strip().split("\n")[0]))
    sys.exit(0)

jobs, seen = [], set()
for n in names:
    for j in build(n):
        k = (j["csv"], j["key"], tuple(j["cmd"]) if j["key"] is None else None)
        if k in seen:
            continue
        seen.add(k)
        if j["key"] is not None and is_done(j["csv"], *j["key"]):
            continue
        j["cmd"][0] = args.python_copernicus if j["env"] == "copernicus" else args.python_dofa
        jobs.append(j)
print("%d runs to do" % len(jobs), flush=True)
if args.dry_run:
    for j in jobs:
        print(" ".join(j["cmd"]))
    sys.exit(0)

logs = args.logs or os.path.join(args.out, "logs")
os.makedirs(logs, exist_ok=True)
gpu_q = deque(j for j in jobs if j["gpu"])
cpu_q = deque(j for j in jobs if not j["gpu"])
lock, failed = threading.Lock(), []


def worker(queue, gpu):
    while True:
        with lock:
            if not queue:
                return
            j = queue.popleft()
        tag = "_".join(a.replace("/", "-").replace(":", "") for a in j["cmd"][2:] if not a.startswith("--") and len(a) < 40)
        env = dict(os.environ, PYTHONPATH=ROOT)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        t0 = time.time()
        with open(os.path.join(logs, tag[:150] + ".log"), "w") as log:
            rc = subprocess.call(j["cmd"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        with lock:
            print("[%s] %s  %.0f min  %s" % ("ok" if rc == 0 else "FAILED", "gpu %s" % gpu if gpu else "cpu",
                                              (time.time() - t0) / 60, " ".join(j["cmd"][1:])), flush=True)
            if rc:
                failed.append(j)


threads = [threading.Thread(target=worker, args=(gpu_q, g)) for g in args.gpus.split(",") for _ in range(args.per_gpu)]
threads += [threading.Thread(target=worker, args=(cpu_q, None)) for _ in range(args.cpu_jobs)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("finished; %d failed (see %s)" % (len(failed), logs))
sys.exit(1 if failed else 0)
