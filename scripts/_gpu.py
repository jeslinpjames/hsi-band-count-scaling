"""Pick the GPU with the most free memory unless CUDA_VISIBLE_DEVICES is already set (useful on shared servers)."""
import os
import subprocess


def pick_free_gpu():
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                                      timeout=30).decode().strip().splitlines()
        os.environ["CUDA_VISIBLE_DEVICES"] = max((int(l.split(",")[1]), l.split(",")[0].strip()) for l in out if l.strip())[1]
    except Exception:
        pass
