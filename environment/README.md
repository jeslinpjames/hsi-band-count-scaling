# Environments

Two Python environments were used, because TorchGeo 0.7.0 (Copernicus-FM) needs Python 3.10 or newer while the DOFA
experiments ran in an existing Python 3.8 environment.

| Environment | Used for | Python | Key packages |
|---|---|---|---|
| `dofa` | DOFA training, token measurement, classical and deep baselines, analysis | 3.8.10 | torch 2.4.1+cu118, peft 0.13.2, timm 1.0.29 |
| `copernicus` | Copernicus-FM training and token measurement | 3.11.18 | torch 2.4.1+cu118, torchgeo 0.7.0, peft 0.13.2 |

Hardware: NVIDIA A100-SXM4-80GB (driver 535.183.06); one run uses one GPU.

```bash
# DOFA environment
python3.8 -m venv envs/dofa && source envs/dofa/bin/activate
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r environment/requirements-dofa.txt

# Copernicus-FM environment (e.g. with uv, which can install Python 3.11 without root access)
uv venv --python 3.11 envs/copernicus && source envs/copernicus/bin/activate
uv pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
uv pip install -r environment/requirements-copernicus.txt
```

`freeze-*.txt` list every package of the two original environments (`pip freeze`). The analysis scripts (`analysis/`)
only need numpy, scipy, pandas and matplotlib and run in either environment.
