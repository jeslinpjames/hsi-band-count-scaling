#!/bin/bash
# Fetches the DOFA code at the revision used in the paper and the DOFA weights.
# usage:  bash scripts/setup_third_party.sh            (run from the repository root)
set -e
mkdir -p third_party weights
if [ ! -d third_party/DOFA ]; then
  git clone https://github.com/zhu-xlab/DOFA.git third_party/DOFA
fi
git -C third_party/DOFA checkout 0cfb7e1099f4d4c4022946ff7862c7cd7b8411b9
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ("DOFA_ViT_base_e100.pth", "DOFA_ViT_large_e100.pth"):
    print(hf_hub_download(repo_id="earthflow/DOFA", filename=f, local_dir="weights"))
PY
echo "DOFA code in third_party/DOFA, weights in weights/"
