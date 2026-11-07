# Third-party code

`scripts/setup_third_party.sh` clones the DOFA repository (https://github.com/zhu-xlab/DOFA, MIT licence) into
`third_party/DOFA` and checks out revision `0cfb7e1099f4d4c4022946ff7862c7cd7b8411b9`, the one used for every run in
the paper. The DOFA code is used unmodified; the corrections are applied at run time by `bcs/corrections.py`.
Set `DOFA_REPO` to use a copy elsewhere.

Copernicus-FM comes from TorchGeo 0.7.0 (MIT licence), installed with pip; its weights are downloaded by TorchGeo on
first use (set `CFM_CKPT` to a local `.pth` file on machines without internet access).
