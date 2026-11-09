# More bands, larger tokens

Code and per-run results for

> **More bands, larger tokens: band-count scaling in wavelength-conditioned foundation models and its correction for
> few-label hyperspectral classification.**
> Jeslin P James, Shailesh S. Department of Computer Science, Cochin University of Science and Technology (CUSAT),
> Kochi, India. Submitted to *ISPRS Journal of Photogrammetry and Remote Sensing*.
>
> Contact: Jeslin P James (jeslin@ug.cusat.ac.in), corresponding author.

Wavelength-conditioned foundation models such as DOFA and Copernicus-FM generate one patch-embedding kernel per band
from its wavelength and **sum** the per-band responses. Nothing divides by the number of bands N, and the band responses
are weakly but consistently aligned, so the patch-token norm grows almost linearly with N: with the 200 bands of a
hyperspectral scene, tokens are more than ten times larger than with the few bands of most pretraining sensors.
Two corrections without trainable parameters remove this dependence and leave inputs with at most 63 bands unchanged:

* **BCR**, band-count rescaling: multiply the input by `min(1, 63/N)`;
* **GTN**, gated token normalisation: layer normalisation without learned parameters of every patch token when `N > 63`.

```python
from bcs import corrections
corrections.patch_dofa("gtn")      # or "bcr"; call once before using a DOFA model (DOFA code on sys.path)
```

For Copernicus-FM, wrap the TorchGeo model: `bcs.copernicus.Corrected(model, "gtn")`.

## Contents

```
bcs/            library: corrections, splits, data loading, wavelengths, DOFA and Copernicus-FM set-up
scripts/        training (DOFA, Copernicus-FM), baselines, token measurement, experiment launcher, data check
analysis/       every number, table and figure of the paper and supplement, from the per-run results
results/        one CSV row per training run (about 2 000 runs); results/unreported: exploratory runs not in the paper
docs/           DATA.md (datasets), REPRODUCE.md (paper item -> experiment -> file -> analysis output)
environment/    requirements and exact package versions of the two environments
legacy/         the original scripts with which the runs were made, unchanged, for provenance
tests/          unit tests of the corrections and splits (CPU, no data needed)
```

## Quick start

```bash
git clone https://github.com/jeslinpjames/hsi-band-count-scaling.git && cd hsi-band-count-scaling

# 1. environment (Python 3.8 for DOFA; see environment/README.md for Copernicus-FM)
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r environment/requirements-dofa.txt

# 2. recompute all numbers, tables and figures from the included per-run results (no GPU)
python analysis/stats.py && python analysis/tables.py && python analysis/make_figs.py

# 3. DOFA code (pinned revision), DOFA weights and datasets for new runs
bash scripts/setup_third_party.sh
python scripts/check_data.py                    # after placing the scenes as described in docs/DATA.md

# 4. one run: Indian Pines, Generic LoRA, GTN, split 0
python scripts/train_dofa.py --dataset ip --method generic --correction gtn --seed 0 --csv results/new/demo.csv

# 5. any experiment of the paper, on several GPUs
python scripts/run_experiments.py --list
python scripts/run_experiments.py --exp main --gpus 0,1,2,3 --out results/new
```

See `docs/REPRODUCE.md` for the experiment behind every table and figure, and the settings used.

## Protocols

* **Random split:** 10 training and 5 validation pixels per class, all other labelled pixels are test pixels;
  16 paired splits (seeds 0–15).
* **Spatially separated split:** training and validation pixels come from a few 32×32 tiles; test pixels lie at least
  33 pixels from every training and validation pixel, so no 33×33 test window overlaps a training window
  (Pavia University and Salinas, seeds 0–7). In file names this split is called `disj`/`disjoint`.

## Citation

```bibtex
@article{james2026morebands,
  title   = {More bands, larger tokens: band-count scaling in wavelength-conditioned foundation models and its
             correction for few-label hyperspectral classification},
  author  = {James, Jeslin P and S, Shailesh},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  note    = {submitted},
  year    = {2026}
}
```

## Licence

Code: MIT (see `LICENSE`). The per-run results in `results/` are released under CC BY 4.0. The datasets, the DOFA
code and weights, and Copernicus-FM keep their own licences and are not redistributed here.

## Acknowledgements

DOFA (Xiong et al., 2024), Copernicus-FM (Wang et al., 2025) via TorchGeo, the EHU hyperspectral scenes, and the
cross-scene releases of Zhang et al. (2023).
