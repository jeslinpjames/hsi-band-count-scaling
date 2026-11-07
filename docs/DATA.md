# Data

Place the scenes under `data/` (or set `BCS_DATA` to another folder) with this layout:

```
data/
├── IndianPines/Indian_pines_corrected.mat     Indian_pines_gt.mat
├── Salinas/Salinas_corrected.mat              Salinas_gt.mat
├── PaviaU/paviaU.mat                          paviaU_7gt.mat
└── Houston/Houston13.mat                      Houston13_7gt.mat
```

| Scene | Source | Notes |
|---|---|---|
| Indian Pines | [EHU Hyperspectral Remote Sensing Scenes](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) (original: Baumgardner et al., 2015, Purdue University Research Repository, doi:10.4231/R7RX991C) | 200 bands (water-absorption bands removed), 16 classes. The paper used the copy distributed with HyperSIGMA (keys `data`, `groundT`); the EHU files (keys `indian_pines_corrected`, `indian_pines_gt`) are also accepted. |
| Salinas | [EHU Hyperspectral Remote Sensing Scenes](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) | `Salinas_corrected.mat`, 204 bands, 16 classes. |
| Pavia University | [Data-CSHSI](https://github.com/YuxiangZhang-BIT/Data-CSHSI) (Zhang et al., 2023), "Pavia datasets" | Cross-scene release: 102 bands, **7 classes**. This differs from the usual 103-band, 9-class benchmark. |
| Houston 2013 | [Data-CSHSI](https://github.com/YuxiangZhang-BIT/Data-CSHSI), "Houston datasets" | Cross-scene release: 48 bands, 7 classes (MATLAB v7.3 files). Used only for the test with few bands (Table 4). |

Check the files with

```bash
python scripts/check_data.py
```

which compares shapes, class counts and the number of labelled pixels with those of the paper
(Indian Pines 145×145×200, 10 249 labelled; Salinas 512×217×204, 54 129; Pavia University 610×340×102, 39 332;
Houston 2013 954×210×48, 2 530).

**Wavelengths.** Band-centre wavelengths are approximated by uniform spacing over each sensor's nominal range, with
the removed AVIRIS bands omitted (`bcs/wavelengths.py`). No calibration files were used; this is a stated limitation
of the paper. Copernicus-FM additionally receives nominal bandwidths (10 nm for AVIRIS, 4 nm for ROSIS).

**Preprocessing.** Every band is standardised over the whole scene (mean 0, standard deviation 1). The foundation
models see the 33×33 window around each labelled pixel, zero-padded at the scene border and bilinearly upsampled to
224×224.
