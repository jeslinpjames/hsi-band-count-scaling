"""Every experiment of the paper as a list of runs.

Each run is one command line; its results go to a CSV whose file name matches the corresponding file in results/,
so that the analysis can be pointed at freshly generated results (BCS_RESULTS=<dir> python analysis/stats.py).

python scripts/run_experiments.py --list   prints every experiment with its number of runs.
"""
import os

PY = "python"
T_DOFA = "scripts/train_dofa.py"
LRS = {"generic": "6e-4", "spectral": "2e-4", "full": "2e-5"}           # selected learning rates (Section 4.3)
RANK = {"generic": 32, "spectral": 4, "full": 8}
TAG = {"none": "nofix", "bcr": "fix63", "gtn": "tokenln", "sqrt": "sqrt63", "ref9": "ref9", "ln": "tokenln"}
VAL = {5: 3, 10: 5, 20: 5, 50: 10, 100: 20}                             # validation pixels per class for each budget


def dofa_run(out, csv, dataset, method, correction, seed, lr, split="random", bands="all", r=None, n=10,
             max_epoch=400, patience=100):
    r = RANK.get(method, 8) if r is None else r
    cmd = [PY, T_DOFA, "--dataset", dataset, "--method", method, "--correction", correction, "--split", split,
           "--bands", bands, "--seed", str(seed), "--lr", lr, "--lora_r", str(r), "--lora_alpha", str(2 * r),
           "--train_num", str(n), "--val_num", str(VAL[n]), "--max_epoch", str(max_epoch), "--patience", str(patience),
           "--csv", os.path.join(out, csv)]
    return dict(cmd=cmd, csv=os.path.join(out, csv), key=(method, n, seed, r), env="dofa", gpu=True)


def exp_main(out):
    """Table 2, Fig. 4, Fig. 6 and Section 5.5: three scenes x three methods x {none, BCR, GTN}, 16 splits."""
    runs = []
    for ds in ("ip", "salinas", "pavia"):
        for m, lr in LRS.items():
            for c in ("none", "bcr", "gtn"):
                stem = "results_fixvar" if c == "gtn" else "results_lr"
                for s in range(16):
                    runs.append(dofa_run(out, "%s_%s_%s_lr%s.csv" % (stem, ds, TAG[c], lr), ds, m, c, s, lr))
    return runs


def exp_lr_selection(out):
    """Supplementary Section S2: learning-rate grid on Indian Pines and Pavia, with and without BCR, splits 0-7."""
    grid = {"generic": ("6e-5", "2e-4", "6e-4", "2e-3"), "spectral": ("6e-5", "2e-4", "6e-4", "2e-3"),
            "full": ("2e-5", "6e-5", "2e-4")}
    return [dofa_run(out, "results_lr_%s_%s_lr%s.csv" % (ds, TAG[c], lr), ds, m, c, s, lr)
            for ds in ("ip", "pavia") for c in ("none", "bcr") for m, lrs in grid.items() for lr in lrs for s in range(8)]


def exp_alternatives(out):
    """Fig. 5: sqrt(63/N) and 9/N rescaling, Generic LoRA and full fine-tuning, splits 0-7."""
    return [dofa_run(out, "results_fixvar_%s_%s_lr%s.csv" % (ds, TAG[c], LRS[m]), ds, m, c, s, LRS[m])
            for ds in ("ip", "pavia") for c in ("sqrt", "ref9") for m in ("generic", "full") for s in range(8)]


def exp_ungated(out):
    """Table 4: ungated token normalisation on inputs with few bands (25-band Indian Pines, 48-band Houston)."""
    runs = []
    for name, ds, bands in (("ip25", "ip", "uniform:25"), ("houston", "houston13", "all")):
        for c in ("none", "ln"):
            for m in ("generic", "full"):
                for s in range(8):
                    runs.append(dofa_run(out, "results_fixvar_%s_%s_lr%s.csv" % (name, TAG[c], LRS[m]), ds, m, c, s,
                                         LRS[m], bands=bands))
    return runs


def exp_separated(out):
    """Table 5, Fig. 7: spatially separated split, Pavia and Salinas, splits 0-7."""
    return [dofa_run(out, "results_disj_%s_%s_lr%s.csv" % (ds, TAG[c], LRS[m]), ds, m, c, s, LRS[m], split="separated")
            for ds in ("pavia", "salinas") for c in ("none", "bcr", "gtn") for m in ("generic", "full") for s in range(8)]


def exp_copernicus(out):
    """Table 3: Copernicus-FM with Generic LoRA, three scenes x {none, BCR, GTN}, splits 0-7."""
    runs = []
    for ds in ("ip", "salinas", "pavia"):
        for c in ("none", "bcr", "gtn"):
            for s in range(8):
                csv = os.path.join(out, "results_cfm_random_%s_%s.csv" % (ds, c))
                runs.append(dict(cmd=[PY, "scripts/train_copernicus.py", "--dataset", ds, "--correction", c, "--seed",
                                      str(s), "--csv", csv], csv=csv, key=("generic", 10, s, 32), env="copernicus", gpu=True))
    return runs


def exp_classic(out):
    """Table 2 (classical rows) and Table 5 (SVM rows). CPU only."""
    runs = []
    for ds in ("ip", "salinas", "pavia"):
        csv = os.path.join(out, "results_classic_%s.csv" % ds)
        runs.append(dict(cmd=[PY, "scripts/classic_baselines.py", "--dataset", ds, "--seeds", "0-15", "--csv", csv],
                         csv=csv, key=None, env="dofa", gpu=False))
    for ds in ("pavia", "salinas"):
        csv = os.path.join(out, "results_classic_disjoint_%s.csv" % ds)
        runs.append(dict(cmd=[PY, "scripts/classic_baselines.py", "--dataset", ds, "--split", "separated", "--methods",
                              "svm,svm_sp33", "--seeds", "0-7", "--csv", csv], csv=csv, key=None, env="dofa", gpu=False))
    return runs


def exp_deep(out):
    """Table 2 (HybridSN, SSRN) and Table 5."""
    runs = []
    for model in ("hybridsn", "ssrn"):
        for ds in ("ip", "salinas", "pavia"):
            csv = os.path.join(out, "results_deep_random_%s.csv" % ds)
            runs.append(dict(cmd=[PY, "scripts/deep_baselines.py", "--model", model, "--dataset", ds, "--seeds", "0-15",
                                  "--csv", csv], csv=csv, key=None, env="dofa", gpu=True))
        for ds in ("pavia", "salinas"):
            csv = os.path.join(out, "results_deep_disjoint_%s.csv" % ds)
            runs.append(dict(cmd=[PY, "scripts/deep_baselines.py", "--model", model, "--dataset", ds, "--split",
                                  "separated", "--seeds", "0-7", "--csv", csv], csv=csv, key=None, env="dofa", gpu=True))
    return runs


def exp_tokens(out):
    """Fig. 2 and Supplementary Sections S3 and S10: forward passes only."""
    return [dict(cmd=[PY, "scripts/measure_tokens_dofa.py", "--out", out], csv=os.path.join(out, "coherence.csv"),
                 key=None, env="dofa", gpu=True),
            dict(cmd=[PY, "scripts/measure_tokens_copernicus.py", "--out", out], csv=os.path.join(out, "cfm_coherence.csv"),
                 key=None, env="copernicus", gpu=True)]


# --------------------------------------------------------------------------------------------------------------
# Supplementary experiments run earlier in the project, before the learning rates were selected: learning rate
# 6e-5 (the training script's default at the time), alpha = 2 x rank, at most 400 epochs, patience 100. Used in
# Supplementary Sections S5 (Spectral vs Generic LoRA, label budgets) and S6 (band subsets, other baselines).
# For each file, the configurations (method, rank, labels per class, seeds) are exactly those in results/.
SUPP = {   # csv -> (dataset, bands, correction)
    "results_baselines.csv": ("ip", "all", "none"), "results_baselines_rep.csv": ("ip", "all", "none"),
    "results_ip_n10_rep.csv": ("ip", "all", "none"), "results_ip_labelcurve.csv": ("ip", "all", "none"),
    "results_rank.csv": ("ip", "all", "none"), "results_rank_patience100.csv": ("ip", "all", "none"),
    "results_rank_n50_100.csv": ("ip", "all", "none"), "results_rank_n50_r4r32.csv": ("ip", "all", "none"),
    "results_ip_bnref63.csv": ("ip", "all", "bcr"), "results_ip_bnsafe.csv": ("ip", "all", "bcr"),
    "results_ip_bnsafe_full.csv": ("ip", "all", "bcr"),
    "results_ip_uniform25.csv": ("ip", "uniform:25", "none"), "results_ip_uniform50.csv": ("ip", "uniform:50", "none"),
    "results_ip_uniform63.csv": ("ip", "uniform:63", "none"), "results_ip_uniform100.csv": ("ip", "uniform:100", "none"),
    "results_ip_uniform150.csv": ("ip", "uniform:150", "none"), "results_ip_vnir.csv": ("ip", "vnir", "none"),
    "results_ip_u100_bnsafe.csv": ("ip", "uniform:100", "bcr"), "results_ip_u150_bnsafe.csv": ("ip", "uniform:150", "bcr"),
}


def exp_supplementary(out, reference_dir):
    """Supplementary Sections S5 and S6: earlier runs (learning rate 6e-5), configurations read from results/."""
    import csv as _csv
    runs = []
    for name, (ds, bands, c) in SUPP.items():
        with open(os.path.join(reference_dir, name)) as f:
            for row in _csv.DictReader(f):
                m, n, s, r = row["method"], int(row["train_num"]), int(row["seed"]), int(row["lora_r"])
                # the rank-1 runs of results_rank.csv were made with the initial setting (300 epochs, patience 50)
                me, pa = (300, 50) if (name == "results_rank.csv" and r == 1) else (400, 100)
                runs.append(dofa_run(out, name, ds, m, c, s, "6e-5", bands=bands, r=r, n=n, max_epoch=me, patience=pa))
    return runs


EXPERIMENTS = {"main": exp_main, "lr_selection": exp_lr_selection, "alternatives": exp_alternatives,
               "ungated": exp_ungated, "separated": exp_separated, "copernicus": exp_copernicus,
               "classic": exp_classic, "deep": exp_deep, "tokens": exp_tokens, "supplementary": exp_supplementary}
