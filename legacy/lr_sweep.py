"""lr_sweep.py : learning-rate sweep for the reviewer issue "one learning rate (6e-5) for all methods".

What it runs (seeds 0-7, ten labels per class, same settings as the paper except the learning rate):
  datasets   : Indian Pines (200 bands), Pavia University (102 bands)
  conditions : no fix, safe fix REF 63
  methods    : Generic LoRA r32/a64 and Spectral LoRA r4/a8 at lr 6e-5, 2e-4, 6e-4, 2e-3
               full fine-tuning                         at lr 2e-5, 6e-5, 2e-4
  = 2 x 2 x (2*4 + 3) x 8 = 352 runs.  (6e-5 is re-run on purpose: it checks the old numbers reproduce,
    and it gives the validation accuracy needed to choose the learning rate fairly.)

Safety for the shared server (same rules as batch E):
  - at most MAX parallel runs (file lr_max_runs.txt, default 6, change it any time)
  - 8 CPU threads per run
  - a run only starts on a GPU with enough free memory (counting runs that just started)
  - a run only starts if, after it, at least 250 GB RAM stays free for others
  - no new runs while the 1-minute load average is above 200
  - runs that crash (e.g. out of memory) are put back in the queue, at most 3 tries
  - touch STOP_LR  -> no new runs start; running ones finish; then the launcher exits

Usage (inside tmux):  python lr_sweep.py            # start or resume (finished runs are skipped)
                      python lr_sweep.py --dry-run  # only print the job list and pre-flight checks
Progress:             cat lr_status.txt
Results:              results_lr_<dataset>_<cond>_lr<lr>.csv  (same columns as before)
                      lr_runs.csv  (one line per finished run incl. best validation accuracy)
"""
import os, re, sys, time, glob, json, shutil, subprocess, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--seeds", default="0-7")
ap.add_argument("--plan", default="sweep", choices=["sweep", "final", "fixvar", "lncheck", "lnfinal", "disjoint"],
                help="sweep = the 352-run learning-rate sweep; final = tuned rates, IP/Pavia seeds 8-15 + Salinas seeds 0-15; "
                     "fixvar = alternative fixes (sqrt63, ref9, tokenln) at tuned rates, IP + Pavia, seeds 0-7; "
                     "lncheck = token-LayerNorm safety test (25-band IP, 48-band Houston), Salinas, and IP/Pavia seeds 8-15; "
                     "lnfinal = token LayerNorm: Salinas seeds 8-15 (Generic, Full) + Spectral LoRA on IP/Pavia/Salinas seeds 0-15; "
                     "disjoint = spatially disjoint split, Pavia + Salinas, Generic/Full x none/BCR/GTN, seeds 0-7")
args = ap.parse_args()
a, b = args.seeds.split("-")
SEEDS = list(range(int(a), int(b) + 1))

# ------------------------------------------------------------------ job definition
SCRIPTS = {("ip", "nofix"): "train_dofa.py",
           ("ip", "fix63"): "train_dofa_bnsafe.py",
           ("pavia", "nofix"): "train_dofa_pavia.py",
           ("pavia", "fix63"): "train_dofa_pavia_bnsafe.py",
           ("salinas", "nofix"): "train_dofa_salinas.py",
           ("salinas", "fix63"): "train_dofa_salinas_bnsafe.py"}
METHODS = {"generic": dict(r=32, alpha=64, lrs=["6e-5", "2e-4", "6e-4", "2e-3"]),
           "spectral": dict(r=4, alpha=8, lrs=["6e-5", "2e-4", "6e-4", "2e-3"]),
           "full": dict(r=8, alpha=16, lrs=["2e-5", "6e-5", "2e-4"])}
NEED_RAM_GB = {"ip": 45, "pavia": 115, "salinas": 150, "ip25": 25, "houston": 70}          # measured peaks: Pavia ~105 GB; IP ~20-35 GB
if os.environ.get("LR_TEST"): NEED_RAM_GB = {k: 0.1 for k in NEED_RAM_GB}
NEED_GPU_GB = {"generic": 12, "spectral": 12, "full": 17}
RESERVE_RAM_GB = float(os.environ.get("LR_RESERVE_GB", "250"))
LOAD_LIMIT = float(os.environ.get("LR_LOAD_LIMIT", "200"))
THREADS = 8
MAX_TRIES = 3

def csv_name(ds, cond, lr):
    if args.plan == "disjoint":
        return "results_disj_%s_%s_lr%s.csv" % (ds, cond, lr)
    if args.plan in ("fixvar", "lncheck", "lnfinal"):
        return "results_fixvar_%s_%s_lr%s.csv" % (ds, cond, lr)
    return "results_lr_%s_%s_lr%s.csv" % (ds, cond, lr)

TUNED = {"generic": "6e-4", "spectral": "2e-4", "full": "2e-5"}   # one rate per method, chosen by pooled validation
if args.plan == "final":
    for m in METHODS:
        METHODS[m] = dict(METHODS[m], lrs=[TUNED[m]])
    PLAN = [("salinas", list(range(0, 16))), ("ip", list(range(8, 16))), ("pavia", list(range(8, 16)))]
    CONDS = ("nofix", "fix63")
elif args.plan == "fixvar":
    VARIANTS = ["sqrt63", "ref9", "tokenln"]
    METHODS = {m: dict(METHODS[m], lrs=[TUNED[m]]) for m in ("generic", "full")}
    PLAN = [("ip", SEEDS), ("pavia", SEEDS)]
    CONDS = tuple(VARIANTS)
    for v in VARIANTS:
        SCRIPTS[("ip", v)] = "train_dofa_fixvar.py"
        SCRIPTS[("pavia", v)] = "train_dofa_pavia_fixvar.py"
elif args.plan == "lncheck":
    METHODS = {m: dict(METHODS[m], lrs=[TUNED[m]]) for m in ("generic", "full")}
    # (dataset label, condition, script, extra environment, seeds)
    SPEC = [("ip25", "nofix", "train_dofa_ip_u25.py", {"NBANDS": "25"}, range(8)),
            ("ip25", "tokenln", "train_dofa_ip_u25_fixvar.py", {"NBANDS": "25", "FIX_VARIANT": "tokenln"}, range(8)),
            ("houston", "nofix", "train_dofa_houston13.py", {}, range(8)),
            ("houston", "tokenln", "train_dofa_houston13_fixvar.py", {"FIX_VARIANT": "tokenln"}, range(8)),
            ("salinas", "tokenln", "train_dofa_salinas_fixvar.py", {"FIX_VARIANT": "tokenln"}, range(8)),
            ("ip", "tokenln", "train_dofa_fixvar.py", {"FIX_VARIANT": "tokenln"}, range(8, 16)),
            ("pavia", "tokenln", "train_dofa_pavia_fixvar.py", {"FIX_VARIANT": "tokenln"}, range(8, 16))]
    PLAN = [(d, list(sd)) for d, _, _, _, sd in SPEC]
    CONDS = ("nofix", "tokenln")
    for d, c, sc, _, _ in SPEC:
        SCRIPTS[(d, c)] = sc
elif args.plan == "lnfinal":
    METHODS = {m: dict(METHODS[m], lrs=[TUNED[m]]) for m in ("generic", "spectral", "full")}
    LN = {"FIX_VARIANT": "tokenln"}
    # (dataset label, condition, script, extra environment, seeds, methods)
    SPEC = [("salinas", "tokenln", "train_dofa_salinas_fixvar.py", LN, range(8, 16), ("generic", "full")),
            ("ip", "tokenln", "train_dofa_fixvar.py", LN, range(16), ("spectral",)),
            ("pavia", "tokenln", "train_dofa_pavia_fixvar.py", LN, range(16), ("spectral",)),
            ("salinas", "tokenln", "train_dofa_salinas_fixvar.py", LN, range(16), ("spectral",))]
    PLAN = [(x[0], list(x[4])) for x in SPEC]
    CONDS = ("tokenln",)
    for d, c, sc, *_ in SPEC:
        SCRIPTS[(d, c)] = sc
elif args.plan == "disjoint":
    METHODS = {m: dict(METHODS[m], lrs=[TUNED[m]]) for m in ("generic", "full")}
    DJ = {"SPLIT_MODE": "disjoint"}
    SPEC = []
    for d in ("pavia", "salinas"):
        SPEC += [(d, "nofix", "train_dofa_%s_disj.py" % d, dict(DJ), range(8), ("generic", "full")),
                 (d, "fix63", "train_dofa_%s_fixvar_disj.py" % d, dict(DJ, FIX_VARIANT="safe63"), range(8), ("generic", "full")),
                 (d, "tokenln", "train_dofa_%s_fixvar_disj.py" % d, dict(DJ, FIX_VARIANT="tokenln"), range(8), ("generic", "full"))]
    PLAN = [(x[0], list(x[4])) for x in SPEC]
    CONDS = ("nofix", "fix63", "tokenln")
    for d, c, sc, *_ in SPEC:
        SCRIPTS[(d, c)] = sc
else:
    PLAN = [("ip", SEEDS), ("pavia", SEEDS)]
    CONDS = ("nofix", "fix63")
jobs = []
if args.plan in ("lncheck", "lnfinal", "disjoint"):
    for row in SPEC:
        d, cond, sc, env, seeds = row[:5]
        meths = row[5] if len(row) > 5 else tuple(METHODS)
        for seed in seeds:
            for m, spec in METHODS.items():
                if m not in meths:
                    continue
                lr = spec["lrs"][0]
                tag = "%s_%s_%s_lr%s_s%d" % (d, cond, m, lr, seed)
                cmd = [sys.executable, "-u", sc, "--method", m, "--train_num", "10", "--val_num", "5",
                       "--seed", str(seed), "--lr", lr, "--batch_size", "16", "--max_epoch", "400", "--patience", "100",
                       "--lora_r", str(spec["r"]), "--lora_alpha", str(spec["alpha"]), "--csv", csv_name(d, cond, lr)]
                jobs.append(dict(tag=tag, ds=d, cond=cond, method=m, r=spec["r"], lr=lr, seed=seed,
                                 csv=csv_name(d, cond, lr), cmd=cmd, tries=0, env=dict(env)))
    PLAN = []                                   # jobs already built
for ds, seeds in PLAN:                          # first dataset first; later ones fill spare RAM (not used by lncheck)
    for seed in seeds:
        for cond in CONDS:
            for m, spec in METHODS.items():
                for lr in spec["lrs"]:
                    tag = "%s_%s_%s_lr%s_s%d" % (ds, cond, m, lr, seed)
                    cmd = [sys.executable, "-u", SCRIPTS[(ds, cond)], "--method", m, "--train_num", "10",
                           "--val_num", "5", "--seed", str(seed), "--lr", lr, "--batch_size", "16",
                           "--max_epoch", "400", "--patience", "100", "--lora_r", str(spec["r"]),
                           "--lora_alpha", str(spec["alpha"]), "--csv", csv_name(ds, cond, lr)]
                    jobs.append(dict(tag=tag, ds=ds, cond=cond, method=m, r=spec["r"], lr=lr, seed=seed,
                                     csv=csv_name(ds, cond, lr), cmd=cmd, tries=0,
                                     env={"FIX_VARIANT": cond} if args.plan == "fixvar" else {}))

def done_keys(csv):
    keys = set()
    if os.path.exists(csv):
        with open(csv) as f:
            head = f.readline().strip().split(",")
            try:
                im, isd, ir = head.index("method"), head.index("seed"), head.index("lora_r")
            except ValueError:
                return keys
            for line in f:
                p = line.strip().split(",")
                if len(p) > max(im, isd, ir):
                    keys.add((p[im], int(float(p[isd])), int(float(p[ir]))))
    return keys

def is_done(j):
    return (j["method"], j["seed"], j["r"]) in done_keys(j["csv"])

# ------------------------------------------------------------------ pre-flight checks
problems = []
for (ds, cond), sc in SCRIPTS.items():
    if args.plan in ("lncheck", "lnfinal", "disjoint"):
        if (ds, cond) not in [(x[0], x[1]) for x in SPEC]:
            continue
    elif ds not in [x for x, _ in PLAN] or cond not in CONDS:
        continue
    if not os.path.exists(sc):
        problems.append("missing script %s" % sc); continue
    src = open(sc).read()
    for flag in ("--lr", "--csv", "--patience", "--max_epoch", "--val_num", "--lora_alpha"):
        if "'%s'" % flag not in src and '"%s"' % flag not in src:
            problems.append("%s does not define %s" % (sc, flag))
    if args.plan == "disjoint" and "_sm.wrap(" not in src:
        problems.append("%s is not a patched _disj script (run patch_split.py)" % sc)
    if cond == "fix63" and args.plan != "disjoint" and "bandnorm_safe_patch" not in src:
        problems.append("%s does not import bandnorm_safe_patch" % sc)
    if args.plan in ("fixvar", "lncheck", "lnfinal", "disjoint") and cond != "nofix" and ("import fix_variants_patch" not in src or "bandnorm_safe_patch" in src):
        problems.append("%s must import fix_variants_patch (and not bandnorm_safe_patch)" % sc)
for f in (["split_modes.py", "data_common.py"] if args.plan == "disjoint" else []):
    if not os.path.exists(f):
        problems.append("%s is missing" % f)
if args.plan in ("fixvar", "lncheck", "lnfinal", "disjoint") and not os.path.exists("fix_variants_patch.py"):
    problems.append("fix_variants_patch.py is missing")
if os.environ.get("BANDNORM_REF", "63") != "63":
    problems.append("environment variable BANDNORM_REF is set to %s; unset it" % os.environ["BANDNORM_REF"])
todo = [j for j in jobs if not is_done(j)]
print("pre-flight: %d jobs defined, %d already finished, %d to run" % (len(jobs), len(jobs) - len(todo), len(todo)))
for p in problems:
    print("  PROBLEM:", p)
if args.dry_run:
    for j in todo[:6]:
        print("  e.g.", " ".join(j["cmd"][1:]))
    print("dry run only - nothing started"); sys.exit(1 if problems else 0)
if problems:
    sys.exit("fix the problems above first")
os.makedirs("logs", exist_ok=True)

# ------------------------------------------------------------------ resources
def free_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
        res = {}
        for line in out.strip().splitlines():
            i, tot, used = [int(x) for x in line.split(",")]
            res[i] = (tot - used) / 1024.0
        return res
    except Exception:
        return {}

def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1048576.0
    return 0.0

def rss_gb(pid):
    try:
        with open("/proc/%d/status" % pid) as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1048576.0
    except Exception:
        pass
    return 0.0

def max_runs():
    try:
        return max(0, int(open("lr_max_runs.txt").read().strip()))
    except Exception:
        return 6

# ------------------------------------------------------------------ bookkeeping
running = {}        # tag -> dict(job, proc, gpu, start, log)
durations = {}      # (ds, method) -> list of minutes
queue = [j for j in jobs if not is_done(j)]
finished_now, failed = 0, []

def last_epoch_info(log):
    try:
        with open(log, "rb") as f:
            f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 6000))
            txt = f.read().decode(errors="ignore")
        m = re.findall(r"epoch\s+(\d+).*?best\s+([0-9.]+)@(\d+)", txt)
        return m[-1] if m else None
    except Exception:
        return None

def record_run(j, log, minutes):
    info = last_epoch_info(log)
    best_val = float(info[1]) if info else float("nan")
    oa = ""
    if os.path.exists(j["csv"]):
        with open(j["csv"]) as f:
            head = f.readline().strip().split(",")
            for line in f:
                p = line.strip().split(",")
                if p[head.index("method")] == j["method"] and int(float(p[head.index("seed")])) == j["seed"] \
                        and int(float(p[head.index("lora_r")])) == j["r"]:
                    oa = p[head.index("OA")]
    new = not os.path.exists("lr_runs.csv")
    with open("lr_runs.csv", "a") as f:
        if new:
            f.write("dataset,condition,method,lora_r,lr,seed,best_val,test_OA,minutes\n")
        f.write("%s,%s,%s,%d,%s,%d,%.4f,%s,%.1f\n" % (j["ds"], j["cond"], j["method"], j["r"], j["lr"],
                                                     j["seed"], best_val, oa, minutes))

def write_status():
    now = time.time()
    total = len(jobs); done = sum(1 for j in jobs if is_done(j))
    lines = ["LR SWEEP STATUS  %s" % time.strftime("%a %d %b %H:%M"),
             "done %d / %d  (%.0f%%) | running %d (limit %d) | waiting %d | failed for good %d" % (
                 done, total, 100.0 * done / total, len(running), max_runs(), len(queue), len(failed)),
             "RAM available %.0f GB | load %.0f | STOP_LR %s" % (
                 mem_available_gb(), os.getloadavg()[0], "present (stopping)" if os.path.exists("STOP_LR") else "absent"),
             "", "running:"]
    for tag, r in sorted(running.items(), key=lambda kv: kv[1]["start"]):
        info = last_epoch_info(r["log"])
        lines.append("  GPU%d  %-42s %5.0f min  %s" % (r["gpu"], tag, (now - r["start"]) / 60,
                     ("epoch %s, best val %s@%s" % info) if info else "starting"))
    est = 0.0
    for j in queue:
        d = durations.get((j["ds"], j["method"]))
        est += (sum(d) / len(d)) if d else (45 if j["ds"] == "ip" else 15)
    par = max(1, min(max_runs(), len(running) or 1))
    lines.append("")
    lines.append("rough time left: %.1f h (queue %.0f run-hours / %d in parallel, plus runs in progress)" % (
        est / 60 / par, est / 60, par))
    if failed:
        lines.append("failed 3 times: " + ", ".join(failed))
    with open("lr_status.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

reserved_gpu = {}   # gpu -> list of (time, GB) for runs started in the last 4 minutes

def try_start():
    if os.path.exists("STOP_LR") or len(running) >= max_runs() or not queue:
        return False
    if os.getloadavg()[0] > LOAD_LIMIT:
        return False
    gpus = free_gpus()
    if not gpus:
        return False
    now = time.time()
    for g in list(reserved_gpu):
        reserved_gpu[g] = [(t, gb) for t, gb in reserved_gpu[g] if now - t < (1 if os.environ.get("LR_TEST") else 240)]
    pending_ram = sum(max(0.0, NEED_RAM_GB[r["job"]["ds"]] - rss_gb(r["proc"].pid)) for r in running.values())
    ram_free = mem_available_gb() - pending_ram - RESERVE_RAM_GB
    for j in list(queue):                         # first job that fits
        if NEED_RAM_GB[j["ds"]] > ram_free:
            continue
        need = NEED_GPU_GB[j["method"]]
        cands = [(free - sum(gb for _, gb in reserved_gpu.get(g, [])), g) for g, free in gpus.items()]
        cands = [c for c in cands if c[0] >= need + 1.5]
        if not cands:
            continue
        _, g = max(cands)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), OMP_NUM_THREADS=str(THREADS),
                   MKL_NUM_THREADS=str(THREADS), OPENBLAS_NUM_THREADS=str(THREADS), BANDNORM_REF="63", **j.get("env", {}))
        log = "logs/lr_%s.log" % j["tag"]
        lf = open(log, "a")
        p = subprocess.Popen(["nice", "-n", "5"] + j["cmd"], stdout=lf, stderr=subprocess.STDOUT, env=env)
        running[j["tag"]] = dict(job=j, proc=p, gpu=g, start=now, log=log, fh=lf)
        reserved_gpu.setdefault(g, []).append((now, need))
        queue.remove(j)
        print(time.strftime("%H:%M"), "START", j["tag"], "GPU%d" % g, flush=True)
        return True
    return False

def reap():
    global finished_now
    for tag, r in list(running.items()):
        rc = r["proc"].poll()
        if rc is None:
            continue
        r["fh"].close()
        j = r["job"]; minutes = (time.time() - r["start"]) / 60
        del running[tag]
        if is_done(j):
            finished_now += 1
            durations.setdefault((j["ds"], j["method"]), []).append(minutes)
            record_run(j, r["log"], minutes)
            print(time.strftime("%H:%M"), "DONE ", tag, "%.0f min" % minutes, flush=True)
        else:
            j["tries"] += 1
            tail = open(r["log"], errors="ignore").read()[-3000:]
            why = "out of GPU memory" if "out of memory" in tail.lower() else "exit code %s" % rc
            if j["tries"] < MAX_TRIES:
                queue.append(j)
                print(time.strftime("%H:%M"), "RETRY", tag, "(%s, try %d)" % (why, j["tries"]), flush=True)
            else:
                failed.append(tag)
                print(time.strftime("%H:%M"), "FAILED", tag, "(%s) - see %s" % (why, r["log"]), flush=True)

print("LR sweep: %d runs to do. Progress: cat lr_status.txt   Stop: touch STOP_LR" % len(queue), flush=True)
last_start = 0
while queue or running:
    reap()
    if os.path.exists("STOP_LR") and not running:
        print("STOP_LR found and nothing running - exiting"); break
    if time.time() - last_start > (0 if os.environ.get("LR_TEST") else 20) and try_start():   # at most one new run every 20 s
        last_start = time.time()
    write_status()
    time.sleep(0.2 if os.environ.get("LR_TEST") else 10)
write_status()
print("all done" if not queue else "stopped with %d runs left" % len(queue), flush=True)
