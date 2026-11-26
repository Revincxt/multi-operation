"""后台启动DL-MRD监督训练，保存日志与进程号。"""

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys

from baseline.dlmrd.train import DEFAULT_CACHE
from baseline.data import DEFAULT_INPUT
from offline_rl.data import save_json


def run(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    command = [sys.executable, "-u", "-m", "baseline.dlmrd.train", "--input", args.input,
               "--prepared-cache", args.prepared_cache, "--output", str(root), "--epochs", str(args.epochs),
               "--batch-size", str(args.batch_size), "--learning-rate", str(args.learning_rate),
               "--optimizer", args.optimizer, "--weight-decay", str(args.weight_decay),
               "--seed", str(args.seed), "--workers", str(args.workers),
               "--threads", str(args.threads), "--device", args.device,
               "--revenue-weight-multiplier", str(args.revenue_weight_multiplier),
               "--normalization", args.normalization]
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        command = ["caffeinate", "-i", *command]
    environment = dict(os.environ, MPLCONFIGDIR="/private/tmp/dlmrd_mpl", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    with (root / "run.log").open("ab") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=environment)
    save_json(root / "launch.json", {"pid": process.pid, "command": command, "log": str(root / "run.log")})
    print(f"started pid={process.pid} log={root / 'run.log'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--prepared-cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--output", default="models/dlmrd_joint_fusion_grid6_low47k_cpu100")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adamw")
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--revenue-weight-multiplier", type=float, default=4.5)
    parser.add_argument("--normalization", choices=("zscore", "minmax"), default="zscore")
    run(parser.parse_args())
