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
               "--batch-size", str(args.batch_size), "--seed", str(args.seed), "--workers", str(args.workers),
               "--threads", str(args.threads), "--device", args.device]
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
    parser.add_argument("--output", default="models/dlmrd_grid6_low47k_mps1000_20261117")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    run(parser.parse_args())
