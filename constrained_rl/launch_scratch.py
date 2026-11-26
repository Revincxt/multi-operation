"""独立进程组启动从零长训练，保存日志；重复运行会按冻结协议续训。"""

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys

from constrained_rl.scratch_protocol import make_protocol, load_protocol
from offline_rl.data import save_json


def run(args):
    root = Path(args.output)
    if not root.exists():
        make_protocol(root, episodes=args.episodes, training_seed=args.training_seed)
    load_protocol(root / "protocol.json")
    with (root / ".train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    command = [sys.executable, "-u", "-m", "constrained_rl.train_scratch",
               "--output", str(root), "--workers", str(args.workers), "--device", args.device]
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        command = ["caffeinate", "-i", *command]
    environment = dict(os.environ, MPLCONFIGDIR="/private/tmp/constrained_rl_mpl",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       VECLIB_MAXIMUM_THREADS="1")
    with (root / "run.log").open("ab") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=environment)
    save_json(root / "launch.json", {"pid": process.pid, "command": command, "log": str(root / "run.log")})
    print(f"started pid={process.pid} log={root / 'run.log'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="models/constrained_ppo_scratch100k_20261116")
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--training-seed", type=int, default=412)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    run(parser.parse_args())
