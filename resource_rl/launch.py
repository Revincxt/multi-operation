"""在独立进程组中启动可恢复训练，日志与PID均保存在新输出目录。"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

from resource_rl.protocol import save_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="mps")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", "-m", "resource_rl.train", "--protocol", args.protocol,
               "--output", args.output, "--device", args.device, "--workers", str(args.workers), "--evaluate-at-end"]
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        command = ["caffeinate", "-i", *command]
    environment = dict(os.environ, MPLCONFIGDIR="/private/tmp/resource_rl_mpl")
    with (output/"run.log").open("ab") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=environment)
    save_json(output/"launch.json", {"pid": process.pid, "command": command, "log": str(output/"run.log")})
    print(f"started pid={process.pid} log={output/'run.log'}", flush=True)
