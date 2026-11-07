"""真实仿真与MPS更新小试；输出数值、速度、原始轨迹，不用其权重初始化正式训练。"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from resource_rl.environment import ResourceEnvironment
from resource_rl.policy import ResourcePolicy
from resource_rl.ppo import update
from resource_rl.protocol import save_json
from resource_rl.rollout import collect, collate, cpu_weights, initialize_worker, save_batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    reference = json.loads(Path("experiments/online_ppo_100000_20261104/protocol.json").read_text())
    env = ResourceEnvironment(reference["config"])
    info = env.encoder.metadata()
    config = dict(grid_count=6, node_size=len(info["node_columns"]), edge_size=len(info["edge_columns"]),
        driver_size=info["driver_size"], order_size=info["order_size"], price_count=11, radius_count=10,
        hidden_size=64, heads=4, layers=1, window=4)
    settings = dict(warmup_steps=4, gamma=1., gae_lambda=.95, reward_scale=100., clip=.1, target_kl=.02,
        actor_epochs=4, critic_epochs=4, sequence_batch_size=8, learning_rate=3e-4,
        value_weight=.5, auxiliary_weight=.1, entropy_weight=.001, max_grad_norm=.5)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
            initializer=initialize_worker, initargs=(reference["config"], config, ["flow", "window"])) as pool:
        for variant in ("flow", "window"):
            torch.manual_seed(201)
            model = ResourcePolicy(**config, variant=variant).to(args.device).eval()
            torch.save(cpu_weights(model), output/f"{variant}_initial.pt")
            begun = perf_counter()
            episodes = collect(pool, model, range(1_060_000_201, 1_060_000_217), 201, 0)
            collected = perf_counter()
            data = collate(episodes, args.device)
            with torch.no_grad():
                logp = torch.cat([model.actor_sequence({k: v[i:i+8] for k,v in data.items()})[0] for i in range(0,16,8)])
                difference = float((logp[:, 4:]-data["log_probs"][:, 4:]).abs().max().cpu())
            save_batch(output/f"{variant}_batch.npz", episodes)
            actor = torch.optim.Adam(model.actor_parameters(), lr=3e-4, eps=1e-5)
            critic = torch.optim.Adam(model.critic_parameters(), lr=3e-4, eps=1e-5)
            diagnostics = update(model, actor, critic, episodes, settings, np.random.default_rng(201), args.device)
            record = {"variant": variant, "episodes":16, "scored_city_transitions":16*56,
                "parameters":sum(p.numel() for p in model.parameters()), "collection_seconds":collected-begun,
                "update_seconds":perf_counter()-collected, "stream_sequence_logp_max_error":difference,
                "finite_diagnostics":bool(np.isfinite(list(diagnostics.values())).all()), **diagnostics}
            if args.device == "mps":
                record["mps_allocator_mb"] = torch.mps.driver_allocated_memory()/1024**2
            reports.append(record)
            save_json(output/"report.json", {"device":args.device, "scope":"correctness and cost only; no model selection; main starts from random weights", "runs":reports})
            print(json.dumps(record), flush=True)
