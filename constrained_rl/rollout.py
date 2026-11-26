"""冻结策略采完整仿真，分别记录收入与服务成本，不在worker中更新参数。"""

import numpy as np
import torch

from constrained_rl.environment import ConstrainedEnvironment
from constrained_rl.policy import ConstrainedPolicy
from online_rl.protocol import policy_seed


_ENV = _POLICY = None


def initialize_worker(config, metadata, model_config):
    global _ENV, _POLICY
    torch.set_num_threads(1)
    _ENV = ConstrainedEnvironment(config, metadata)
    _POLICY = ConstrainedPolicy(**model_config).eval()


def run_episode(job):
    weights, seed, training_seed, policy_name = job
    _POLICY.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
    rng = torch.Generator().manual_seed(policy_seed(training_seed, seed, namespace=20261115))
    state = _ENV.reset(seed)
    rows = {k: [] for k in ("states", "actions", "log_probs", "values", "cost_values", "rewards", "terminals", "demand", "matches", "wait_seconds")}
    done = False
    while not done:
        tensor = torch.from_numpy(state[None])
        with torch.inference_mode():
            if policy_name == "fixed_f3":
                pair = (_ENV.config["fixed_price"], _ENV.config["fixed_radius"])
                index = [tuple(p) for p in _ENV.pairs].index(pair)
                actions = torch.full((1, _ENV.config["grid_num"]), index, dtype=torch.long)
                logp, _, value = _POLICY.evaluate_actions(tensor, actions)
            else:
                actions, logp, value = _POLICY.sample(tensor, rng)
            cost_values = _POLICY.cost_values(tensor)[0].numpy()
        following, reward, done = _ENV.step(actions[0].numpy())
        signal = _ENV.last_signals
        items = (state, actions[0].numpy(), float(logp[0]), float(value[0]), cost_values,
                 reward, done, signal["cohort_demand"], signal["unique_matched"], signal["queue_wait_seconds"])
        for key, value in zip(rows, items):
            rows[key].append(value)
        state = following
    result = {k: np.asarray(v, dtype="int64" if k == "actions" else "float32") for k, v in rows.items()}
    summary = _ENV.summary()
    for key, column in (("rewards", "platform_revenue"), ("demand", "cohort_demand"),
                        ("matches", "unique_matched"), ("wait_seconds", "queue_wait_seconds")):
        np.testing.assert_allclose(result[key].astype(float).sum(), summary[column], rtol=1e-6)
    return {**result, "summary": summary, "seed": seed}


def collect(pool, model, seeds, training_seed, policy_name):
    weights = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
    jobs = [(weights, int(seed), training_seed, policy_name) for seed in seeds]
    return list(pool.map(run_episode, jobs))
