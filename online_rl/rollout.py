"""固定一个策略版本收集完整轨迹；仿真RNG与策略RNG彼此独立。"""

from time import perf_counter

import numpy as np
import torch

from online_rl.environment import DecisionEnvironment
from online_rl.policy import JointPolicy


_ENVIRONMENT = None
_POLICY = None


def initialize_worker(config, metadata, model_config):
    global _ENVIRONMENT, _POLICY
    torch.set_num_threads(1)
    _ENVIRONMENT = DecisionEnvironment(config, metadata)
    _POLICY = JointPolicy(**model_config).eval()


def collect_episode(job):
    weights, scenario_seed, action_seed, version = job
    _POLICY.load_state_dict({name: torch.from_numpy(value) for name, value in weights.items()})
    generator = torch.Generator().manual_seed(action_seed)
    observation = _ENVIRONMENT.reset(scenario_seed)
    records = {key: [] for key in ("states", "actions", "log_probs", "values", "rewards", "terminals")}
    started, terminal = perf_counter(), False
    while not terminal:
        state = torch.from_numpy(observation[None])
        with torch.inference_mode():
            actions, logp, value = _POLICY.sample(state, generator)
        following, reward, terminal = _ENVIRONMENT.step(actions[0].numpy())
        for key, item in zip(records, (observation.copy(), actions[0].numpy().copy(), float(logp[0]),
                                      float(value[0]), reward, terminal)):
            records[key].append(item)
        observation = following
    result = {key: np.asarray(value, dtype="int64" if key == "actions" else "float32") for key, value in records.items()}
    summary = _ENVIRONMENT.summary()
    np.testing.assert_allclose(result["rewards"].astype(float).sum(), summary["platform_revenue"], rtol=1e-6)
    return {**result, "summary": summary, "seed": scenario_seed, "action_seed": action_seed,
            "policy_version": version, "elapsed_seconds": perf_counter()-started}


def collect(pool, weights, seeds, training_seed, version):
    from online_rl.protocol import policy_seed
    # 数组序列化保持原始精度，不依赖平台的Torch共享内存服务。
    weights = {name: value.numpy() for name, value in weights.items()}
    jobs = [(weights, int(seed), policy_seed(training_seed, int(seed)), version) for seed in seeds]
    return list(pool.map(collect_episode, jobs)) if pool is not None else [collect_episode(job) for job in jobs]
