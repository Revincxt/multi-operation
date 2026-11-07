"""同版本策略收集完整episode；保留预热、微观集合和真实资源转移标签。"""

from time import perf_counter

import numpy as np
import torch

from resource_rl.environment import ResourceEnvironment
from resource_rl.policy import ResourcePolicy


_ENVIRONMENT = _POLICIES = None


def action_seed(training_seed, scenario_seed):
    return int(np.random.SeedSequence([20261105, training_seed, scenario_seed]).generate_state(1)[0])


def initialize_worker(config, model_config, variants):
    global _ENVIRONMENT, _POLICIES
    torch.set_num_threads(1)
    _ENVIRONMENT = ResourceEnvironment(config)
    _POLICIES = {name: ResourcePolicy(**model_config, variant=name).eval() for name in variants}


def observation_tensor(observation, device="cpu"):
    result = {name: torch.as_tensor(value[None], device=device) for name, value in observation.items()}
    result["order_mask"] = torch.zeros((1, len(observation["orders"])), dtype=torch.bool, device=device)
    return result


def episode(job):
    weights, variant, seed, training_seed, version = job
    model = _POLICIES[variant]
    model.load_state_dict({name: torch.from_numpy(value) for name, value in weights.items()})
    return run_episode(_ENVIRONMENT, model, int(seed), training_seed, version)


def run_episode(env, model, seed, training_seed, version):
    generator = torch.Generator().manual_seed(action_seed(training_seed, seed))
    current = env.reset(seed)
    actor_state = critic_state = None
    observations, actions, probabilities, values, rewards, terminals, targets = [], [], [], [], [], [], []
    started = perf_counter()
    with torch.inference_mode():
        for observation, reward, target in env.warmup:
            forced = torch.full((1, env.config["grid_num"]), env.warm_action, dtype=torch.long)
            chosen, logp, value, actor_state, critic_state = model.step(
                observation_tensor(observation), actor_state, critic_state, generator, forced)
            observations.append(observation); actions.append(chosen[0].numpy().copy())
            probabilities.append(float(logp[0])); values.append(float(value[0]))
            rewards.append(reward); terminals.append(False); targets.append(target)
        terminal = False
        while not terminal:
            chosen, logp, value, actor_state, critic_state = model.step(
                observation_tensor(current), actor_state, critic_state, generator)
            following, reward, terminal = env.step(chosen[0].numpy())
            observations.append(current); actions.append(chosen[0].numpy().copy())
            probabilities.append(float(logp[0])); values.append(float(value[0]))
            rewards.append(reward); terminals.append(terminal); targets.append(env.transition_target())
            current = following
    result = {name: np.stack([row[name] for row in observations]) for name in ("nodes", "edges", "drivers", "previous_actions")}
    result.update(orders=np.concatenate([row["orders"] for row in observations]),
        order_lengths=np.asarray([len(row["orders"]) for row in observations], dtype="int64"),
        actions=np.asarray(actions, dtype="int64"), log_probs=np.asarray(probabilities, dtype="float32"),
        values=np.asarray(values, dtype="float32"), rewards=np.asarray(rewards, dtype="float32"),
        terminals=np.asarray(terminals, dtype="float32"), aux_targets=np.asarray(targets, dtype="float32"))
    summary = env.summary()
    np.testing.assert_allclose(result["rewards"][env.config["history_steps"]:].sum(dtype=float), summary["platform_revenue"], rtol=1e-6)
    return {**result, "summary": summary, "seed": int(seed), "action_seed": action_seed(training_seed, seed),
            "policy_version": version, "elapsed_seconds": perf_counter()-started}


def cpu_weights(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def collect(pool, model, seeds, training_seed, version):
    weights = {name: value.numpy() for name, value in cpu_weights(model).items()}
    jobs = [(weights, model.variant, int(seed), training_seed, version) for seed in seeds]
    return list(pool.map(episode, jobs))


def collate(episodes, device):
    names = ("nodes", "edges", "drivers", "previous_actions", "actions", "log_probs", "values", "rewards", "terminals", "aux_targets")
    result = {name: torch.as_tensor(np.stack([row[name] for row in episodes]), device=device) for name in names}
    b, t = result["nodes"].shape[:2]
    maximum = max(1, max(int(row["order_lengths"].max()) for row in episodes))
    features = episodes[0]["orders"].shape[-1]
    orders = np.zeros((b, t, maximum, features), dtype="float32")
    mask = np.ones((b, t, maximum), dtype=bool)
    for batch, row in enumerate(episodes):
        start = 0
        for time, length in enumerate(row["order_lengths"]):
            orders[batch, time, :length] = row["orders"][start:start+length]
            mask[batch, time, :length] = False
            start += length
        if start != len(row["orders"]):
            raise ValueError
    result["orders"], result["order_mask"] = torch.as_tensor(orders, device=device), torch.as_tensor(mask, device=device)
    return result


def save_batch(path, episodes):
    """集合使用flat数组和长度编码，避免object数组及pickle依赖。"""
    names = ("nodes", "edges", "drivers", "previous_actions", "actions", "log_probs", "values", "rewards", "terminals", "aux_targets", "order_lengths")
    arrays = {name: np.stack([row[name] for row in episodes]) for name in names}
    arrays["orders"] = np.concatenate([row["orders"] for row in episodes])
    arrays["scenario_seeds"] = np.asarray([row["seed"] for row in episodes], dtype="int64")
    arrays["action_seeds"] = np.asarray([row["action_seed"] for row in episodes], dtype="int64")
    np.savez_compressed(path, **arrays)
