"""分批采集密度场景；每次仿真独立落盘，重复执行会校验并跳过已完成记录。

使用方式:
    python3 -m data_pipeline.collect_density --scenario low --runs 100000
    python3 -m data_pipeline.collect_density --scenario low --runs 100000 --max-new-runs 3
    python3 -m data_pipeline.collect_density --scenario low --status
    python3 -m data_pipeline.collect_density --scenario low --stop

默认低密度为 6 网格、250 名司机、0.077 订单采样比例，05:00–10:00。
原始 CSV 按 run 压缩、按批次分目录；不将所有轨迹拼接到内存中。
planned_split 是后续大规模数据构建应沿用的 run 级划分，不直接训练模型。
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import redirect_stdout
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import shutil
import tempfile
from time import perf_counter, sleep
import traceback

import numpy as np

from data_pipeline.collect_trajectories import (
    check_collection_inputs, collection_run_id, run_trajectory_collection,
    simulation_window, joint_action_sampling,
)
from src.env.actions import DEFAULT_MATCHING_RADII_KM, DEFAULT_PRICE_MULTIPLIERS
from src.env.simulator_pattern import input_data_paths
from src.env.observation import load_neighbor_adjacency, state_metadata
from data_pipeline.collection_inputs import prepare_worker_inputs, load_worker_pattern


SCENARIOS = {
    'low': {'driver_num': 250, 'order_sample_ratio': 0.077},
    'medium': {'driver_num': 500, 'order_sample_ratio': 0.154},
    'high': {'driver_num': 1000, 'order_sample_ratio': 0.385},
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_WORKER_PATTERN = None


def initialize_worker(prepared, counter, stagger_seconds):
    """仅首次加载错峰，后续每轮立即接续，不插入人为等待。"""
    global _WORKER_PATTERN
    with counter.get_lock():
        slot = counter.value
        counter.value += 1
    if slot and stagger_seconds:
        sleep(slot * stagger_seconds)
    _WORKER_PATTERN = load_worker_pattern(prepared)


def config_signature(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def validate_execution_revision(root, original, current):
    """只接受已逐字段复核的精确代码版本，不放宽实验参数或输入校验。"""
    if original == current:
        return
    if ({k: v for k, v in original.items() if k != 'code_sha256'}
            != {k: v for k, v in current.items() if k != 'code_sha256'}):
        raise ValueError
    path = Path(root) / 'execution_revision.json'
    if not path.is_file():
        raise ValueError
    revision = json.loads(path.read_text())
    if (revision.get('collection_config_sha256') != config_signature(original)
            or revision.get('code_sha256') != current.get('code_sha256')
            or revision.get('validation', {}).get('passed') is not True):
        raise ValueError


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    """同目录原子替换；中断时不留下半份清单。"""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def make_config(scenario, runs, seed_start, batch_size, data_dir, date,
                start_time, simulation_steps):
    if scenario not in SCENARIOS:
        raise ValueError
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0
           for n in (runs, batch_size, simulation_steps)):
        raise ValueError
    if (isinstance(seed_start, bool) or not isinstance(seed_start, int)
            or seed_start < 0 or seed_start + runs > 2**32):
        raise ValueError
    start_time, start, end = simulation_window(start_time, simulation_steps)
    check_collection_inputs([date], 6, data_dir)
    inputs = list(input_data_paths(date, data_dir)) + [Path(data_dir) / 'grid_manifest.json']
    with inputs[0].open('rb') as file:
        orders = pickle.load(file)
    if start < min(orders) or end > max(orders) + 1:
        raise ValueError
    # 现有循环第一步不加载订单，最后加载 [结束前 2 分钟, 结束前 1 分钟)。
    minute_counts = [sum(len(orders.get(t, [])) for t in range(minute, minute + 60))
                     for minute in range(start, end - 60, 60)]
    ratio = SCENARIOS[scenario]['order_sample_ratio']
    code_paths = sorted((PROJECT_ROOT / 'src').rglob('*.py')) + [
        PROJECT_ROOT / 'data_pipeline' / 'collect_trajectories.py',
        PROJECT_ROOT / 'data_pipeline' / 'collection_inputs.py', Path(__file__).resolve()]
    return {
        'schema_version': 'density_collection_v2', 'scenario': scenario,
        'run_count': runs, 'seed_start': seed_start, 'batch_size': batch_size,
        'date': date, 'start_time': start_time, 'simulation_steps': simulation_steps,
        'grid_num': 6, 'decision_freq': 5, **SCENARIOS[scenario],
        'price_response_model': 'utility_choice', 'enable_reposition': False,
        'price_action_values': list(DEFAULT_PRICE_MULTIPLIERS),
        'matching_radius_values': list(DEFAULT_MATCHING_RADII_KM),
        'action_sampling': joint_action_sampling(DEFAULT_PRICE_MULTIPLIERS, DEFAULT_MATCHING_RADII_KM),
        **state_metadata(load_neighbor_adjacency(data_dir, 6)),
        'data_dir': str(Path(data_dir).resolve()),
        'available_baseline_requests': sum(minute_counts),
        'expected_sampled_baseline_requests': sum(int(np.rint(ratio * n)) for n in minute_counts),
        'split_seed': 42, 'validation_ratio': 0.15, 'test_ratio': 0.15,
        'source_sha256': {str(p.resolve()): sha256(p) for p in inputs},
        'code_sha256': {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in code_paths},
    }


def run_path(root, config, index):
    seed = config['seed_start'] + index
    name = collection_run_id(
        config['date'], seed, seed, config['grid_num'], config['decision_freq'],
        config['price_response_model'], config['start_time'], config['simulation_steps'],
        config['driver_num'], config['order_sample_ratio'])
    return Path(root) / f'batch_{index // config["batch_size"]:05d}' / name


def planned_splits(config):
    """按完整 run 预先划分，同编号的三个密度场景使用相同归属。"""
    count = config['run_count']
    order = np.random.default_rng(config['split_seed']).permutation(count)
    splits = np.full(count, 'train', dtype='<U10')
    n_validation = int(count * config['validation_ratio'])
    n_test = int(count * config['test_ratio'])
    splits[order[:n_validation]] = 'validation'
    splits[order[n_validation:n_validation + n_test]] = 'test'
    return splits


def completed_manifest(path, signature):
    """只跳过配置和文件摘要均一致的完整 run；损坏记录不静默覆盖。"""
    if not path.exists():
        return None
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('collection_config_sha256') != signature
            or not manifest.get('file_sha256')):
        raise ValueError
    for name, expected in manifest['file_sha256'].items():
        if Path(name).name != name or sha256(path / name) != expected:
            raise ValueError
    return manifest


def collect_one(root, config, signature, index, split):
    """先在临时目录完成保存与校验，再发布整个 run 目录。"""
    destination = run_path(root, config, index)
    if destination.exists():
        raise FileExistsError
    destination.parent.mkdir(parents=True, exist_ok=True)
    seed = config['seed_start'] + index
    with tempfile.TemporaryDirectory(prefix='.working_', dir=destination.parent) as temporary:
        with redirect_stdout(io.StringIO()):
            result = run_trajectory_collection(
                date=config['date'], seed=seed, action_seed=seed,
                grid_num=config['grid_num'], driver_num=config['driver_num'],
                order_sample_ratio=config['order_sample_ratio'],
                decision_freq=config['decision_freq'], simulation_steps=config['simulation_steps'],
                start_time=config['start_time'], data_dir=config['data_dir'],
                price_action_values=config['price_action_values'],
                matching_radius_values=config['matching_radius_values'],
                price_response_model=config['price_response_model'],
                enable_reposition=config['enable_reposition'], output_dir=temporary, compress=True,
                _pattern=_WORKER_PATTERN)
        manifest = result['manifest']
        if (manifest['summary']['baseline_requests'] != config['expected_sampled_baseline_requests']
                or manifest['neighbor_adjacency_sha256'] != config['neighbor_adjacency_sha256']
                or manifest['action_sampling'] != config['action_sampling']):
            raise RuntimeError
        minute = result['minute_metrics']
        initial = minute[minute['minute_step'] == 0]
        if int(initial['idle_driver_num'].sum()) != config['driver_num']:
            raise RuntimeError
        generated = Path(result['run_dir'])
        manifest.update({
            'scenario': config['scenario'], 'run_index': index, 'planned_split': str(split),
            'collection_config_sha256': signature,
            'execution_code_sha256': config['code_sha256'],
            'file_sha256': {name: sha256(generated / name) for name in manifest['files'].values()},
        })
        save_json(generated / 'manifest.json', manifest)
        generated.rename(destination)
    return manifest


def run_density_collection(scenario='low', runs=100000, seed_start=42, workers=2,
                           batch_size=1000, output=None, data_dir='my_data/grid6',
                           date='2015-05-18', start_time='05:00', simulation_steps=300,
                           max_new_runs=None, resume=False, min_free_gb=10.0,
                           startup_stagger_seconds=2.0):
    """限制在途任务数量，逐次保存进度；失败最多重试两次后安全停止。"""
    if workers < 1 or (max_new_runs is not None and max_new_runs < 1):
        raise ValueError
    if not np.isfinite(min_free_gb) or min_free_gb < 0:
        raise ValueError
    if not np.isfinite(startup_stagger_seconds) or startup_stagger_seconds < 0:
        raise ValueError
    root = Path(output or f'trajectories_grid6_{scenario}').resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.collection.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = make_config(scenario, runs, seed_start, batch_size, data_dir,
                             date, start_time, simulation_steps)
        config_path = root / 'collection_config.json'
        if config_path.exists():
            original = json.loads(config_path.read_text(encoding='utf-8'))
            validate_execution_revision(root, original, config)
        else:
            if any(root.glob('batch_*')):
                raise FileExistsError
            save_json(config_path, config)
            original = config
        signature = config_signature(original)
        stop_path = root / 'STOP'
        if resume and stop_path.exists():
            stop_path.unlink()
        splits = planned_splits(config)
        completed = new_completed = started = index = 0
        run_seconds = 0.0
        sampling_modes = {name: 0 for name in ('preferred', 'uniform', 'reference')}
        began = perf_counter()
        stop_reason = None

        def progress(status):
            state = {
                'status': status, 'pid': os.getpid(), 'scenario': scenario,
                'target_runs': runs, 'completed_runs': completed,
                'new_completed_runs': new_completed, 'workers': workers,
                'startup_stagger_seconds': startup_stagger_seconds,
                'session_elapsed_seconds': perf_counter() - began,
                'mean_new_run_seconds': run_seconds / new_completed if new_completed else None,
                'new_run_sampling_modes': dict(sampling_modes),
                'updated_at': datetime.now(timezone.utc).isoformat(),
            }
            save_json(root / 'progress.json', state)
            return state

        progress('running')
        print(f'{scenario}: 目标 {runs:,} 次，每次基准请求 '
              f'{config["expected_sampled_baseline_requests"]:,}，司机 {config["driver_num"]}', flush=True)
        # 停止标记存在时不加载数据；恢复时仅在父进程准备一次窗口缓存。
        prepared = None
        if not stop_path.exists() and shutil.disk_usage(root).free >= min_free_gb * 1024**3:
            prepared = prepare_worker_inputs(root, config)
        worker_options = {}
        if prepared is not None:
            counter = multiprocessing.get_context('spawn').Value('i', 0)
            worker_options = {'initializer': initialize_worker,
                              'initargs': (prepared, counter, startup_stagger_seconds)}
        with ProcessPoolExecutor(max_workers=workers, **worker_options) as pool:
            pending = {}
            try:
                while index < runs or pending:
                    if stop_path.exists():
                        stop_reason = 'stopped'
                    elif shutil.disk_usage(root).free < min_free_gb * 1024**3:
                        stop_reason = 'disk_guard'
                    while (stop_reason is None and len(pending) < workers and index < runs
                           and (max_new_runs is None or started < max_new_runs)):
                        previous = completed_manifest(run_path(root, config, index), signature)
                        if previous is not None:
                            completed += 1
                            index += 1
                            continue
                        future = pool.submit(collect_one, root, config, signature, index, splits[index])
                        pending[future] = (index, 0)
                        index += 1
                        started += 1
                    if not pending:
                        break
                    done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                    for future in done:
                        run_index, attempt = pending.pop(future)
                        try:
                            manifest = future.result()
                        except Exception:
                            with (root / 'errors.log').open('a', encoding='utf-8') as file:
                                file.write(f'run_index={run_index}, attempt={attempt}\n{traceback.format_exc()}\n')
                            if attempt < 2 and stop_reason is None:
                                retry = pool.submit(collect_one, root, config, signature,
                                                    run_index, splits[run_index])
                                pending[retry] = (run_index, attempt + 1)
                            else:
                                stop_reason = 'failed'
                            continue
                        completed += 1
                        new_completed += 1
                        mode = manifest.get('sampling_mode')
                        if mode in sampling_modes:
                            sampling_modes[mode] += 1
                        run_seconds += manifest['elapsed_seconds']
                        print(f'完成 {completed:,}/{runs:,}，seed={manifest["simulator_seed"]}，'
                              f'基准请求={manifest["summary"]["baseline_requests"]}，'
                              f'耗时={manifest["elapsed_seconds"]:.1f}s', flush=True)
                    progress(stop_reason or 'running')
            except BaseException:
                progress('interrupted')
                raise
        status = stop_reason or ('complete' if completed == runs else 'batch_limit')
        return progress(status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=SCENARIOS, default='low')
    parser.add_argument('--runs', type=int, default=100000)
    parser.add_argument('--seed-start', type=int, default=42)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--output')
    parser.add_argument('--data-dir', default='my_data/grid6')
    parser.add_argument('--date', default='2015-05-18')
    parser.add_argument('--start-time', default='05:00')
    parser.add_argument('--simulation-steps', type=int, default=300)
    parser.add_argument('--max-new-runs', type=int)
    parser.add_argument('--min-free-gb', type=float, default=10.0)
    parser.add_argument('--startup-stagger-seconds', type=float, default=2.0)
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument('--status', action='store_true')
    controls.add_argument('--stop', action='store_true')
    controls.add_argument('--resume', action='store_true')
    args = vars(parser.parse_args())
    status, stop = args.pop('status'), args.pop('stop')
    root = Path(args['output'] or f'trajectories_grid6_{args["scenario"]}')
    if status:
        print((root / 'progress.json').read_text(encoding='utf-8'))
    elif stop:
        if not (root / 'collection_config.json').is_file():
            raise FileNotFoundError
        (root / 'STOP').touch()
        print('已请求停止；当前在途仿真完成后退出。')
    else:
        result = run_density_collection(**args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result['status'] in ('failed', 'disk_guard'):
            raise RuntimeError


if __name__ == '__main__':
    main()
