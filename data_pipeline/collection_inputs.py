"""密度采集的轻量输入：截取时段、去掉未使用路径列，供工作进程复用。"""

import hashlib
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace

from src.env.simulator_pattern import input_data_paths


def file_sha256(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def compact_orders(orders, start, end):
    """保留订单顺序及全部有效字段；第 11、12 列在仿真器中立即丢弃。"""
    if not orders or start >= end or start < min(orders) or end > max(orders) + 1:
        raise ValueError
    compact = {}
    for timestamp, records in orders.items():
        if start <= timestamp < end:
            rows = []
            for record in records:
                if len(record) != 16:
                    raise ValueError
                rows.append(list(record[:11]) + [None, None] + list(record[13:]))
            compact[timestamp] = rows
    # 空边界仍表示原始时间索引覆盖该窗口，不制造任何订单。
    compact.setdefault(start, [])
    compact.setdefault(end - 1, [])
    return compact


def prepare_worker_inputs(root, config):
    """父进程只生成一次轻量缓存；源文件、窗口或缓存内容不一致时拒绝复用。"""
    order_path, driver_path = input_data_paths(config['date'], config['data_dir'])
    source_hash = config['source_sha256'][str(order_path.resolve())]
    hour, minute = map(int, config['start_time'].split(':'))
    start = hour * 3600 + minute * 60
    end = start + config['simulation_steps'] * 60
    cache = Path(root) / '.input_cache'
    cache.mkdir(exist_ok=True)
    name = f'compact_v1_{start}_{end}_{source_hash}'
    payload = cache / f'{name}.pkl'
    receipt = cache / f'{name}.json'
    identity = {'source_sha256': source_hash, 'start': start, 'end': end}
    if payload.exists() and receipt.exists():
        metadata = json.loads(receipt.read_text())
        if (metadata != {**identity, 'sha256': file_sha256(payload)}):
            raise ValueError
    else:
        if file_sha256(order_path) != source_hash:
            raise ValueError
        with order_path.open('rb') as file:
            orders = pickle.load(file)
        compact = compact_orders(orders, start, end)
        del orders
        with tempfile.NamedTemporaryFile(dir=cache, delete=False) as file:
            temporary = Path(file.name)
            try:
                pickle.dump(compact, file, protocol=pickle.HIGHEST_PROTOCOL)
                file.flush()
                temporary.replace(payload)
            finally:
                temporary.unlink(missing_ok=True)
        metadata = {**identity, 'sha256': file_sha256(payload)}
        temporary = receipt.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(metadata, indent=2))
        temporary.replace(receipt)
    return {
        'orders_path': str(payload.resolve()), 'orders_sha256': metadata['sha256'],
        'drivers_path': str(driver_path.resolve()),
        'drivers_sha256': config['source_sha256'][str(driver_path.resolve())],
    }


def load_worker_pattern(prepared):
    """每个进程仅加载一次；仿真器只读取订单，司机状态在每轮 reset 时重建。"""
    for name in ('orders', 'drivers'):
        if file_sha256(prepared[f'{name}_path']) != prepared[f'{name}_sha256']:
            raise ValueError
    with Path(prepared['orders_path']).open('rb') as file:
        orders = pickle.load(file)
    with Path(prepared['drivers_path']).open('rb') as file:
        drivers = pickle.load(file)
    return SimpleNamespace(request_all=orders, driver_info=drivers)
