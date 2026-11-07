"""数据加载：历史订单和司机排班。"""

import json
import pickle
from pathlib import Path
import pandas as pd


def load_grid_manifest(data_dir):
    """新划分附带网格清单；没有清单的旧数据保持原有加载方式。"""
    path = Path(data_dir) / 'grid_manifest.json'
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if (manifest.get('schema_version') != 'hex_grid_inputs_v1'
            or not isinstance(manifest.get('grid_num'), int)
            or manifest['grid_num'] <= 0):
        raise ValueError
    return manifest


def input_data_paths(date, data_dir):
    """解析已映射的新网格输入或历史文件名。"""
    root = Path(data_dir)
    manifest = load_grid_manifest(root)
    if manifest is not None:
        if date not in manifest['dates']:
            raise FileNotFoundError
        return root / f'orders_grid{manifest["grid_num"]}_{date}.pkl', root / 'drivers.pickle'
    return root / f'orders_grid35_{date}.pkl', root / 'drivers_grid35_1000.pickle'


class SimulatorPattern:
    def __init__(self, date, data_dir='my_data'):
        order_path, driver_path = input_data_paths(date, data_dir)
        with order_path.open('rb') as f:
            self.request_all = pickle.load(f)
        with driver_path.open('rb') as f:
            self.driver_info = pickle.load(f)

    @staticmethod
    def load_mapping(date, data_dir='my_data'):
        """加载 order_id → grid_id 映射表。"""
        return pd.read_csv(f'{data_dir}/orders_grid35_{date}-map263.csv')
