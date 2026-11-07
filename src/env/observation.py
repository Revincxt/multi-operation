"""决策前的本区及一阶邻区状态；只读仿真器，不推进状态或消耗随机数。"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


STATE_SCHEMA = "pre_action_grid_state_v1"
LOCAL_STATE_COLUMNS = (
    "start_waiting_request_num",
    "start_idle_driver_num",
    "start_busy_driver_num",
    "start_repositioning_driver_num",
)
NEIGHBOR_STATE_COLUMNS = tuple(f"neighbor_{name}" for name in LOCAL_STATE_COLUMNS)
STATE_COLUMNS = LOCAL_STATE_COLUMNS + NEIGHBOR_STATE_COLUMNS


def state_columns(columns):
    """旧数据不伪造当前状态；新数据必须完整包含八个状态列。"""
    present = set(columns).intersection(STATE_COLUMNS)
    if present and present != set(STATE_COLUMNS):
        raise ValueError
    return list(STATE_COLUMNS) if present else []


def validate_adjacency(adjacency, grid_num):
    adjacency = np.asarray(adjacency)
    if (adjacency.shape != (grid_num, grid_num)
            or not np.isin(adjacency, [0, 1]).all()
            or not np.array_equal(adjacency, adjacency.T)
            or np.diag(adjacency).any()):
        raise ValueError
    return adjacency.astype(np.int64)


def load_neighbor_adjacency(data_dir, grid_num):
    """读取已有六边形邻接，不用匹配半径或全城总量代替邻区。"""
    if grid_num == 1:
        return np.zeros((1, 1), dtype=np.int64)
    root = Path(data_dir)
    path = root / ("grid_adjacency.csv" if (root / "grid_manifest.json").is_file()
                   else "hexo_updated_r300_adj.csv")
    table = pd.read_csv(path)
    if list(table.columns) != [str(grid) for grid in range(grid_num)]:
        raise ValueError
    return validate_adjacency(table.to_numpy(), grid_num)


def state_metadata(adjacency):
    adjacency = validate_adjacency(adjacency, len(adjacency))
    return {
        "decision_state_schema": STATE_SCHEMA,
        "state_feature_columns": list(STATE_COLUMNS),
        "decision_state_timing": "before current action, cancellation, dispatch and arrivals",
        "neighbor_definition": "one-hop adjacent grids, sum, excluding own grid",
        "neighbor_adjacency": adjacency.tolist(),
        "neighbor_adjacency_sha256": hashlib.sha256(adjacency.astype('uint8').tobytes()).hexdigest(),
    }


def capture_decision_state(sim, adjacency):
    """订单按起点网格、司机按当前 grid_id 统计；忙碌含接驾及送客。"""
    adjacency = validate_adjacency(adjacency, sim.grid_num)

    def counts(values):
        values = np.asarray(values, dtype=float)
        if (not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all()
                or (values < 0).any() or (values >= sim.grid_num).any()):
            raise ValueError
        return np.bincount(values.astype(int), minlength=sim.grid_num)

    drivers = sim.driver_table
    local = np.column_stack([
        counts(sim.wait_requests['origin_grid_id']),
        counts(drivers.loc[drivers.status == 0, 'grid_id']),
        counts(drivers.loc[drivers.status.isin([1, 2]), 'grid_id']),
        counts(drivers.loc[drivers.status == 4, 'grid_id']),
    ])
    state = pd.DataFrame(np.column_stack([local, adjacency @ local]), columns=STATE_COLUMNS)
    state.insert(0, 'grid_id', np.arange(sim.grid_num))
    return state


def validate_decision_states(table, adjacency):
    """每周期覆盖全部网格，计数为非负整数，邻区和可由本区状态复算。"""
    grid_num = len(adjacency)
    adjacency = validate_adjacency(adjacency, grid_num)
    if not state_columns(table.columns):
        raise ValueError
    for _, period in table.groupby('decision_step'):
        period = period.sort_values('grid_id')
        if not np.array_equal(period.grid_id.to_numpy(), np.arange(grid_num)):
            raise ValueError
        values = period[list(STATE_COLUMNS)].to_numpy(dtype=float)
        if (not np.isfinite(values).all() or (values < 0).any()
                or not np.equal(values, np.floor(values)).all()
                or not np.array_equal(values[:, 4:], adjacency @ values[:, :4])):
            raise ValueError
