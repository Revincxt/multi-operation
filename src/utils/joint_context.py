"""将同一决策时刻的全网格状态和动作编码为共享模型特征。"""

import numpy as np
import pandas as pd

from src.env.observation import LOCAL_STATE_COLUMNS, state_columns


GLOBAL_STATE_COLUMNS = tuple(f"global_{column}" for column in LOCAL_STATE_COLUMNS)


def joint_action_columns(grid_ids):
    """返回稳定排序的全网格价格/半径列名。"""
    return tuple(
        name
        for grid_id in sorted(int(value) for value in grid_ids)
        for name in (f"joint_price_grid_{grid_id}", f"joint_radius_grid_{grid_id}")
    )


def joint_context_columns(grid_ids, include_global_state=True):
    """返回训练数据中联合动作上下文列的规范顺序。"""
    return (GLOBAL_STATE_COLUMNS if include_global_state else ()) + joint_action_columns(grid_ids)


def add_joint_context(period_metrics):
    """将全网格当期动作向量及全局司机/积压状态复制到每个网格行。"""
    keys = ["run_id", "decision_step"]
    required = set(keys + ["grid_id", "price_multiplier", "matching_radius_km"])
    if not required.issubset(period_metrics.columns) or period_metrics.duplicated(
            keys + ["grid_id"]).any():
        raise ValueError

    grids_by_run = {
        run_id: tuple(sorted(group.grid_id.astype(int).unique()))
        for run_id, group in period_metrics.groupby("run_id", sort=False)
    }
    if not grids_by_run or len(set(grids_by_run.values())) != 1:
        raise ValueError
    grid_ids = next(iter(grids_by_run.values()))
    if grid_ids != tuple(range(len(grid_ids))):
        raise ValueError

    expected = set(grid_ids)
    for _, group in period_metrics.groupby(keys, sort=False):
        if set(group.grid_id.astype(int)) != expected:
            raise ValueError

    values = period_metrics[keys + ["grid_id", "price_multiplier", "matching_radius_km"]].copy()
    wide = values.pivot(index=keys, columns="grid_id",
                        values=["price_multiplier", "matching_radius_km"])
    wide.columns = [
        (f"joint_price_grid_{int(grid_id)}" if metric == "price_multiplier"
         else f"joint_radius_grid_{int(grid_id)}")
        for metric, grid_id in wide.columns
    ]
    wide = wide.reset_index()
    extra_columns = list(joint_action_columns(grid_ids))

    local_columns = state_columns(period_metrics.columns)
    if local_columns:
        if not set(LOCAL_STATE_COLUMNS).issubset(local_columns):
            raise ValueError
        global_state = period_metrics.groupby(keys, sort=False)[list(LOCAL_STATE_COLUMNS)].sum()
        global_state.columns = list(GLOBAL_STATE_COLUMNS)
        wide = wide.merge(global_state.reset_index(), on=keys, validate="one_to_one")
        extra_columns = list(GLOBAL_STATE_COLUMNS) + extra_columns

    enriched = period_metrics.merge(wide, on=keys, how="left", validate="many_to_one")
    if enriched[extra_columns].isna().any().any() or len(enriched) != len(period_metrics):
        raise ValueError
    if not np.isfinite(enriched[extra_columns].to_numpy(dtype=float)).all():
        raise ValueError
    return enriched, extra_columns


def joint_schema(context_columns):
    """解析上下文中的全局状态和联合动作列，返回列名及网格编号。"""
    columns = list(context_columns)
    globals_present = [column for column in columns if column in GLOBAL_STATE_COLUMNS]
    action_columns = [column for column in columns
                      if column.startswith("joint_price_grid_")
                      or column.startswith("joint_radius_grid_")]
    if not action_columns:
        if globals_present:
            raise ValueError
        return (), (), ()
    grid_ids = sorted({int(column.rsplit("_", 1)[1]) for column in action_columns})
    expected_actions = list(joint_action_columns(grid_ids))
    if action_columns != expected_actions:
        raise ValueError
    if globals_present and globals_present != list(GLOBAL_STATE_COLUMNS):
        raise ValueError
    return tuple(grid_ids), tuple(globals_present), tuple(action_columns)
