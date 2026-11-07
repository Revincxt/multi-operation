"""共享工具函数：司机采样、订单分配、状态更新等（纯网格级仿真器）"""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from src.utils.dispatch_alg import LD

# pandas 3.0+ defaults to PyArrow string dtype
pd.options.future.infer_string = False


# ==============================================================================
# 距离计算
# ==============================================================================

def distance_array(coord_1, coord_2):
    """批量 Haversine 距离 (km)。coord 为 [lng, lat] 格式。"""
    coord_1 = np.atleast_2d(np.asarray(coord_1, dtype=float))
    coord_2 = np.atleast_2d(np.asarray(coord_2, dtype=float))
    lng1, lat1 = coord_1[:, 0], coord_1[:, 1]
    lng2, lat2 = coord_2[:, 0], coord_2[:, 1]
    lat1, lng1 = np.radians(lat1), np.radians(lng1)
    lat2, lng2 = np.radians(lat2), np.radians(lng2)
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def distance(coord_1, coord_2):
    """单点 Haversine 距离。coord 为 (lng, lat)。"""
    lng1, lat1 = coord_1
    lng2, lat2 = coord_2
    return distance_array(
        np.array([[lng1, lat1]]),
        np.array([[lng2, lat2]]),
    )[0]


# ==============================================================================
# 订单分配
# ==============================================================================

def order_dispatch(wait_requests, driver_table, matching_radius_by_grid,
                   dispatch_method='LD', method='d'):
    """
    匹配等待订单与空闲司机。

    参数
    ----
    wait_requests : DataFrame
        含 origin_lng, origin_lat, order_id, weight 列。
    driver_table : DataFrame
        含 lng, lat, driver_id, status 列。
    matching_radius_by_grid : np.ndarray
        一维网格动作数组，以 grid ID 为下标。订单半径严格由其
        ``origin_grid_id`` 查表获得。
    dispatch_method : str
        分配算法，当前仅支持 'LD'。
    method : str
        奖励计算模式：'d' 按距离，其他按 weight。

    返回
    ----
    matched_pairs : list of [order_id, driver_id, reward, flag]
    pickup_distances : np.ndarray
        每对匹配的接客距离 (km)。
    """
    idle_driver_table = driver_table[driver_table['status'] == 0]
    if wait_requests.shape[0] == 0 or idle_driver_table.shape[0] == 0:
        return [], np.array([])

    if 'origin_grid_id' not in wait_requests:
        raise ValueError("origin_grid_id is required for grid-level matching radii")
    grid_ids = wait_requests['origin_grid_id'].to_numpy(dtype=int)
    radii_by_grid = np.asarray(matching_radius_by_grid, dtype=float)
    if radii_by_grid.ndim != 1:
        raise ValueError("matching_radius_by_grid must be one-dimensional")
    if np.any(grid_ids < 0) or np.any(grid_ids >= len(radii_by_grid)):
        raise ValueError("origin_grid_id is outside the matching-radius table")
    request_radii = radii_by_grid[grid_ids]
    if not np.all(np.isfinite(request_radii)) or np.any(request_radii <= 0):
        raise ValueError("matching_radius_by_grid values must be finite and positive")

    request_array_temp = wait_requests[['origin_lng', 'origin_lat', 'order_id', 'weight']].copy()
    driver_loc_array_temp = idle_driver_table[['lng', 'lat', 'driver_id']].copy()

    request_array_temp['order_id'] = request_array_temp['order_id'].astype(int)
    driver_loc_array_temp['driver_id'] = driver_loc_array_temp['driver_id'].astype(int)
    request_array_temp.set_index('order_id', inplace=True)
    driver_loc_array_temp.set_index('driver_id', inplace=True)

    order_data = wait_requests[['origin_lng', 'origin_lat', 'order_id', 'weight']].values
    driver_data = idle_driver_table[['lng', 'lat', 'driver_id']].values

    order_coords = order_data[:, :2].astype(np.float64)
    driver_coords = driver_data[:, :2].astype(np.float64)

    # BallTree 粗筛（lat/lng 转弧度，[lat, lng] 顺序）
    order_rad = np.radians(order_coords[:, ::-1])
    driver_rad = np.radians(driver_coords[:, ::-1])
    driver_tree = BallTree(driver_rad, metric='haversine')
    radius_rad = request_radii / 6371.0
    candidates = driver_tree.query_radius(order_rad, r=radius_rad)

    m_idx, n_idx = [], []
    for m, drv_list in enumerate(candidates):
        for n in drv_list:
            m_idx.append(m)
            n_idx.append(n)
    if not m_idx:
        return [], np.array([])

    # 精筛
    dists = distance_array(order_coords[m_idx], driver_coords[n_idx])
    valid = dists <= request_radii[np.asarray(m_idx)]
    if not np.any(valid):
        return [], np.array([])

    fm = np.array(m_idx)[valid]
    fn = np.array(n_idx)[valid]
    fd = dists[valid]
    f_order_ids = order_data[fm, 2]
    f_driver_ids = driver_data[fn, 2]

    # 构建 LD 输入
    # Radius controls edge eligibility only.  Use one shared offset so that an
    # order with a larger radius does not receive artificial matching priority.
    distance_reward_offset = float(request_radii.max()) + 1.0
    pair_list = []
    for i in range(len(fm)):
        reward = distance_reward_offset - fd[i] if method == 'd' else float(order_data[fm[i], 3])
        # LD中flag的作用是什么
        flag = fd[i] if method == 'd' else float(order_data[fm[i], 3])
        pair_list.append([f_order_ids[i], f_driver_ids[i], reward, flag])

    matched_pairs = LD(pair_list, method)
    if not matched_pairs:
        return [], np.array([])

    req_idx = np.array(matched_pairs)[:, 0].astype(float).astype(int)
    drv_idx = np.array(matched_pairs)[:, 1].astype(float).astype(int)

    req_coords = request_array_temp.loc[req_idx, ['origin_lng', 'origin_lat']].values
    drv_coords = driver_loc_array_temp.loc[drv_idx, ['lng', 'lat']].values
    pickup_distances = distance_array(drv_coords, req_coords)

    return matched_pairs, pickup_distances


# ==============================================================================
# 司机管理
# ==============================================================================

def sample_all_drivers(driver_info, t_initial, t_end, driver_sample_ratio=1):
    """
    初始化司机表，设置起始状态和必要字段。

    保留列：driver_id, start_time, end_time, lng, lat, grid_id, status,
            target_loc_lng, target_loc_lat, target_grid_id,
            remaining_time, matched_order_id,
            total_idle_time, time_to_last_cruising
    """
    sampled = (driver_info.copy() if driver_sample_ratio == 1
               else driver_info.sample(frac=driver_sample_ratio, random_state=42))

    sampled['status'] = 3
    # A driver is available at reset only when their shift already covers the
    # simulation start. Future shifts are activated by the per-step schedule update.
    loc_on = (sampled['start_time'] <= t_initial) & (sampled['end_time'] > t_initial)
    sampled.loc[loc_on, 'status'] = 0

    sampled['target_loc_lng'] = sampled['lng'].astype(float)
    sampled['target_loc_lat'] = sampled['lat'].astype(float)
    sampled['target_grid_id'] = sampled['grid_id'].astype(int)
    sampled['remaining_time'] = 0.0
    sampled['matched_order_id'] = 'None'
    sampled['total_idle_time'] = 0
    sampled['time_to_last_cruising'] = 0

    return sampled


def driver_online_offline_decision(driver_table, current_time):
    """根据排班时间更新司机上下线状态。"""
    should_on = driver_table.index[
        (driver_table['start_time'] <= current_time) & (driver_table['end_time'] > current_time)]
    should_off = driver_table.index[
        (driver_table['start_time'] > current_time) | (driver_table['end_time'] <= current_time)]
    eligible = driver_table.index[driver_table['status'].isin([0, 3])]
    driver_table.loc[should_on.intersection(eligible), 'status'] = 0
    driver_table.loc[should_off.intersection(eligible), 'status'] = 3
    return driver_table
