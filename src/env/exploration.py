"""空间 block 随机化探索计划生成。

核心逻辑:
1. 预计算（带缓存）所有 grid 的 R=1、R=2 邻域和冲突矩阵
2. 贪心随机独立集选取 treatment cores（间距 ≥ 5 hops，R=2 邻域不重叠）
3. N 个事件均匀分布在仿真时段内（避开首尾 margin）
4. 独立加权采样离散价格与匹配半径，组成联合 action

使用方式:
    from src.env.exploration import ExplorationPlanner
    planner = ExplorationPlanner(adjacency_matrix, grid_num=263)
    events, meta = planner.build_plan(rng, n_events=10, simulation_steps=300)
"""

import os
import numpy as np

from src.env.actions import DEFAULT_MATCHING_RADII_KM, DEFAULT_PRICE_MULTIPLIERS


def _compute_distance_mask(adj, max_dist):
    """返回 bool 矩阵 mask[i,j] = True if dist(i,j) <= max_dist。

    用 BFS 从每个节点出发，计算到其他节点的最短路径距离。
    """
    n = adj.shape[0]
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        mask[i, i] = True
        visited = {i}
        frontier = {i}
        for _ in range(max_dist):
            if not frontier:
                break
            next_frontier = set()
            for u in frontier:
                for v in np.where(adj[u])[0]:
                    v = int(v)
                    if v not in visited:
                        visited.add(v)
                        mask[i, v] = True
                        next_frontier.add(v)
            frontier = next_frontier
    return mask


_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'my_data')


def _cache_path(adjacency_path, r_exposure):
    """生成缓存文件路径。"""
    base = os.path.splitext(os.path.basename(adjacency_path))[0]
    return os.path.join(_CACHE_DIR, f'_exploration_r{r_exposure}_{base}.npz')


def _load_or_build_masks(adjacency, r_exposure, adjacency_path):
    """加载或构建 mask 缓存。

    首次计算后保存为 .npz，后续直接从缓存加载。
    返回 (r1_mask, r2_mask, conflict_mask, r1_neighbors, r2_neighbors, conflicts)。
    """
    cache_file = _cache_path(adjacency_path, r_exposure)
    if os.path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        return (data['r1_mask'], data['r2_mask'], data['conflict_mask'],
                data['r1_neighbors'].tolist(), data['r2_neighbors'].tolist(),
                data['conflicts'].tolist())

    grid_num = adjacency.shape[0]
    r1_mask = _compute_distance_mask(adjacency, 1)
    r2_mask = _compute_distance_mask(adjacency, r_exposure)
    conflict_mask = _compute_distance_mask(adjacency, 2 * r_exposure)

    r1_neighbors = [set(np.where(r1_mask[i])[0].tolist())
                    for i in range(grid_num)]
    r2_neighbors = [set(np.where(r2_mask[i])[0].tolist())
                    for i in range(grid_num)]

    conflicts = [set() for _ in range(grid_num)]
    for i in range(grid_num):
        for j in np.where(conflict_mask[i])[0]:
            j = int(j)
            if j > i:
                conflicts[i].add(j)
                conflicts[j].add(i)

    os.makedirs(_CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache_file,
                        r1_mask=r1_mask,
                        r2_mask=r2_mask,
                        conflict_mask=conflict_mask,
                        r1_neighbors=np.array(r1_neighbors, dtype=object),
                        r2_neighbors=np.array(r2_neighbors, dtype=object),
                        conflicts=np.array(conflicts, dtype=object))

    return (r1_mask, r2_mask, conflict_mask,
            r1_neighbors, r2_neighbors, conflicts)


class ExplorationPlanner:
    """空间 block 随机化探索计划器。

    Parameters
    ----------
    adjacency : np.ndarray, shape (grid_num, grid_num)
        对称的二值邻接矩阵。adj[i,j] = 1 表示 grid i 和 grid j 相邻。
    grid_num : int
        Grid 总数。
    r_exposure : int, default=2
        空间暴露半径。任意两个 treatment core 的 R=2 邻域不能重叠，
        即 core 间距 ≥ 2*R_exposure + 1 = 5 hops。
    treatment_minutes : int, default=10
        Treatment 持续时间（分钟）。
    price_action_weights : dict, optional
        {multiplier: weight} 价格动作及抽样权重。默认在统一候选集合中
        均匀抽样，排除仅用于 baseline control 的 1.0。
    matching_radius_weights : dict, optional
        {radius_km: weight} 匹配半径动作及抽样权重。默认在统一候选集合中
        均匀抽样。
    adjacency_path : str, optional
        邻接矩阵文件路径，用于缓存 key。
    """

    def __init__(self, adjacency, grid_num=263, r_exposure=2,
                 treatment_minutes=10, price_action_weights=None,
                 matching_radius_weights=None,
                 adjacency_path=None):
        if price_action_weights is None:
            price_action_weights = {
                price: 1.0 for price in DEFAULT_PRICE_MULTIPLIERS if price != 1.0
            }
        if matching_radius_weights is None:
            matching_radius_weights = {
                radius: 1.0 for radius in DEFAULT_MATCHING_RADII_KM
            }
        if adjacency_path is None:
            adjacency_path = os.path.join(_CACHE_DIR, 'hexo_updated_r300_adj.csv')

        self.grid_num = grid_num
        self.r_exposure = r_exposure
        self.treatment_minutes = treatment_minutes

        self.price_action_weights = self._normalize_weights(
            price_action_weights, 'price_action_weights')
        self.matching_radius_weights = self._normalize_weights(
            matching_radius_weights, 'matching_radius_weights')
        self.price_action_values = np.array(
            sorted(self.price_action_weights), dtype=float)
        self.matching_radius_values = np.array(
            sorted(self.matching_radius_weights), dtype=float)

        # 预计算 mask（带磁盘缓存）
        (self._r1_mask, self._r2_mask, self._conflict_mask,
         self._r1_neighbors, self._r2_neighbors, self._conflicts) = \
            _load_or_build_masks(adjacency, r_exposure, adjacency_path)

    def _classify_grids_for_cores(self, cores):
        """对一组 cores，将所有 grids 按空间距离分类。

        Returns
        -------
        dict with keys:
            treated_core: list of int
            spillover_r1: list of int (距离 = 1, 排除 core 自身)
            spillover_r2: list of int (距离 = 2, 排除 core 和 R1)
            clean_control: list of int (距离 > 2)
        """
        cores_set = set(cores)
        r1_set = set()
        for c in cores:
            r1_set.update(self._r1_neighbors[c])
        r1_set -= cores_set

        r2_set = set()
        for c in cores:
            r2_set.update(self._r2_neighbors[c])
        r2_set -= cores_set
        r2_only = r2_set - r1_set

        clean = set(range(self.grid_num)) - cores_set - r1_set - r2_only

        return {
            'treated_core': sorted(cores_set),
            'spillover_r1': sorted(r1_set),
            'spillover_r2': sorted(r2_only),
            'clean_control': sorted(clean),
        }

    def _sample_treatment_block(self, rng):
        """贪心随机选取互不冲突的 treatment cores。

        算法: 从剩余候选集中随机取一个 grid，加入已选，
        然后移除它的所有冲突 grid（距离 ≤ 2*R_exposure），
        重复直到候选集为空。

        Returns
        -------
        selected : list of int
            被选中的 treatment core grid IDs。
        """
        selected = []
        blocked_core = set()
        available = list(range(self.grid_num))

        while available:
            available = [g for g in available if g not in blocked_core]
            if not available:
                break

            i = rng.integers(0, len(available))
            core = available[i]
            selected.append(core)
            blocked_core.update(
                int(j) for j in np.where(self._conflict_mask[core])[0])

        return selected

    @staticmethod
    def _normalize_weights(weights, name):
        """Validate and normalize one discrete action distribution."""
        if not weights:
            raise ValueError(f"{name} must not be empty")
        values = np.asarray(list(weights.keys()), dtype=float)
        raw_weights = np.asarray(list(weights.values()), dtype=float)
        if (not np.all(np.isfinite(values)) or np.any(values <= 0)
                or not np.all(np.isfinite(raw_weights))
                or np.any(raw_weights < 0) or raw_weights.sum() <= 0):
            raise ValueError(
                f"{name} must contain positive finite actions and "
                "non-negative finite weights with positive total")
        total = float(raw_weights.sum())
        return {float(value): float(weight / total)
                for value, weight in zip(values, raw_weights)}

    def _sample_joint_action(self, rng):
        """Independently sample one price/radius pair for an event."""
        price_probabilities = [
            self.price_action_weights[v] for v in self.price_action_values]
        radius_probabilities = [
            self.matching_radius_weights[v]
            for v in self.matching_radius_values]
        price_idx = rng.choice(
            len(self.price_action_values), p=price_probabilities)
        radius_idx = rng.choice(
            len(self.matching_radius_values), p=radius_probabilities)
        return (
            float(self.price_action_values[price_idx]),
            float(self.matching_radius_values[radius_idx]),
        )

    def build_plan(self, rng, n_events=10, simulation_steps=300,
                   margin_minutes=15):
        """生成探索计划：N 个事件均匀分布在仿真时段内。

        Parameters
        ----------
        rng : np.random.Generator
        n_events : int
            事件数量，默认 10。
        simulation_steps : int
            仿真总步数（分钟），默认 300（5 小时）。
        margin_minutes : int
            首尾 margin（分钟）。

        Returns
        -------
        events : list of dict
            每个 event 含 treatment 参数和 per-zone grid 分类。
        metadata : dict
        """
        t_min = margin_minutes
        t_max = simulation_steps - self.treatment_minutes - margin_minutes
        if t_max <= t_min:
            raise ValueError(
                f"Not enough room: usable [{t_min}, {t_max}) with "
                f"margin={margin_minutes}, treatment={self.treatment_minutes}")

        treatment_starts = np.linspace(t_min, t_max, n_events, dtype=int)

        events = []
        for event_id, t_start in enumerate(treatment_starts):
            cores = self._sample_treatment_block(rng)
            price, matching_radius = self._sample_joint_action(rng)
            zones = self._classify_grids_for_cores(cores)

            events.append({
                'exploration_event_id': event_id,
                'treatment_start_step': int(t_start),
                'treatment_end_step': int(t_start + self.treatment_minutes),
                'exploration_price_multiplier': price,
                'matching_radius_km': matching_radius,
                'exploration_grid_ids': [int(c) for c in sorted(cores)],
                'n_cores': len(cores),
                'zones': zones,
            })

        metadata = {
            'treatment_minutes': self.treatment_minutes,
            'margin_minutes': margin_minutes,
            'r_exposure': self.r_exposure,
            'price_action_weights': self.price_action_weights,
            'matching_radius_weights': self.matching_radius_weights,
            'joint_action_sampling': 'independent_product',
            'simulation_steps': simulation_steps,
            'n_events': len(events),
        }

        return events, metadata

    @staticmethod
    def load_adjacency(csv_path):
        """从 CSV 文件加载邻接矩阵。"""
        return np.loadtxt(csv_path, delimiter=',', dtype=int, skiprows=1)
