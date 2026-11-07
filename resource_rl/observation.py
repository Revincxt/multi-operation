"""只读取当前任务和过去指标；固定物理量纲缩放，不拟合测试数据。"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.env.observation import capture_decision_state, STATE_COLUMNS
from src.utils.utilities import distance_array


HISTORY_COLUMNS = (
    "candidate_request_num", "matched_request_num", "gbv", "platform_revenue",
    "baseline_request_num", "submitted_request_num", "request_rate", "driver_offer_num",
    "driver_accept_num", "driver_accept_rate", "unmatched_cancel_num", "pickup_cancel_num",
    "unmatched_cancel_rate", "pickup_cancel_rate", "order_completion_rate",
    "mean_wait_to_match_seconds", "mean_pickup_time_seconds", "mean_pickup_distance_km",
    "idle_driver_num", "busy_driver_num", "repositioning_driver_num", "vehicle_occupancy_rate",
    "price_multiplier", "matching_radius_km")
TASK_COLUMNS = (
    "pickup_driver_num", "delivery_driver_num", "incoming_5min", "incoming_10min", "incoming_20min",
    "outgoing_release_5min", "outgoing_release_10min", "outgoing_release_20min",
    "queue_age_0_2min", "queue_age_2_5min", "queue_age_5_8min", "queue_age_over8min",
    "queue_age_mean", "queue_age_p90", "queue_age_max", "queue_deadline_within5min",
    "queue_fare_mean", "queue_fare_std", "queue_fare_p90", "queue_price_mean", "queue_price_std",
    "queue_trip_km_mean", "queue_trip_km_p90", "queue_trip_seconds_mean", "queue_trip_seconds_p90",
    "nearest_driver_p25_km", "nearest_driver_p50_km", "nearest_driver_p75_km")
EDGE_COLUMNS = ("adjacent", "center_distance_km", "queued_OD_num", "release_5min", "release_10min",
                "release_20min", "reachable_idle_num", "reachable_pair_density", "reachable_distance_km")


def release_seconds(drivers, delta=60.):
    """按当前仿真的分钟推进规则计算计划释放时间；接驾结束后还有送客阶段。"""
    status = drivers.status.to_numpy(int)
    remaining = drivers.remaining_time.to_numpy(float)
    result = np.maximum(np.ceil(np.maximum(remaining, 0.)/delta), 1.)*delta
    pickup = status == 2
    pending = drivers.get("_pending_trip_time", pd.Series(0., index=drivers.index)).to_numpy(float)
    result[pickup] += np.ceil(np.maximum(pending[pickup], 0.)/delta)*delta
    result[status == 0] = 0.
    return result


def coordinate_features(table, longitude, latitude, columns):
    xy = table[list(columns)].to_numpy(float)
    return (xy-np.array([longitude, latitude]))*np.array([111.195*np.cos(np.radians(latitude)), 111.195])/4.


class ResourceObservation:
    def __init__(self, config, adjacency):
        self.config, self.adjacency = config, adjacency
        self.grids = config["grid_num"]
        self.unit = max(config["driver_num"]/self.grids, 1.)
        self.radii = np.asarray(config["matching_radius_values"], dtype=float)
        self.maximum_radius = float(self.radii.max())
        info = pd.read_csv(Path(config["data_dir"])/"grid_info.csv").sort_values("grid_id")
        if info.grid_id.tolist() != list(range(self.grids)):
            raise ValueError
        self.centers = info[["center_lon", "center_lat"]].to_numpy(float)
        self.longitude, self.latitude = self.centers.mean(0)
        first = np.repeat(self.centers, self.grids, axis=0)
        second = np.tile(self.centers, (self.grids, 1))
        self.distances = distance_array(first, second).reshape(self.grids, self.grids)
        current = list(STATE_COLUMNS)+["global_"+name for name in STATE_COLUMNS[:4]]
        self.node_columns = (current+list(HISTORY_COLUMNS)+["elapsed_fraction", "remaining_fraction"]
            +list(TASK_COLUMNS)+[f"reachable_fraction_r{r:g}" for r in self.radii]
            +[f"eligible_idle_per_order_r{r:g}" for r in self.radii])
        self.driver_size = 2+2*self.grids+4+4+2
        self.order_size = 4+2*self.grids+7

    def metadata(self):
        return {"schema_version": "resource_state_v1", "grid_count": self.grids,
            "node_columns": self.node_columns, "edge_columns": list(EDGE_COLUMNS),
            "driver_size": self.driver_size, "order_size": self.order_size,
            "count_unit": self.unit, "position_unit_km": 4., "fare_unit_usd": 20.,
            "time_unit_seconds": 1200., "release_horizons_seconds": [300, 600, 1200],
            "information": "current queues, committed tasks, estimated travel times, static geography, past outcomes; no future request rows or random draws",
            "critic_information": "individual current driver/task and queued-order tokens; both experimental structures receive the same inputs",
            "edge_orientation": "source current driver grid / order origin -> destination or order-origin grid",
            "normalization": "fixed physical scales from configuration; no train/validation/test fit"}

    def _history(self, periods):
        if periods is None:
            return np.zeros((self.grids, len(HISTORY_COLUMNS)))
        latest = periods[periods.decision_step == periods.decision_step.max()].set_index("grid_id").reindex(range(self.grids))
        values = latest[list(HISTORY_COLUMNS)].fillna(0.).to_numpy(float)
        scales = []
        for name in HISTORY_COLUMNS:
            if name in ("gbv", "platform_revenue"):
                scale = 100.
            elif name in ("mean_wait_to_match_seconds", "mean_pickup_time_seconds"):
                scale = 600.
            elif name == "mean_pickup_distance_km" or name == "matching_radius_km":
                scale = self.maximum_radius
            elif name == "price_multiplier":
                scale = 2.
            elif name.endswith("_num"):
                scale = self.unit
            else:
                scale = 1.
            scales.append(scale)
        return values/np.asarray(scales)

    def capture(self, sim, periods, previous_actions):
        g, unit = self.grids, self.unit
        drivers, queue = sim.driver_table, sim.wait_requests
        status = drivers.status.to_numpy(int)
        source = drivers.grid_id.to_numpy(int)
        destination = drivers.target_grid_id.to_numpy(int)
        active = np.isin(status, [1, 2, 4])
        eta = release_seconds(drivers, sim.delta_t)
        local = capture_decision_state(sim, self.adjacency)[list(STATE_COLUMNS)].to_numpy(float)
        global_counts = np.broadcast_to(local[:, :4].sum(0), (g, 4))
        fraction = sim.current_step/self.config["simulation_steps"]
        nodes = np.column_stack([local/unit, global_counts/unit, self._history(periods),
                                 np.full(g, fraction), np.full(g, 1.-fraction)])
        edges = np.zeros((g, g, len(EDGE_COLUMNS)), dtype=float)
        edges[..., 0], edges[..., 1] = self.adjacency, self.distances/self.maximum_radius
        for index, horizon in enumerate((300., 600., 1200.)):
            selected = active & (eta <= horizon)
            np.add.at(edges[..., 3+index], (source[selected], destination[selected]), 1./unit)
        if len(queue):
            np.add.at(edges[..., 2], (queue.origin_grid_id.to_numpy(int), queue.dest_grid_id.to_numpy(int)), 1./unit)
        idle = drivers[status == 0]
        # 距离是当前位置的确定函数，不调用匹配器，也不消耗仿真随机数。
        if len(queue) and len(idle):
            a = np.repeat(queue[["origin_lng", "origin_lat"]].to_numpy(float), len(idle), axis=0)
            b = np.tile(idle[["lng", "lat"]].to_numpy(float), (len(queue), 1))
            distances = distance_array(a, b).reshape(len(queue), len(idle))
        else:
            distances = np.zeros((len(queue), len(idle)))
        task = []
        for grid in range(g):
            part = queue[queue.origin_grid_id == grid]
            age = part.wait_time.to_numpy(float)
            values = [np.sum((source == grid)&(status == code))/unit for code in (2, 1)]
            values += [edges[:, grid, k].sum() for k in (3, 4, 5)]
            values += [edges[grid, :, k].sum() for k in (3, 4, 5)]
            values += (np.histogram(age, [0., 120., 300., 480., np.inf])[0]/unit).tolist()
            def stats(column, std=False):
                x = part[column].to_numpy(float)
                return ([float(x.mean()), float(x.std()), float(np.quantile(x, .9))] if std
                        else [float(x.mean()), float(np.quantile(x, .9))]) if len(x) else ([0.]* (3 if std else 2))
            values += ([float(age.mean())/600., float(np.quantile(age, .9))/600., float(age.max())/600.] if len(age) else [0.]*3)
            values += [float(np.sum(part.maximum_wait_time.to_numpy(float)-age <= 300.))/unit]
            values += [x/20. for x in stats("quoted_fare", True)]
            prices = part.price_multiplier.to_numpy(float)
            values += [float(prices.mean())/2., float(prices.std())/2.] if len(prices) else [0., 0.]
            values += [x/10. for x in stats("trip_distance")]
            values += [x/1200. for x in stats("trip_time")]
            selected = distances[queue.origin_grid_id.to_numpy(int) == grid]
            nearest = selected.min(1) if len(part) and len(idle) else np.full(len(part), self.maximum_radius*2.)
            values += (np.quantile(nearest, [.25, .5, .75])/self.maximum_radius).tolist() if len(nearest) else [0.]*3
            values += [float(np.mean(nearest <= radius)) if len(nearest) else 0. for radius in self.radii]
            values += [float(np.mean((selected <= radius).sum(1)))/unit if len(part) and len(idle) else 0. for radius in self.radii]
            task.append(values)
            for driver_grid in range(g):
                block = selected[:, idle.grid_id.to_numpy(int) == driver_grid]
                feasible = block <= self.maximum_radius
                if block.size:
                    edges[driver_grid, grid, 6] = feasible.any(0).sum()/unit
                    edges[driver_grid, grid, 7] = feasible.sum()/unit**2
                    edges[driver_grid, grid, 8] = block[feasible].mean()/self.maximum_radius if feasible.any() else 0.
        nodes = np.column_stack((nodes, np.asarray(task)))
        identity = np.eye(g)
        driver_tokens = np.column_stack((coordinate_features(drivers, self.longitude, self.latitude, ("lng", "lat")),
            identity[source], identity[np.where(active, destination, source)],
            np.column_stack([status == code for code in (0, 1, 2, 4)]),
            drivers.remaining_time.to_numpy(float).clip(0)/1200.,
            np.where(status == 2, drivers.get("_pending_trip_time", pd.Series(0., index=drivers.index)).to_numpy(float), 0.)/1200.,
            eta/1200., drivers.total_idle_time.to_numpy(float)/1200.,
            coordinate_features(drivers, self.longitude, self.latitude, ("target_loc_lng", "target_loc_lat"))))
        # 空闲司机的目标字段可能为历史值；将其清零，避免无效坐标放大。
        driver_tokens[~active, -2:] = 0.
        order_tokens = np.column_stack((coordinate_features(queue, self.longitude, self.latitude, ("origin_lng", "origin_lat")),
            coordinate_features(queue, self.longitude, self.latitude, ("dest_lng", "dest_lat")),
            identity[queue.origin_grid_id.to_numpy(int)], identity[queue.dest_grid_id.to_numpy(int)],
            queue.wait_time.to_numpy(float)/600., queue.maximum_wait_time.to_numpy(float)/600.,
            queue.quoted_fare.to_numpy(float)/20., queue.price_multiplier.to_numpy(float)/2.,
            queue.trip_distance.to_numpy(float)/10., queue.trip_time.to_numpy(float)/1200.,
            (queue.maximum_wait_time.to_numpy(float)-queue.wait_time.to_numpy(float))/600.))
        result = {"nodes": nodes.astype("float32"), "edges": edges.astype("float32"),
            "drivers": driver_tokens.astype("float32"), "orders": order_tokens.astype("float32"),
            "previous_actions": np.asarray(previous_actions, dtype="int64").copy()}
        if (nodes.shape != (g, len(self.node_columns)) or driver_tokens.shape[1] != self.driver_size
                or order_tokens.shape[1] != self.order_size or not all(np.isfinite(x).all() for x in result.values())):
            raise ValueError
        return result
