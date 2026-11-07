"""网约车仿真器环境 — 核心仿真循环，支持网格级联合动作。

使用方式:
    actions = {g: GridAction(1.0, 1.0) for g in range(263)}
    sim = Simulator(grid_num=263, grid_action=actions,
                    price_response_model='utility_choice', ...)
    sim.reset(seed=42, request_databases=request_dict, driver_info=driver_df)
    while not sim.end_of_episode:
        sim.step()
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path

# pandas 3.0+ defaults to PyArrow string dtype, which rejects numeric
# values in string columns.  Disable for compatibility.
pd.options.future.infer_string = False

from src.env.simulator_pattern import SimulatorPattern
from src.env.actions import (
    DEFAULT_MATCHING_RADII_KM,
    DEFAULT_PRICE_MULTIPLIERS,
    GridAction,
)
from src.env.price_response import (
    DriverOffer, PassengerMatch, PassengerOffer,
    PriceResponseModel, create_price_response_model,
)
from src.utils.utilities import sample_all_drivers, order_dispatch, driver_online_offline_decision


class Simulator:
    """核心仿真环境（网格级别）。按时间步推进，包含订单生成、匹配、价格响应、状态更新。"""

    def __init__(self, mapping_dict=None, road_network=None, **kwargs):
        # road_network is accepted for API compatibility but unused in grid-level simulation

        # 已移除的行为开关必须显式拒绝，避免实验配置被静默忽略。
        if any(name in kwargs for name in (
                'post_match_cancellation', 'driver_supply_response', 'repo_mode')):
            raise TypeError

        legacy_action_args = [
            name for name in (
                'price_multiplier', 'maximal_pickup_distance',
                'allowed_maximal_pickup_distances')
            if name in kwargs
        ]
        if legacy_action_args:
            raise TypeError(
                f"legacy action arguments {legacy_action_args} are no longer supported; "
                "use grid_action and allowed_matching_radii_km"
            )

        # === 基础参数 ===
        self.base_fare_start = float(kwargs.get('base_fare_start', 3.0))
        self.price_per_km = float(kwargs.get('price_per_km', 0.7))
        configured_grid_action = kwargs.get('grid_action')
        self.grid_action = None
        self._grid_action_decision_index = None
        self._price_by_grid = None
        self._radius_by_grid = None
        self.commission_rate = float(kwargs.get('commission_rate', 0.25))
        self.allowed_price_multipliers = np.asarray(
            kwargs.get('allowed_price_multipliers', DEFAULT_PRICE_MULTIPLIERS),
            dtype=float,
        )
        if np.any(self.allowed_price_multipliers <= 0):
            raise ValueError("allowed_price_multipliers must be strictly positive")

        # === 价格响应模型 ===
        response_params = kwargs.get('price_response_parameters', {})
        response_config = kwargs.get('price_response_config', None)
        configured_model = kwargs.get('price_response_model', None)
        self.passenger_response_model = self._build_response_model(
            kwargs.get('passenger_response_model', configured_model),
            kwargs.get('passenger_response_parameters', response_params),
            kwargs.get('passenger_response_config', response_config),
        )
        self.driver_response_model = self._build_response_model(
            kwargs.get('driver_response_model', configured_model),
            kwargs.get('driver_response_parameters', response_params),
            kwargs.get('driver_response_config', response_config),
        )

        # === 需求/供给参数 ===
        self.demand_input_mode = kwargs.get('demand_input_mode', 'observed_requests')
        self.demand_reference_price_multiplier = float(kwargs.get('demand_reference_price_multiplier', 1.0))
        # Cap expected low-price demand expansion at five copies per observed order.
        self.maximum_demand_multiplier = float(kwargs.get('maximum_demand_multiplier', 5.0))
        # Cancellation is modelled separately while unmatched and while waiting
        # for pickup. It is enabled by default for the utility response model.
        self.unmatched_cancellation = bool(kwargs.get('unmatched_cancellation', True))
        self.pickup_cancellation = bool(kwargs.get('pickup_cancellation', True))
        # All supplied drivers are online for the entire simulation episode.
        # Price-sensitive participation and scheduled online/offline transitions
        # are deliberately out of scope for this pricing environment.

        # === Idle-driver repositioning ===
        self.enable_reposition = bool(kwargs.get('enable_reposition', False))
        # Smoothstep trigger reaches probability 1 at this idle duration.
        self.reposition_idle_threshold_seconds = float(
            kwargs.get('reposition_idle_threshold_seconds', 600.0))
        self.reposition_income_window_minutes = int(
            kwargs.get('reposition_income_window_minutes', 10))
        # Choice utility is a 10-minute income total in USD, not USD/min.
        self.reposition_logit_temperature = float(
            kwargs.get('reposition_logit_temperature', 10.0))
        self.reposition_cost_per_km = float(kwargs.get('reposition_cost_per_km', 0.3))
        self.reposition_adjacency_path = kwargs.get(
            'reposition_adjacency_path', 'my_data/hexo_updated_r300_adj.csv')
        self.reposition_grid_info_path = kwargs.get(
            'reposition_grid_info_path', 'my_data/hex_updated_r300_info.csv')
        if self.reposition_idle_threshold_seconds <= 0:
            raise ValueError("reposition_idle_threshold_seconds must be positive")
        if self.reposition_income_window_minutes <= 0:
            raise ValueError("reposition_income_window_minutes must be positive")
        if self.reposition_logit_temperature <= 0:
            raise ValueError("reposition_logit_temperature must be positive")
        if self.reposition_cost_per_km < 0:
            raise ValueError("reposition_cost_per_km must be non-negative")

        # === 时间 ===
        self.t_initial = 18000   # 5:00 AM (秒)
        self.t_end = 36000       # 10:00 AM
        self.delta_t = 60        # 1 分钟/步
        self.vehicle_speed = 22.788
        self.time = None
        self.current_step = None
        self.finish_run_step = int((self.t_end - self.t_initial) // self.delta_t)
        self.end_of_episode = 0

        # === 网格 ===
        self.grid_num = kwargs.get('grid_num', 35)
        self.decision_freq = int(kwargs.get('decision_freq', 10))
        if self.decision_freq <= 0:
            raise ValueError("decision_freq must be positive")
        self.set_grid_action(configured_grid_action)
        self.experiment_date = None
        self.mapping_dict = mapping_dict  # 看一下这个dict长什么样子

        # === 订单参数 ===
        # Sampling is retained only for controlled simulation-scale experiments.
        # Production trajectory collection should normally use the full fixed input data.
        self.order_sample_ratio = kwargs.get('order_sample_ratio', 1)
        self.request_interval = 60
        self.maximum_wait_time_mean = float(kwargs.get('maximum_wait_time_mean', 600))
        self.allowed_matching_radii_km = np.asarray(
            kwargs.get('allowed_matching_radii_km', DEFAULT_MATCHING_RADII_KM),
            dtype=float,
        )
        if (self.allowed_matching_radii_km.ndim != 1
                or len(self.allowed_matching_radii_km) == 0
                or not np.all(np.isfinite(self.allowed_matching_radii_km))
                or np.any(self.allowed_matching_radii_km <= 0)):
            raise ValueError(
                "allowed_matching_radii_km must be a non-empty list of "
                "finite positive values")
        self.dispatch_method = kwargs.get('dispatch_method', 'LD')
        # Legacy dispatch scoring mode: 'd' prioritises short pickup distance;
        # other values prioritise order weight. It is not a pricing action.
        self.dispatch_scoring_mode = kwargs.get('dispatch_scoring_mode', kwargs.get('method', 'd'))

        # === 司机参数 ===
        # See order_sample_ratio: keep at 1.0 for main experiments.
        self.driver_sample_ratio = kwargs.get('driver_sample_ratio', 1)
        driver_num = kwargs.get('driver_num', 1000)
        if (isinstance(driver_num, bool)
                or int(driver_num) != driver_num
                or driver_num <= 0):
            raise ValueError("driver_num must be a positive integer")
        self.driver_num = int(driver_num)

        # === 列定义 ===
        self.request_columns = [
            'order_id', 'origin_id', 'origin_lat', 'origin_lng', 'dest_id', 'dest_lat', 'dest_lng',
            'trip_distance', 'start_time', 'origin_grid_id', 'dest_grid_id',
            'trip_time', 'cancel_prob', 't_matched',
            'pickup_time', 'wait_time', 't_end', 'status', 'driver_id', 'maximum_wait_time',
            'designed_reward', 'pickup_distance', 'base_fare', 'price_multiplier', 'quoted_fare',
            'commission_rate', 'driver_payment', 'platform_revenue',
            'passenger_request_probability', 'baseline_passenger_request_probability',
            'demand_adjustment_factor', 'unmatched_cancel_probability',
            'pickup_cancel_probability', 'driver_accept_probability',
        ]
        self.driver_columns = [
            'driver_id', 'start_time', 'end_time', 'lng', 'lat', 'grid_id', 'status',
            'target_loc_lng', 'target_loc_lat', 'target_grid_id', 'remaining_time',
            'matched_order_id', 'total_idle_time', 'time_to_last_cruising',
        ]
        self.reposition_candidates = None
        self.grid_centroids = None
        self.reposition_income_history = None
        self.reposition_income_sum = None
        self.reposition_events = None
        self.total_reposition_distance_km = 0.0
        self.total_reposition_cost = 0.0


        self.rng = None
        self.seed = None

    # =====================================================================
    # 网格动作
    # =====================================================================
    def set_grid_action(self, grid_action):
        """Set the complete per-grid platform action source.

        ``grid_action`` may be:

        - a complete ``{grid_id: GridAction}`` mapping; or
        - a callable receiving ``(sim_time, grid_state_dataframe)`` and
          returning such a complete mapping.

        The source is evaluated exactly once at the start of each decision
        period and the result is held for ``decision_freq`` simulator steps.
        Every grid must be present, so all orders from the same origin grid use
        the same price and matching radius in that period.

        """
        if grid_action is None:
            raise TypeError("grid_action is required")
        if not callable(grid_action) and not isinstance(grid_action, Mapping):
            raise TypeError(
                "grid_action must be a complete grid mapping or a callable")
        self.grid_action = grid_action
        self._invalidate_grid_action_cache()

    def _invalidate_grid_action_cache(self):
        self._grid_action_decision_index = None
        self._price_by_grid = None
        self._radius_by_grid = None

    def _validate_action_values(self, prices, radii):
        prices = np.asarray(prices, dtype=float)
        radii = np.asarray(radii, dtype=float)
        if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
            raise ValueError("price_multiplier values must be finite and positive")
        allowed_prices = np.isclose(
            prices[:, None], self.allowed_price_multipliers[None, :]).any(axis=1)
        if not np.all(allowed_prices):
            invalid = np.unique(prices[~allowed_prices]).tolist()
            raise ValueError(
                f"price_multiplier must use the discrete action set "
                f"{self.allowed_price_multipliers.tolist()}; got {invalid}"
            )
        if not np.all(np.isfinite(radii)) or np.any(radii <= 0):
            raise ValueError("matching_radius_km values must be finite and positive")
        allowed_radii = np.isclose(
            radii[:, None], self.allowed_matching_radii_km[None, :]).any(axis=1)
        if not np.all(allowed_radii):
            invalid = np.unique(radii[~allowed_radii]).tolist()
            raise ValueError(
                f"matching_radius_km must use the discrete action set "
                f"{self.allowed_matching_radii_km.tolist()}; got {invalid}"
            )

    def _normalize_grid_action_mapping(self, source):
        """Validate one decision period's complete grid-level action mapping."""
        if not isinstance(source, Mapping):
            raise TypeError("grid_action must resolve to a complete grid mapping")

        normalized = {}
        for raw_grid_id, action in source.items():
            if isinstance(raw_grid_id, bool) or not isinstance(raw_grid_id, Integral):
                raise TypeError("grid action keys must be integer grid IDs")
            grid_id = int(raw_grid_id)
            if grid_id in normalized:
                raise ValueError(f"duplicate grid action for grid {grid_id}")
            if not isinstance(action, GridAction):
                raise TypeError("every grid action value must be a GridAction")
            normalized[grid_id] = action

        expected = set(range(self.grid_num))
        actual = set(normalized)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise ValueError(
                "grid_action must contain exactly one action for every grid; "
                f"missing={missing}, extra={extra}"
            )

        prices = np.asarray(
            [normalized[g].price_multiplier for g in range(self.grid_num)], dtype=float)
        radii = np.asarray(
            [normalized[g].matching_radius_km for g in range(self.grid_num)], dtype=float)
        self._validate_action_values(prices, radii)
        return prices, radii

    def _current_grid_action_arrays(self):
        """Return cached ``price[grid]`` and ``radius[grid]`` for this period."""
        if self.grid_action is None:
            raise RuntimeError("no grid_action is configured")

        current_step = 0 if self.current_step is None else self.current_step
        decision_index = current_step // self.decision_freq
        if self._grid_action_decision_index != decision_index:
            source = self.grid_action
            if callable(source):
                if hasattr(self, 'evaluate_df'):
                    grid_state = self.evaluate_df.copy()
                else:
                    grid_state = pd.DataFrame(
                        {'origin_grid_id': np.arange(self.grid_num)})
                source = source(self.time, grid_state)
            prices, radii = self._normalize_grid_action_mapping(source)
            self._price_by_grid = prices
            self._radius_by_grid = radii
            self._grid_action_decision_index = decision_index
        return self._price_by_grid, self._radius_by_grid

    @staticmethod
    def _grid_values_for_rows(rows, values_by_grid):
        """Look up one cached grid-level value for every order row."""
        if 'origin_grid_id' not in rows:
            raise ValueError("grid action lookup requires origin_grid_id")
        grid_ids = rows['origin_grid_id'].to_numpy(dtype=int)
        if np.any(grid_ids < 0) or np.any(grid_ids >= len(values_by_grid)):
            invalid = np.unique(
                grid_ids[(grid_ids < 0) | (grid_ids >= len(values_by_grid))]).tolist()
            raise ValueError(f"origin_grid_id is outside the action table: {invalid}")
        return values_by_grid[grid_ids]

    def current_grid_actions(self):
        """Return the cached action table for the current decision period."""
        prices, radii = self._current_grid_action_arrays()
        return pd.DataFrame({
            'grid_id': np.arange(self.grid_num),
            'price_multiplier': prices.copy(),
            'matching_radius_km': radii.copy(),
        })

    def _prices_for_orders(self, orders):
        """Look up each order's price from the current origin-grid action."""
        prices, _ = self._current_grid_action_arrays()
        return self._grid_values_for_rows(orders, prices)

    @staticmethod
    def _build_response_model(model, parameters=None, config=None):
        """Instantiate a configured response model, preserving legacy no-model mode."""
        if model is None or model == 'legacy':
            return None
        if isinstance(model, PriceResponseModel):
            return model
        if config is not None:
            return create_price_response_model(model, config=config)
        return create_price_response_model(model, parameters=dict(parameters or {}))

    # =====================================================================
    # 内部辅助
    # =====================================================================

    def _ensure_rng(self):
        if self.rng is None:
            self.rng = np.random.RandomState(self.seed)
        return self.rng

    @staticmethod
    def _attach_profiles(table, profiles):
        for field_name, values in profiles.items():
            values = np.asarray(values)
            if values.ndim == 0:
                values = np.full(len(table), values.item())
            table[field_name] = values
        return table
    @staticmethod
    def _profiles_from_table(table, field_names):
        """Extract model-specific, numeric profile columns from a table."""
        return {f: table[f].astype(float).values for f in field_names if f in table.columns}

    def _resample_orders_by_demand_factor(self, orders, demand_factor):
        """Apply a per-order expected-demand multiplier without fractional rows.

        Factors below one probabilistically remove observed requests; factors above
        one duplicate requests and assign synthetic negative IDs to duplicates.
        """
        if len(orders) == 0:
            return orders.copy()
        factor = np.clip(np.asarray(demand_factor, dtype=float), 0.0, self.maximum_demand_multiplier)
        if factor.ndim == 0:
            factor = np.full(len(orders), float(factor))
        if np.all(factor == 1.0):
            return orders.copy().reset_index(drop=True)
        int_count = np.floor(factor).astype(int)
        frac_accept = self._ensure_rng().random(len(factor)) < (factor - int_count)
        repeat = int_count + frac_accept.astype(int)
        positions = np.repeat(np.arange(len(orders)), repeat)
        adjusted = orders.iloc[positions].copy().reset_index(drop=True)
        if len(adjusted) > 0:
            dup_mask = adjusted.duplicated('order_id', keep='first')
            if dup_mask.sum():
                new_ids = np.arange(self.synthetic_order_id_counter,
                                    self.synthetic_order_id_counter - dup_mask.sum(), -1, dtype=int)
                adjusted.loc[dup_mask, 'order_id'] = new_ids
                self.synthetic_order_id_counter -= dup_mask.sum()
        return adjusted

    def _compute_evaluate_df(self, candidate_orders, new_matched):
        """Return per-grid, one-minute flow metrics with explicit denominators.

        ``candidate_orders`` are orders eligible for dispatch at this minute.
        ``new_matched`` are orders retained after driver acceptance and pickup-wait
        cancellation. Ratios are zero when their denominator is zero.
        """
        columns = [
            'origin_grid_id', 'candidate_request_num', 'matched_request_num',
            'gbv', 'platform_revenue', 'baseline_request_num', 'submitted_request_num',
            'request_rate', 'request_price_multiplier', 'action_price_multiplier',
            'matching_radius_km', 'driver_offer_num',
            'driver_accept_num', 'driver_accept_rate', 'unmatched_cancel_num', 'pickup_cancel_num', 'unmatched_cancel_rate',
            'pickup_cancel_rate', 'mean_wait_to_match_seconds',
            'mean_pickup_time_seconds', 'mean_pickup_distance_km',
            'idle_driver_num', 'busy_driver_num', 'repositioning_driver_num',
            'vehicle_occupancy_rate',
        ]
        result = pd.DataFrame(0.0, index=np.arange(self.grid_num), columns=columns)
        result['origin_grid_id'] = np.arange(self.grid_num)
        idx = np.arange(self.grid_num)
        action_prices, action_radii = self._current_grid_action_arrays()
        result['action_price_multiplier'] = action_prices
        result['matching_radius_km'] = action_radii
        if len(candidate_orders):
            g = candidate_orders.groupby('origin_grid_id')
            result['candidate_request_num'] = g.size().reindex(idx, fill_value=0).to_numpy()
        if len(new_matched):
            g = new_matched.groupby('origin_grid_id')
            result['matched_request_num'] = g.size().reindex(idx, fill_value=0).to_numpy()
            result['gbv'] = g['quoted_fare'].sum().reindex(idx, fill_value=0).to_numpy()
            result['platform_revenue'] = g['platform_revenue'].sum().reindex(idx, fill_value=0).to_numpy()
            result['mean_wait_to_match_seconds'] = g['wait_time'].mean().reindex(idx, fill_value=0).to_numpy()
            result['mean_pickup_time_seconds'] = g['pickup_time'].mean().reindex(idx, fill_value=0).to_numpy()
            result['mean_pickup_distance_km'] = g['pickup_distance'].mean().reindex(idx, fill_value=0).to_numpy()
        events = self._step_events
        for event_name, column in [('unmatched_cancel', 'unmatched_cancel_num'),
                                   ('pickup_cancel', 'pickup_cancel_num')]:
            event_orders = events[events['event'] == event_name]
            if len(event_orders):
                counts = event_orders.groupby('origin_grid_id').size().reindex(idx, fill_value=0)
                result[column] = counts.to_numpy()
        driver_offers = events[events['event'].isin(['driver_accept', 'driver_reject'])]
        if len(driver_offers):
            accepted = driver_offers[driver_offers['event'] == 'driver_accept'].groupby('origin_grid_id').size()
            offered = driver_offers.groupby('origin_grid_id').size()
            result['driver_offer_num'] = offered.reindex(idx, fill_value=0).to_numpy()
            result['driver_accept_num'] = accepted.reindex(idx, fill_value=0).to_numpy()
            result['driver_accept_rate'] = (accepted / offered).reindex(idx, fill_value=0).to_numpy()
        baseline_events = events[events['event'] == 'request_baseline']
        submitted_events = events[events['event'] == 'request_submitted']
        baseline = baseline_events.groupby('origin_grid_id').size()
        submitted = submitted_events.groupby('origin_grid_id').size()
        result['baseline_request_num'] = baseline.reindex(idx, fill_value=0).to_numpy()
        result['submitted_request_num'] = submitted.reindex(idx, fill_value=0).to_numpy()
        # No baseline arrival means a demand ratio is undefined, not zero.  The
        # request events originate in the preceding simulation minute, so retain
        # their actually applied order-level price for correct action alignment.
        request_rate = (submitted / baseline).reindex(idx)
        result['request_rate'] = request_rate.to_numpy()
        if 'price_multiplier' in baseline_events:
            request_price = baseline_events.groupby('origin_grid_id')['price_multiplier'].mean()
        else:
            request_price = pd.Series(dtype=float)
        result['request_price_multiplier'] = request_price.reindex(idx).to_numpy()
        pre_cancel_pool = result['candidate_request_num'] + result['unmatched_cancel_num']
        result['unmatched_cancel_rate'] = (
            result['unmatched_cancel_num'] / pre_cancel_pool.replace(0, np.nan)
        ).fillna(0)
        offered_pickup = result['matched_request_num'] + result['pickup_cancel_num']
        result['pickup_cancel_rate'] = (result['pickup_cancel_num'] / offered_pickup.replace(0, np.nan)).fillna(0)
        idle = self.driver_table[self.driver_table['status'] == 0].groupby('grid_id').size()
        busy = self.driver_table[self.driver_table['status'].isin([1, 2])].groupby('grid_id').size()
        repositioning = self.driver_table[self.driver_table['status'] == 4].groupby('grid_id').size()
        result['idle_driver_num'] = idle.reindex(idx, fill_value=0).to_numpy()
        result['busy_driver_num'] = busy.reindex(idx, fill_value=0).to_numpy()
        result['repositioning_driver_num'] = repositioning.reindex(idx, fill_value=0).to_numpy()
        total = result['idle_driver_num'] + result['busy_driver_num']
        result['vehicle_occupancy_rate'] = (result['busy_driver_num'] / total.replace(0, np.nan)).fillna(0)
        return result.reset_index(drop=True)

    # =====================================================================
    # Idle-driver repositioning
    # =====================================================================

    def _initialize_repositioning(self):
        """Load R=2 candidates and initialize lagged driver-income history."""
        self.reposition_income_history = np.zeros(
            (self.reposition_income_window_minutes, self.grid_num))
        self.reposition_income_sum = np.zeros(self.grid_num)
        self.reposition_events = []
        self.total_reposition_distance_km = 0.0
        self.total_reposition_cost = 0.0
        if not self.enable_reposition:
            return
        adjacency = pd.read_csv(Path(self.reposition_adjacency_path)).to_numpy(dtype=bool)
        centroid_table = pd.read_csv(Path(self.reposition_grid_info_path))
        if adjacency.shape != (self.grid_num, self.grid_num):
            raise ValueError("reposition adjacency shape must match grid_num")
        if len(centroid_table) != self.grid_num:
            raise ValueError("reposition grid-info rows must match grid_num")
        self.grid_centroids = centroid_table[['center_lon', 'center_lat']].to_numpy(dtype=float)
        # A boolean square of the adjacency matrix yields graph paths of length <=2.
        # Include the origin grid as an explicit zero-cost ``stay`` alternative.
        reach_two = (adjacency.astype(int) @ adjacency.astype(int)) > 0
        reachable = adjacency | reach_two
        np.fill_diagonal(reachable, True)
        self.reposition_candidates = [np.flatnonzero(reachable[g]) for g in range(self.grid_num)]

    def _update_reposition_income(self, new_matched):
        """Append this minute's confirmed origin-grid driver payments to the rolling window."""
        income = np.zeros(self.grid_num)
        if len(new_matched):
            by_grid = new_matched.groupby('origin_grid_id')['driver_payment'].sum()
            income += by_grid.reindex(range(self.grid_num), fill_value=0).to_numpy(dtype=float)
        slot = self.current_step % self.reposition_income_window_minutes
        self.reposition_income_sum -= self.reposition_income_history[slot]
        self.reposition_income_history[slot] = income
        self.reposition_income_sum += income

    def _reposition_cumulative_probability(self, idle_time_seconds):
        """Return the smoothstep CDF for a continuous idle spell."""
        x = np.clip(np.asarray(idle_time_seconds, dtype=float)
                    / self.reposition_idle_threshold_seconds, 0.0, 1.0)
        return 3.0 * x ** 2 - 2.0 * x ** 3

    def _reposition_trigger_probability(self, idle_time_seconds):
        """Convert the idle-spell CDF to a one-minute conditional hazard."""
        current = self._reposition_cumulative_probability(idle_time_seconds)
        previous = self._reposition_cumulative_probability(
            np.maximum(np.asarray(idle_time_seconds, dtype=float) - self.delta_t, 0.0))
        return np.clip((current - previous) / np.maximum(1.0 - previous, 1e-12), 0.0, 1.0)

    def _start_repositioning(self):
        """Sample an idle-spell dispatch decision, then choose R=2/stay by logit."""
        if not self.enable_reposition:
            return
        idle = self.driver_table.index[self.driver_table['status'] == 0]
        trigger_probability = self._reposition_trigger_probability(
            self.driver_table.loc[idle, 'total_idle_time'].to_numpy())
        eligible = idle[self._ensure_rng().random_sample(len(idle)) < trigger_probability]
        for driver_index in eligible:
            driver = self.driver_table.loc[driver_index]
            trigger_prob = float(self._reposition_trigger_probability(driver['total_idle_time']))
            origin_grid = int(driver['grid_id'])
            candidates = self.reposition_candidates[origin_grid]
            if not len(candidates):
                continue
            destination_coords = self.grid_centroids[candidates]
            distances = np.array([
                self._haversine_distance(driver['lng'], driver['lat'], lng, lat)
                for lng, lat in destination_coords
            ])
            # Staying is an explicit alternative: no travel and no empty-mile cost.
            stay = candidates == origin_grid
            distances[stay] = 0.0
            costs = distances * self.reposition_cost_per_km
            # Income and cost are both 10-minute USD totals, preserving units in
            # the logit utility. The cost is a one-off empty-trip outlay.
            income_total = self.reposition_income_sum[candidates]
            probabilities = self.driver_response_model.driver_reposition_probabilities(
                expected_payments=income_total,
                reposition_costs=costs,
                temperature=self.reposition_logit_temperature,
            )
            chosen_position = self._ensure_rng().choice(len(candidates), p=probabilities)
            destination_grid = int(candidates[chosen_position])
            distance_km = float(distances[chosen_position])
            cost = float(costs[chosen_position])
            no_income_signal = bool(np.allclose(income_total, 0.0))
            is_stay = destination_grid == origin_grid
            self.reposition_events.append({
                'sim_time': self.time,
                'minute_step': self.current_step,
                'driver_id': int(driver['driver_id']),
                'origin_grid_id': origin_grid,
                'destination_grid_id': destination_grid,
                'idle_time_before_seconds': float(driver['total_idle_time']),
                'reposition_trigger_probability': trigger_prob,
                'candidate_grid_count': int(len(candidates)),
                'chosen_income_total_10min': float(income_total[chosen_position]),
                'candidate_income_total_10min_mean': float(income_total.mean()),
                'chosen_probability': float(probabilities[chosen_position]),
                'action': 'stay' if is_stay else 'move',
                'reposition_distance_km': distance_km,
                'reposition_time_seconds': distance_km / self.vehicle_speed * 3600,
                'reposition_operating_cost': cost,
                'choice_reason': 'no_income_signal' if no_income_signal else 'income_logit',
            })
            if is_stay:
                # A stay is an explicit completed dispatch decision. Restart the
                # idle spell so a driver is not trapped after the CDF reaches 1.
                self.driver_table.loc[driver_index, 'total_idle_time'] = 0.0
                continue
            destination_lng, destination_lat = self.grid_centroids[destination_grid]
            self.total_reposition_distance_km += distance_km
            self.total_reposition_cost += cost
            self.driver_table.loc[driver_index, 'status'] = 4
            self.driver_table.loc[driver_index, 'target_grid_id'] = destination_grid
            self.driver_table.loc[driver_index, 'target_loc_lng'] = destination_lng
            self.driver_table.loc[driver_index, 'target_loc_lat'] = destination_lat
            self.driver_table.loc[driver_index, 'remaining_time'] = distance_km / self.vehicle_speed * 3600
            self.driver_table.loc[driver_index, 'total_idle_time'] = 0.0

    @staticmethod
    def _haversine_distance(lng1, lat1, lng2, lat2):
        lat1, lng1, lat2, lng2 = np.radians([lat1, lng1, lat2, lng2])
        a = (np.sin((lat2 - lat1) / 2) ** 2
             + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
        return 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    # =====================================================================
    # 初始化与重置
    # =====================================================================

    def initial_base_tables(self, given_data=False, request_databases=None, driver_info=None):
        if not given_data:
            pattern = SimulatorPattern(self.experiment_date)
            self.request_databases = pattern.request_all
            self.driver_info = pattern.driver_info
        else:
            self.request_databases = request_databases
            self.driver_info = driver_info

        self.time = self.t_initial
        self.current_step = 0
        self.end_of_episode = 0

        # 最好不要再用sample driver命名 这就是初始化司机信息
        # Supply is fixed for an episode: every sampled input driver is online at
        # reset and remains online until the episode ends.
        if self.driver_num > len(self.driver_info):
            raise ValueError("driver_num exceeds the available driver data")
        driver_info = self.driver_info
        if self.driver_num < len(driver_info):
            positions = self._ensure_rng().choice(
                len(driver_info), size=self.driver_num, replace=False)
            driver_info = driver_info.iloc[np.sort(positions)].copy()
        self.driver_table = sample_all_drivers(
            driver_info, self.t_initial, self.t_end,
            self.driver_sample_ratio)
        self.driver_table['status'] = 0
        self.driver_table['start_time'] = self.t_initial
        self.driver_table['end_time'] = self.t_end + self.delta_t
        # Idle duration is episode-local state; source-data history must not make
        # a cohort of drivers reposition immediately after reset.
        self.driver_table['total_idle_time'] = 0.0

        # Drop node-level columns if present in the source data
        _node_cols = ['current_road_node_index', 'remaining_time_for_current_node',
                      'itinerary_node_list', 'itinerary_segment_dis_list']
        self.driver_table = self.driver_table.drop(
            columns=[c for c in _node_cols if c in self.driver_table.columns])

        # Coerce types
        self.driver_table['target_grid_id'] = self.driver_table['target_grid_id'].astype(int)
        self.driver_table['grid_id'] = self.driver_table['grid_id'].astype(int)
        self.driver_table['status'] = self.driver_table['status'].astype(int)
        self.driver_table['remaining_time'] = self.driver_table['remaining_time'].astype(float)
        self.driver_table['total_idle_time'] = self.driver_table['total_idle_time'].astype(float)
        for col in ['target_loc_lng', 'target_loc_lat', 'lng', 'lat']:
            self.driver_table[col] = self.driver_table[col].astype(float)
        self.driver_table['matched_order_id'] = self.driver_table['matched_order_id'].astype(object)

        # Internal column: trip_time stored at match time, used when pickup finishes
        self.driver_table['_pending_trip_time'] = 0.0

        if self.driver_response_model is not None:
            self._attach_profiles(self.driver_table,
                                  self.driver_response_model.create_driver_profiles(
                                      len(self.driver_table), self._ensure_rng()))
            self.driver_table['_price_response_online_draw'] = (
                self._ensure_rng().random_sample(len(self.driver_table)))

        self.wait_requests = pd.DataFrame(columns=self.request_columns)
        self.matched_requests = pd.DataFrame(columns=self.request_columns)
        self._step_events = pd.DataFrame(columns=['event', 'origin_grid_id'])
        self._pending_request_events = pd.DataFrame(columns=['event', 'origin_grid_id'])

        self.total_reward = 0.0
        self.matched_requests_num = 0
        self.potential_request_num = 0
        self.baseline_request_num = 0
        self.accepted_quote_num = 0
        self.passenger_cancelled_requests_num = 0
        self.driver_rejected_requests_num = 0
        self.total_online_time = self.driver_table.shape[0] * (self.t_end - self.t_initial)
        self.cumulative_on_trip_driver_num = 0
        self.occupancy_rate = 0

        evaluate_cols = [
            'origin_grid_id', 'candidate_request_num', 'matched_request_num',
            'gbv', 'platform_revenue', 'baseline_request_num', 'submitted_request_num',
            'request_rate', 'request_price_multiplier', 'action_price_multiplier',
            'matching_radius_km', 'driver_offer_num',
            'driver_accept_num', 'driver_accept_rate', 'unmatched_cancel_num', 'pickup_cancel_num', 'unmatched_cancel_rate',
            'pickup_cancel_rate', 'mean_wait_to_match_seconds',
            'mean_pickup_time_seconds', 'mean_pickup_distance_km',
            'idle_driver_num', 'busy_driver_num', 'repositioning_driver_num',
            'vehicle_occupancy_rate',
        ]
        self.evaluate_df = pd.DataFrame(
            np.zeros((self.grid_num, len(evaluate_cols))), columns=evaluate_cols)
        self.evaluate_table = np.zeros((self.finish_run_step, self.grid_num, len(evaluate_cols)))
        self.total_reward_by_grid = pd.Series(np.zeros(self.grid_num))
        self.reward_by_grid_df = pd.Series(np.zeros(self.grid_num))

        self.temp_total_request_record = pd.DataFrame(columns=self.request_columns)
        self.synthetic_order_id_counter = -1
        self._invalidate_grid_action_cache()
        self._initialize_repositioning()

    def reset(self, seed, given_data=False, request_databases=None, driver_info=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            self.seed = seed
        self.initial_base_tables(given_data, request_databases, driver_info)

    # =====================================================================
    # 订单生命周期
    # =====================================================================

    def _cancel_unmatched_orders(self):
        """在构建二分图前移除超时和主动取消的等待订单。"""
        if not len(self.wait_requests):
            return

        expired = (
            self.wait_requests['wait_time'].astype(float)
            > self.wait_requests['maximum_wait_time'].astype(float))
        cancelled_orders = self.wait_requests.loc[expired].copy()
        self.wait_requests = self.wait_requests.loc[~expired].reset_index(drop=True)

        if (self.unmatched_cancellation
                and self.passenger_response_model is not None
                and len(self.wait_requests)):
            profile = self._profiles_from_table(
                self.wait_requests,
                self.passenger_response_model.passenger_profile_fields)
            probability = (
                self.passenger_response_model
                .passenger_cancel_unmatched_probability(
                    wait_time=self.wait_requests['wait_time'].astype(float).values,
                    profile=profile))
            self.wait_requests['unmatched_cancel_probability'] = probability
            cancelled = self.passenger_response_model.sample(
                probability, self._ensure_rng())
            cancelled_orders = pd.concat([
                cancelled_orders,
                self.wait_requests.loc[cancelled],
            ], ignore_index=True)
            self.wait_requests = self.wait_requests.loc[
                ~cancelled].reset_index(drop=True)

        if not len(cancelled_orders):
            return
        cancelled_orders['event'] = 'unmatched_cancel'
        self._step_events = pd.concat(
            [self._step_events, cancelled_orders], ignore_index=True)
        self.passenger_cancelled_requests_num += len(cancelled_orders)

    # =====================================================================
    # 匹配后更新
    # =====================================================================

    def update_info_after_matching_multi_process(self, matched_pair_actual_indexes, matched_itinerary):
        """更新匹配后的司机和订单状态。

        matched_itinerary: np.ndarray of pickup distances (one per matched pair),
                           as returned by the grid-level order_dispatch.
        """
        # ``matched_itinerary`` is a legacy name. In the grid-level simulator it
        # contains only one pickup distance (km) for each returned match, not a route.
        new_matched = pd.DataFrame(columns=self.request_columns)
        update_wait = pd.DataFrame(columns=self.request_columns)

        pair_df = pd.DataFrame(matched_pair_actual_indexes,
                               columns=['order_id', 'driver_id', 'weight', 'pickup_distance'])

        # Override pickup_distance with the precise values from the dispatch return
        if len(matched_itinerary) > 0:
            pair_df['pickup_distance'] = np.asarray(matched_itinerary, dtype=float)

        matched_order_ids = pair_df['order_id'].tolist()
        con_matched = self.wait_requests['order_id'].isin(matched_order_ids)

        df_matched = self.wait_requests[con_matched].reset_index(drop=True)
        if df_matched.shape[0] == 0:
            update_wait = self.wait_requests[~con_matched]
            return new_matched, update_wait
        idle_driver_table = self.driver_table[self.driver_table['status'] == 0]
        order_array = df_matched['order_id'].values
        # The dispatch output order differs from ``df_matched`` row order.
        # ``order_positions`` aligns orders positionally; ``driver_indices`` stores
        # label indices for .loc updates to the main driver table.
        order_positions, driver_indices = [], []
        for _, pair in pair_df.iterrows():
            order_positions.append(np.flatnonzero(order_array == pair['order_id'])[0])
            driver_indices.append(
                idle_driver_table.index[idle_driver_table['driver_id'] == pair['driver_id']][0])
        driver_indices = np.asarray(driver_indices)
        df_matched = df_matched.iloc[order_positions, :]

        # Pickup distance and time from pair_df (already aligned with df_matched via cor_order)
        pickup_dis = np.asarray(pair_df['pickup_distance'].values, dtype=float)
        pickup_t = pickup_dis / self.vehicle_speed * 3600

        # 司机接单概率
        if self.driver_response_model is not None:
            drv_table = self.driver_table.loc[driver_indices]
            drv_profile = self._profiles_from_table(drv_table, self.driver_response_model.driver_profile_fields)
            ref_pay = (df_matched['base_fare'].astype(float).values
                       * (1.0 - df_matched['commission_rate'].astype(float).values))   # 参考payment
            # 这里有点抽象 我需要知道输入参数的具体计算公式
            drv_accept_prob = self.driver_response_model.driver_accept_probability(
                offer=DriverOffer(
                    driver_payment=df_matched['driver_payment'].astype(float).values,
                    reference_payment=ref_pay,
                    pickup_distance=pickup_dis,
                    trip_distance=df_matched['trip_distance'].astype(float).values),
                profile=drv_profile,
            )
            df_matched['driver_accept_probability'] = drv_accept_prob
            con_driver_remain = self.driver_response_model.sample(drv_accept_prob, self._ensure_rng())
        else:
            df_matched['driver_accept_probability'] = 1.0
            con_driver_remain = np.ones(len(df_matched), dtype=bool)

        driver_rejected = ~con_driver_remain
        driver_accepted = ~driver_rejected
        # Stage 4 only occurs after a driver accepts. A rejection returns the
        # passenger to the queue; it must not be turned into a pickup cancellation.
        pickup_cancel_probability = np.zeros(len(df_matched), dtype=float)
        passenger_cancelled = np.zeros(len(df_matched), dtype=bool)
        if (self.passenger_response_model is not None and self.pickup_cancellation
                and driver_accepted.any()):
            passenger_profile = self._profiles_from_table(
                df_matched.loc[driver_accepted],
                self.passenger_response_model.passenger_profile_fields)
            probability = self.passenger_response_model.passenger_cancel_pickup_probability(
                pickup_time=pickup_t[driver_accepted], pickup_distance=pickup_dis[driver_accepted],
                profile=passenger_profile)
            pickup_cancel_probability[driver_accepted] = probability
            passenger_cancelled[driver_accepted] = self.passenger_response_model.sample(
                probability, self._ensure_rng())
        df_matched['pickup_cancel_probability'] = pickup_cancel_probability

        con_remain = driver_accepted & ~passenger_cancelled
        update_wait = df_matched[driver_rejected]
        self.driver_rejected_requests_num += int(driver_rejected.sum())
        self.passenger_cancelled_requests_num += int(passenger_cancelled.sum())
        event_rows = []
        if driver_rejected.any():
            event_rows.append(df_matched.loc[driver_rejected].assign(event='driver_reject'))
        if driver_accepted.any():
            event_rows.append(df_matched.loc[driver_accepted].assign(event='driver_accept'))
        if passenger_cancelled.any():
            event_rows.append(df_matched.loc[passenger_cancelled].assign(event='pickup_cancel'))
        if event_rows:
            self._step_events = pd.concat([self._step_events, *event_rows], ignore_index=True)

        # A rejected offer or pickup-stage cancellation does not make the driver
        # work. Keep the driver idle and preserve total_idle_time; only a retained
        # match resets it below when pickup actually begins.

        new_matched = df_matched[con_remain].copy()
        new_matched['t_matched'] = self.time
        new_matched['pickup_distance'] = pickup_dis[con_remain]
        new_matched['pickup_time'] = new_matched['pickup_distance'].values / self.vehicle_speed * 3600
        new_matched['t_end'] = (self.time
                                + new_matched['pickup_time'].values
                                + new_matched['trip_time'].values)
        new_matched['status'] = 1
        new_matched['driver_id'] = pair_df[con_remain]['driver_id'].values

        accepted_driver_indices = driver_indices[con_remain]
        self.driver_table.loc[accepted_driver_indices, 'status'] = 2
        self.driver_table.loc[accepted_driver_indices, 'target_loc_lng'] = new_matched['dest_lng'].astype(float).values
        self.driver_table.loc[accepted_driver_indices, 'target_loc_lat'] = new_matched['dest_lat'].astype(float).values
        self.driver_table.loc[accepted_driver_indices, 'target_grid_id'] = new_matched['dest_grid_id'].astype(int).values
        self.driver_table.loc[accepted_driver_indices, 'remaining_time'] = new_matched['pickup_time'].astype(float).values
        self.driver_table.loc[accepted_driver_indices, 'matched_order_id'] = new_matched['order_id'].values.astype(object)
        self.driver_table.loc[accepted_driver_indices, 'total_idle_time'] = 0
        self.driver_table.loc[accepted_driver_indices, 'time_to_last_cruising'] = 0
        # Store trip_time so update_state can apply it when pickup phase finishes.
        self.driver_table.loc[accepted_driver_indices, '_pending_trip_time'] = new_matched['trip_time'].astype(float).values

        update_wait = pd.concat(
            [update_wait, self.wait_requests[~con_matched]], axis=0)

        return new_matched, update_wait

    # =====================================================================
    # 订单生成
    # =====================================================================

    def step_bootstrap_new_orders(self):
        min_time = max(self.t_initial, self.time - self.request_interval) # request interval就是仿真单步时间
        temp = [req for t in range(min_time, self.time) for req in self.request_databases.get(t, [])]
        if not temp:
            return

        n_sample = int(np.rint(self.order_sample_ratio * len(temp)))
        sampled = (temp if n_sample >= len(temp)
                   else [temp[i] for i in self.rng.choice(len(temp), n_sample, replace=False)])

        # Read raw record columns; itinerary fields at positions 11-12 are dropped immediately
        col_names = [
            'order_id', 'origin_id', 'origin_lat', 'origin_lng', 'dest_id', 'dest_lat', 'dest_lng',
            'trip_distance', 'start_time', 'origin_grid_id', 'dest_grid_id',
            'itinerary_node_list', 'itinerary_segment_dis_list',
            'trip_time', 'designed_reward', 'cancel_prob',
        ]
        wait_info = pd.DataFrame(sampled, columns=col_names)
        wait_info = wait_info.drop(columns=['itinerary_node_list', 'itinerary_segment_dis_list'])

        wait_info['start_time'] = self.time
        wait_info['trip_distance'] = wait_info['trip_distance'].astype(float)
        wait_info['trip_time'] = wait_info['trip_distance'] / self.vehicle_speed * 3600
        # ``designed_reward`` is retained for compatibility and now has exactly
        # the same no-surge meaning as ``base_fare`` below: USD 3 flag fall plus
        # USD 0.70 per kilometre. Pricing reward uses its surged counterpart.
        wait_info['designed_reward'] = (
            self.base_fare_start + self.price_per_km * wait_info['trip_distance'])

        # Grid ID mapping: for 263-grid runs, merge from mapping_dict[experiment_date]
        # which provides correct origin_grid_id / dest_grid_id keyed by order_id.
        # For other grid sizes the columns are already correct from the raw data.
        if self.mapping_dict is not None and self.grid_num == 263:
            wait_info = wait_info.drop(columns=['origin_grid_id', 'dest_grid_id'], errors='ignore')
            wait_info['order_id'] = wait_info['order_id'].astype(float)
            wait_info = pd.merge(
                wait_info, self.mapping_dict[self.experiment_date], how='left', on='order_id')

        # ``base_fare`` is the no-surge gross fare; ``quoted_fare`` is the actual
        # passenger quote. Price is looked up from the cached action of the
        # order's origin grid.
        wait_info['base_fare'] = (
            self.base_fare_start + wait_info['trip_distance'].astype(float) * self.price_per_km)
        wait_info['price_multiplier'] = self._prices_for_orders(wait_info)
        wait_info['quoted_fare'] = wait_info['base_fare'] * wait_info['price_multiplier']
        wait_info['commission_rate'] = self.commission_rate
        wait_info['driver_payment'] = wait_info['quoted_fare'] * (1.0 - self.commission_rate)
        wait_info['platform_revenue'] = wait_info['quoted_fare'] * self.commission_rate
        # Preserve the applied order-level action on baseline events. These events
        # are consumed next minute when request-flow metrics are evaluated.
        baseline_events = wait_info.copy()
        baseline_events['event'] = 'request_baseline'
        self._pending_request_events = pd.concat(
            [self._pending_request_events, baseline_events], ignore_index=True)
        self.baseline_request_num += len(wait_info)
        self.potential_request_num += len(wait_info)

        # 乘客叫车概率
        if self.passenger_response_model is not None:
            rng = self._ensure_rng()
            self._attach_profiles(
                wait_info,
                self.passenger_response_model.create_passenger_profiles(
                    len(wait_info), wait_info['base_fare'].astype(float).values, rng))
            psg_profile = self._profiles_from_table(
                wait_info, self.passenger_response_model.passenger_profile_fields)
            expected_wait_by_grid = self.evaluate_df[
                'mean_wait_to_match_seconds'].to_numpy(dtype=float)
            expected_wait_time = self._grid_values_for_rows(
                wait_info, expected_wait_by_grid)
            current_offer = PassengerOffer(
                quoted_fare=wait_info['quoted_fare'].astype(float).values,
                base_fare=wait_info['base_fare'].astype(float).values,
                expected_wait_time=expected_wait_time,
                trip_time=wait_info['trip_time'].astype(float).values,
                maximum_wait_time=self.maximum_wait_time_mean,
            )
            request_probability = self.passenger_response_model.passenger_accept_probability(
                offer=current_offer, profile=psg_profile)
            wait_info['passenger_request_probability'] = request_probability
            # Observed requests are already demand realised under a reference price,
            # so rescale by P(current quote) / P(reference quote). Synthetic input
            # instead represents a potential-demand pool and applies P(current quote).
            if self.demand_input_mode == 'observed_requests':
                reference_offer = PassengerOffer(
                    quoted_fare=(wait_info['base_fare'].astype(float).values
                                 * self.demand_reference_price_multiplier),
                    base_fare=wait_info['base_fare'].astype(float).values,
                    expected_wait_time=expected_wait_time,
                    trip_time=wait_info['trip_time'].astype(float).values,
                    maximum_wait_time=self.maximum_wait_time_mean,
                )
                reference_probability = self.passenger_response_model.passenger_accept_probability(
                    offer=reference_offer, profile=psg_profile)
                demand_factor = (np.asarray(request_probability, dtype=float)
                                 / np.maximum(np.asarray(reference_probability, dtype=float), 1e-9))
            else:
                reference_probability = np.ones(len(wait_info))
                demand_factor = np.asarray(request_probability, dtype=float)

            wait_info['baseline_passenger_request_probability'] = reference_probability
            wait_info['demand_adjustment_factor'] = demand_factor
            wait_info = self._resample_orders_by_demand_factor(wait_info, demand_factor)
            self.accepted_quote_num += len(wait_info)
        else:
            self.accepted_quote_num += len(wait_info)
            wait_info['passenger_request_probability'] = 1.0
            wait_info['baseline_passenger_request_probability'] = 1.0
            wait_info['demand_adjustment_factor'] = 1.0

        wait_info['weight'] = np.ones(len(wait_info))
        wait_info['wait_time'] = 0
        wait_info['status'] = 0
        wait_info['maximum_wait_time'] = self.maximum_wait_time_mean

        if self.passenger_response_model is None:
            wait_info['maximum_price_passenger_can_tolerate'] = np.random.normal(
                float('inf'), 0, len(wait_info))
            wait_info = wait_info[
                wait_info['maximum_price_passenger_can_tolerate'] >= wait_info['quoted_fare']]
        wait_info['maximum_pickup_time_passenger_can_tolerate'] = np.random.normal(
            float('inf'), 0, len(wait_info))

        self.temp_total_request_record = pd.concat(
            [self.temp_total_request_record, wait_info], ignore_index=True)
        submitted_events = wait_info.copy()
        submitted_events['event'] = 'request_submitted'
        self._pending_request_events = pd.concat(
            [self._pending_request_events, submitted_events], ignore_index=True)
        self.wait_requests = pd.concat([self.wait_requests, wait_info], ignore_index=True)

    # =====================================================================
    # 状态更新 — 网格级别
    # =====================================================================

    def update_state(self):
        dt = self.driver_table
        dt['remaining_time'] -= self.delta_t
        dt.loc[dt['status'] == 0, 'total_idle_time'] += self.delta_t

        # Finished pickup → enter delivery
        finished_pickup = (dt['status'] == 2) & (dt['remaining_time'] <= 0)
        if finished_pickup.any():
            dt.loc[finished_pickup, 'status'] = 1
            dt.loc[finished_pickup, 'remaining_time'] = (
                dt.loc[finished_pickup, '_pending_trip_time'].values)

        # Finished delivery → idle at destination grid
        finished_delivery = (dt['status'] == 1) & (dt['remaining_time'] <= 0)
        if finished_delivery.any():
            dt.loc[finished_delivery, ['status', 'total_idle_time']] = 0
            dt.loc[finished_delivery, 'lng'] = dt.loc[finished_delivery, 'target_loc_lng'].values
            dt.loc[finished_delivery, 'lat'] = dt.loc[finished_delivery, 'target_loc_lat'].values
            dt.loc[finished_delivery, 'grid_id'] = (
                dt.loc[finished_delivery, 'target_grid_id'].values)
            dt.loc[finished_delivery, 'matched_order_id'] = 'None'

        # Finished empty reposition → idle at the selected grid centroid.  Reset
        # idle time so the driver must remain there another full threshold before
        # becoming eligible for a subsequent reposition.
        finished_reposition = (dt['status'] == 4) & (dt['remaining_time'] <= 0)
        if finished_reposition.any():
            dt.loc[finished_reposition, ['status', 'total_idle_time']] = 0
            dt.loc[finished_reposition, 'lng'] = dt.loc[finished_reposition, 'target_loc_lng'].values
            dt.loc[finished_reposition, 'lat'] = dt.loc[finished_reposition, 'target_loc_lat'].values
            dt.loc[finished_reposition, 'grid_id'] = (
                dt.loc[finished_reposition, 'target_grid_id'].values)
            dt.loc[finished_reposition, 'matched_order_id'] = 'None'

        self.wait_requests['wait_time'] += self.delta_t

    # =====================================================================
    # 司机上下线
    # =====================================================================

    def driver_online_offline_update(self):
        """No-op: the fixed sampled supply remains online for this episode."""
        return None

    # =====================================================================
    # 时间推进
    # =====================================================================

    def update_time(self):
        self.time += self.delta_t
        self.current_step = int((self.time - self.t_initial) // self.delta_t)
        if self.current_step >= self.finish_run_step:
            self.end_of_episode = 1

    # =====================================================================
    # 主循环 — 单步执行
    # =====================================================================

    def step(self):
        """Execute one 60-second transition through the four behavioural stages."""
        # Resolve the complete per-grid action once at the decision boundary.
        # All pricing, dispatch and metric code below reads this cached table.
        self._current_grid_action_arrays()
        self._step_events = self._pending_request_events.copy()
        self._pending_request_events = pd.DataFrame(columns=['event', 'origin_grid_id'])
        # Stage 3 precedes dispatch: queued passengers may abandon before an offer.
        self._cancel_unmatched_orders()
        wait_requests = deepcopy(self.wait_requests)
        driver_table = deepcopy(self.driver_table)

        # Match remaining queued passengers to currently idle drivers.
        _, matching_radius_by_grid = self._current_grid_action_arrays()
        matched_pairs, matched_itinerary = order_dispatch(
            wait_requests, driver_table, matching_radius_by_grid,
            self.dispatch_method, self.dispatch_scoring_mode)
        df_new_matched, df_update_wait = self.update_info_after_matching_multi_process(
            matched_pairs, matched_itinerary)

        # 2. Record gross booking value. Unlike the legacy ``designed_reward``,
        # quoted_fare contains the applied price multiplier and is a valid pricing
        # objective. Platform revenue is quoted_fare * commission_rate.
        self.total_reward += (
            np.sum(df_new_matched['quoted_fare'].values) if len(df_new_matched) > 0 else 0)
        self.matched_requests_num += len(df_new_matched)

        if len(df_new_matched) > 0:
            matched_by_grid = df_new_matched.groupby('origin_grid_id')['quoted_fare'].sum()
            self.total_reward_by_grid += matched_by_grid.reindex(range(self.grid_num), fill_value=0)
            self.reward_by_grid_df += matched_by_grid.reindex(range(self.grid_num), fill_value=0)

        self.cumulative_on_trip_driver_num += (self.driver_table['status'].isin([1, 2])).sum()
        self.occupancy_rate = (self.cumulative_on_trip_driver_num
                               / ((1 + self.current_step) * self.driver_table.shape[0]))

        self.evaluate_df = self._compute_evaluate_df(wait_requests, df_new_matched)
        self.evaluate_table[self.current_step] = self.evaluate_df.values

        # 3. 更新队列
        if self.end_of_episode == 0:
            self.matched_requests = pd.concat(
                [self.matched_requests, df_new_matched]).reset_index(drop=True)
            self.wait_requests = df_update_wait.reset_index(drop=True)
            self.step_bootstrap_new_orders()

        # 4. State transition and idle-driver repositioning. Destination choice
        # sees only the rolling income window through the preceding minute.
        self.update_state()
        self._start_repositioning()
        self._update_reposition_income(df_new_matched)
        self.driver_online_offline_update()
        self.update_time()
