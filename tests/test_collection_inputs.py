"""轻量订单缓存、内容校验及跨轮复用的一致性测试。"""

from contextlib import redirect_stdout
import io
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from data_pipeline.collection_inputs import (
    compact_orders, prepare_worker_inputs, load_worker_pattern, file_sha256,
)
from data_pipeline.collect_trajectories import run_trajectory_collection


def fixture_orders():
    return {0: [], 86399: [], **{
        timestamp: [[minute * 20 + i, 1, 40.75, -73.99, 2, 40.76, -73.98,
                     2.0, timestamp, 0, 0, [1, 2, 3], [0.2, 0.4], 120, 5.0, 0.0]
                    for i in range(20)]
        for minute, timestamp in enumerate(range(18000, 18300, 60))}}


def fixture_drivers():
    return pd.DataFrame({'driver_id': [1, 2], 'start_time': [18000, 18000],
                         'end_time': [36000, 36000], 'lng': [-73.99, -73.99],
                         'lat': [40.75, 40.75], 'grid_id': [0, 0]})


class CollectionInputTests(unittest.TestCase):
    def test_compaction_preserves_order_and_fields_without_mutating_source(self):
        orders = fixture_orders()
        before = pickle.dumps(orders)
        compact = compact_orders(orders, 18000, 18120)
        self.assertEqual(sorted(compact), [18000, 18060, 18119])
        for t in [18000, 18060]:
            for original, row in zip(orders[t], compact[t]):
                self.assertEqual(row[:11] + row[13:], original[:11] + original[13:])
                self.assertEqual(row[11:13], [None, None])
        self.assertEqual(pickle.dumps(orders), before)
        for start, end in [(-1, 10), (0, 86401), (3, 3)]:
            with self.assertRaises(ValueError):
                compact_orders(orders, start, end)

    def test_cache_is_reused_and_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            orders = root / 'orders_grid35_2015-05-18.pkl'
            drivers = root / 'drivers_grid35_1000.pickle'
            orders.write_bytes(pickle.dumps(fixture_orders()))
            drivers.write_bytes(pickle.dumps(fixture_drivers()))
            config = {'data_dir': directory, 'date': '2015-05-18', 'start_time': '05:00',
                      'simulation_steps': 3, 'source_sha256': {
                          str(path.resolve()): file_sha256(path) for path in [orders, drivers]}}
            first = prepare_worker_inputs(root, config)
            payload = Path(first['orders_path'])
            before = payload.stat().st_mtime_ns
            with patch('data_pipeline.collection_inputs.pickle.load', side_effect=AssertionError):
                self.assertEqual(prepare_worker_inputs(root, config), first)
            self.assertEqual(payload.stat().st_mtime_ns, before)
            pattern = load_worker_pattern(first)
            pd.testing.assert_frame_equal(pattern.driver_info, fixture_drivers())
            self.assertEqual(max(pattern.request_all), 18179)
            payload.write_bytes(b'broken')
            with self.assertRaises(ValueError):
                prepare_worker_inputs(root, config)
            with self.assertRaises(ValueError):
                load_worker_pattern(first)

    @patch('data_pipeline.collect_trajectories.check_collection_inputs')
    @patch('data_pipeline.collect_trajectories.SimulatorPattern')
    def test_same_seed_matches_full_input_and_reuse_does_not_leak_state(self, loader, check):
        full = SimpleNamespace(request_all=fixture_orders(), driver_info=fixture_drivers())
        loader.return_value = full
        compact = SimpleNamespace(request_all=compact_orders(full.request_all, 18000, 18300),
                                  driver_info=full.driver_info.copy(deep=True))
        before = pickle.dumps(compact)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            options = dict(date='2015-05-18', grid_num=1, driver_num=2, simulation_steps=5,
                           decision_freq=1, price_action_values=[1.0, 1.6],
                           matching_radius_values=[1.0, 2.0], data_dir=directory,
                           order_sample_ratio=0.2)
            reference = run_trajectory_collection(**options, seed=42,
                                                  output_dir=Path(directory) / 'full')
            for index, seed in enumerate([42, 43, 42]):
                result = run_trajectory_collection(**options, seed=seed, _pattern=compact,
                                                   output_dir=Path(directory) / str(index))
                if seed == 42:
                    for name in ['action_plan', 'minute_metrics']:
                        pd.testing.assert_frame_equal(reference[name], result[name], check_exact=True)
                    self.assertEqual(reference['manifest']['summary'], result['manifest']['summary'])
        self.assertEqual(pickle.dumps(compact), before)
        self.assertEqual(loader.call_count, 1)


if __name__ == '__main__':
    unittest.main()
