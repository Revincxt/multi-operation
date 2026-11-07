"""密度动作扫描的配对公平性、筛选隔离和原始指标校验。"""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from diagnostics.density_action_sweep import (
    METRICS, REFERENCE, common_inputs, paired_comparisons, rank_actions,
    run_action, run_sweep, select_shortlist, plot_results,
)
from baseline.compare_policies import summarize_operation


def example_runs(phase="screening", seeds=(501,)):
    rows = []
    for scenario in ("low", "medium", "high"):
        for seed in seeds:
            for price, radius, reward in ((1.0, 2.0, 10), (1.5, 3.0, 12), (1.6, 4.0, 14)):
                rows.append({"phase": phase, "scenario": scenario, "seed": seed,
                             "price_multiplier": price, "matching_radius_km": radius,
                             "sampled_orders_sha256": f"orders-{scenario}-{seed}",
                             "initial_drivers_sha256": f"drivers-{scenario}-{seed}",
                             "generated_baseline_requests": 100, "baseline_request_num": 90,
                             **dict.fromkeys(METRICS, float(reward))})
    return pd.DataFrame(rows)


class DensityActionSweepTests(unittest.TestCase):
    def test_shortlist_uses_screening_only_and_always_includes_reference(self):
        frame = example_runs()
        selected = select_shortlist(frame, 1)
        self.assertTrue(all(pairs == [(1.6, 4.0), REFERENCE] for pairs in selected.values()))
        with self.assertRaises(ValueError):
            select_shortlist(example_runs("confirmation"), 1)
        with self.assertRaises(ValueError):
            rank_actions(pd.concat([frame, example_runs("confirmation", (502,))]))

    def test_rejects_mismatched_inputs_seeds_or_duplicate_results(self):
        frame = example_runs(seeds=(501, 502))
        self.assertTrue(rank_actions(frame).seed_count.eq(2).all())
        for column in ("sampled_orders_sha256", "initial_drivers_sha256"):
            corrupt = frame.copy()
            corrupt.loc[0, column] = "different"
            with self.assertRaises(ValueError):
                rank_actions(corrupt)
        with self.assertRaises(ValueError):
            rank_actions(frame.iloc[1:])
        with self.assertRaises(ValueError):
            rank_actions(pd.concat([frame, frame.iloc[:1]]))

    def test_paired_revenue_differences_and_independent_seed_validation(self):
        frame = example_runs("confirmation", (601, 602, 603))
        results = paired_comparisons(frame)
        best = results[results.price_multiplier == 1.6]
        self.assertTrue(best.mean_revenue_difference.eq(2.0).all())
        self.assertTrue(best.bootstrap_95_low.eq(2.0).all())
        self.assertTrue(best.bootstrap_95_high.eq(2.0).all())
        self.assertTrue(best.seeds_better_than_reference.eq(3).all())
        self.assertTrue(results[results.price_multiplier == 1.5].mean_revenue_difference.eq(0).all())
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with self.assertRaises(ValueError):
                run_sweep(output, screening_seeds=[42], confirmation_seeds=[42])
            self.assertFalse(output.exists())

    def test_plots_separate_screening_from_confirmation_with_directions(self):
        try:
            import matplotlib.figure
        except ModuleNotFoundError:
            self.skipTest("matplotlib not installed")
        figures = []

        def capture(figure, *args, **kwargs):
            figures.append([axis.get_title() + axis.get_ylabel() for axis in figure.axes])

        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(matplotlib.figure.Figure, 'savefig', capture):
            root = Path(temporary)
            rank_actions(example_runs()).to_csv(root / 'screening_rankings.csv', index=False)
            rank_actions(example_runs('confirmation', (601, 602))).to_csv(
                root / 'confirmation_rankings.csv', index=False)
            (root / 'manifest.json').write_text(json.dumps({'config': {
                'scenarios': ['low', 'medium', 'high'], 'confirmation_seeds': [601, 602]}}))
            plot_results(root)
            self.assertEqual(len(figures), 2)
            self.assertTrue(any('↑' in text for text in figures[0]))
            self.assertTrue(any('↓' in text for text in figures[1]))
            self.assertEqual(len(figures[1]), 12)

    def test_simulator_records_full_window_and_preserves_paired_inputs(self):
        drivers = pd.DataFrame({"driver_id": [1, 2], "start_time": [18000, 18000],
                                "end_time": [36000, 36000], "lng": [-73.99, -73.98],
                                "lat": [40.75, 40.75], "grid_id": [0, 0]})
        requests = {18000 + minute * 60: [
            [minute * 10 + j, 0, 40.75, -73.99 + j * 0.007, 0, 40.75,
             -73.98, 1.0, 18000 + minute * 60, 0, 0, [], [], 160, 3.7, 0]
            for j in range(4)] for minute in range(10)}
        pattern = SimpleNamespace(request_all=requests, driver_info=drivers)
        common_inputs.cache_clear()
        try:
            with tempfile.TemporaryDirectory() as temporary, \
                    patch('diagnostics.density_action_sweep.load_pattern', return_value=pattern), \
                    patch.dict('diagnostics.density_action_sweep.SCENARIOS',
                               {"low": {"driver_num": 2, "order_sample_ratio": 0.5}}):
                config = {"output": temporary, "date": "2015-05-18", "start_time": "05:00",
                          "simulation_steps": 10, "data_dir": "unused"}
                first = run_action(config, "screening", "low", 1.5, 3.0, 42)
                second = run_action(config, "screening", "low", 1.3, 2.0, 42)
                self.assertEqual(first['sampled_orders_sha256'], second['sampled_orders_sha256'])
                self.assertEqual(first['initial_drivers_sha256'], second['initial_drivers_sha256'])
                self.assertEqual(first['generated_baseline_requests'], 18)
                path = Path(temporary) / "runs/screening/low/p1.5_r3_seed42"
                minutes = pd.read_csv(path / 'grid_minute_metrics.csv.gz')
                self.assertEqual(len(minutes), 60)
                self.assertTrue(minutes.action_price_multiplier.eq(1.5).all())
                self.assertTrue(minutes.matching_radius_km.eq(3.0).all())
                direct = summarize_operation(minutes)
                for metric in METRICS:
                    np.testing.assert_allclose(first[metric], direct[metric])
                self.assertEqual(json.loads((path / 'summary.json').read_text())['phase'], 'screening')
        finally:
            common_inputs.cache_clear()


if __name__ == '__main__':
    unittest.main()
