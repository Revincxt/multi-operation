"""低密度采样、压缩文件、断点续采和失败保护测试。"""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from data_pipeline.build_supervised_dataset import build_supervised_dataset
from data_pipeline.collect_trajectories import run_trajectory_collection
from data_pipeline.collect_density import (
    completed_manifest, planned_splits, run_density_collection, run_path,
    save_json, sha256, config_signature, validate_execution_revision, initialize_worker,
)


class DensitySamplingTests(unittest.TestCase):
    def test_invalid_ratios_fail_before_input_loading(self):
        for ratio in (0, -1, 1.1, float('nan'), float('inf'), True):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                run_trajectory_collection('2015-05-18', order_sample_ratio=ratio)

    @patch('data_pipeline.collect_trajectories.check_collection_inputs')
    @patch('data_pipeline.collect_trajectories.SimulatorPattern')
    def test_sample_counts_compression_and_builder(self, pattern, check):
        orders = {0: [], 86399: []}
        for minute in range(2):
            timestamp = 18000 + minute * 60
            orders[timestamp] = [
                [minute * 20 + i, 1, 40.75, -73.99, 2, 40.76, -73.98,
                 2.0, timestamp, 0, 0, [], [], 120, 5.0, 0.0] for i in range(20)]
        pattern.return_value.request_all = orders
        pattern.return_value.driver_info = pd.DataFrame({
            'driver_id': [1, 2], 'start_time': [18000, 18000], 'end_time': [36000, 36000],
            'lng': [-73.99, -73.99], 'lat': [40.75, 40.75], 'grid_id': [0, 0]})
        with tempfile.TemporaryDirectory() as directory:
            result = run_trajectory_collection(
                '2015-05-18', grid_num=1, driver_num=1, simulation_steps=3,
                decision_freq=1, price_action_values=[1.0], matching_radius_values=[1.0],
                data_dir='test_data', output_dir=Path(directory) / 'raw',
                order_sample_ratio=0.1, compress=True)
            manifest = result['manifest']
            self.assertEqual(manifest['summary']['baseline_requests'], 4)
            self.assertEqual(manifest['driver_num'], 1)
            self.assertEqual(manifest['order_sample_ratio'], 0.1)
            self.assertTrue(manifest['files']['minute_metrics'].endswith('.csv.gz'))
            dataset, _ = build_supervised_dataset(
                Path(directory) / 'raw', Path(directory) / 'dataset', history_steps=1)
            self.assertEqual(len(dataset), 2)


def fixture_config(runs=5):
    return {
        'scenario': 'low', 'run_count': runs, 'seed_start': 42, 'batch_size': 2,
        'date': '2015-05-18', 'start_time': '05:00', 'simulation_steps': 300,
        'grid_num': 6, 'decision_freq': 5, 'driver_num': 250,
        'order_sample_ratio': 0.077, 'price_response_model': 'utility_choice',
        'split_seed': 42, 'validation_ratio': 0.15, 'test_ratio': 0.15,
        'expected_sampled_baseline_requests': 4999,
    }


def fake_collection(root, config, signature, index, split):
    destination = run_path(root, config, index)
    destination.mkdir(parents=True)
    file = destination / 'data.csv.gz'
    pd.DataFrame({'grid_id': range(6)}).to_csv(file, index=False)
    manifest = {
        'simulator_seed': config['seed_start'] + index, 'planned_split': str(split),
        'elapsed_seconds': 0.2, 'collection_config_sha256': signature,
        'file_sha256': {file.name: sha256(file)},
        'summary': {'baseline_requests': config['expected_sampled_baseline_requests']},
    }
    save_json(destination / 'manifest.json', manifest)
    return manifest


class DensityCollectionTests(unittest.TestCase):
    def setUp(self):
        # 本组只测调度；真实窗口输入另有加载与一致性测试。
        patcher = patch('data_pipeline.collect_density.prepare_worker_inputs', return_value=None)
        self.prepare_inputs = patcher.start()
        self.addCleanup(patcher.stop)

    def test_planned_splits_are_fixed_and_run_level(self):
        config = fixture_config(100000)
        first = planned_splits(config)
        second = planned_splits(config)
        self.assertTrue((first == second).all())
        self.assertEqual(pd.Series(first).value_counts().to_dict(),
                         {'train': 70000, 'validation': 15000, 'test': 15000})

    @patch('data_pipeline.collect_density.ProcessPoolExecutor', ThreadPoolExecutor)
    @patch('data_pipeline.collect_density.collect_one', side_effect=fake_collection)
    @patch('data_pipeline.collect_density.make_config')
    def test_resume_skips_verified_runs_and_rejects_changed_config(self, configure, collect):
        config = fixture_config()
        configure.return_value = config
        with tempfile.TemporaryDirectory() as directory:
            first = run_density_collection(output=directory, runs=5, workers=1, max_new_runs=2)
            self.assertEqual(first['status'], 'batch_limit')
            self.assertEqual(first['completed_runs'], 2)
            second = run_density_collection(output=directory, runs=5, workers=1)
            self.assertEqual(second['status'], 'complete')
            self.assertEqual(second['completed_runs'], 5)
            self.assertEqual(collect.call_count, 5)
            self.assertEqual(len(list(Path(directory).glob('batch_*'))), 3)
            self.assertEqual(len(list(Path(directory).glob('batch_*/*/manifest.json'))), 5)
            config['driver_num'] = 500
            with self.assertRaises(ValueError):
                run_density_collection(output=directory, runs=5, workers=1)

    @patch('data_pipeline.collect_density.ProcessPoolExecutor', ThreadPoolExecutor)
    @patch('data_pipeline.collect_density.collect_one', side_effect=fake_collection)
    @patch('data_pipeline.collect_density.make_config')
    def test_corrupt_result_is_not_silently_skipped(self, configure, collect):
        config = fixture_config()
        configure.return_value = config
        with tempfile.TemporaryDirectory() as directory:
            run_density_collection(output=directory, runs=5, workers=1, max_new_runs=1)
            destination = run_path(directory, config, 0)
            manifest = json.loads((destination / 'manifest.json').read_text())
            completed_manifest(destination, manifest['collection_config_sha256'])
            (destination / 'data.csv.gz').write_bytes(b'broken')
            with self.assertRaises(ValueError):
                run_density_collection(output=directory, runs=5, workers=1)
            self.assertEqual(collect.call_count, 1)

    @patch('data_pipeline.collect_density.ProcessPoolExecutor', ThreadPoolExecutor)
    @patch('data_pipeline.collect_density.collect_one', side_effect=fake_collection)
    @patch('data_pipeline.collect_density.make_config', return_value=fixture_config())
    def test_stop_marker_and_explicit_resume(self, configure, collect):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'STOP').touch()
            result = run_density_collection(output=directory, runs=5, workers=1)
            self.assertEqual(result['status'], 'stopped')
            collect.assert_not_called()
            result = run_density_collection(output=directory, runs=5, workers=1,
                                            resume=True, max_new_runs=1)
            self.assertEqual(result['completed_runs'], 1)
            self.assertFalse((Path(directory) / 'STOP').exists())

    @patch('data_pipeline.collect_density.ProcessPoolExecutor', ThreadPoolExecutor)
    @patch('data_pipeline.collect_density.collect_one', side_effect=RuntimeError)
    @patch('data_pipeline.collect_density.make_config', return_value=fixture_config())
    def test_repeated_failure_stops_after_bounded_retries(self, configure, collect):
        with tempfile.TemporaryDirectory() as directory:
            result = run_density_collection(output=directory, runs=5, workers=1)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['completed_runs'], 0)
            self.assertEqual(collect.call_count, 3)
            self.assertTrue((Path(directory) / 'errors.log').exists())

    @patch('data_pipeline.collect_density.ProcessPoolExecutor', ThreadPoolExecutor)
    @patch('data_pipeline.collect_density.collect_one')
    @patch('data_pipeline.collect_density.make_config', return_value=fixture_config())
    @patch('data_pipeline.collect_density.shutil.disk_usage')
    def test_low_disk_stops_before_new_simulations(self, disk, configure, collect):
        disk.return_value.free = 0
        with tempfile.TemporaryDirectory() as directory:
            result = run_density_collection(output=directory, runs=5, workers=1)
            self.assertEqual(result['status'], 'disk_guard')
            collect.assert_not_called()

    def test_execution_revision_only_accepts_exact_validated_code(self):
        old = {**fixture_config(), 'code_sha256': {'collector.py': 'old'}}
        new = {**fixture_config(), 'code_sha256': {'collector.py': 'new'}}
        with tempfile.TemporaryDirectory() as directory:
            validate_execution_revision(directory, old, old)
            with self.assertRaises(ValueError):
                validate_execution_revision(directory, old, new)
            receipt = {'collection_config_sha256': config_signature(old),
                       'code_sha256': new['code_sha256'], 'validation': {'passed': True}}
            save_json(Path(directory) / 'execution_revision.json', receipt)
            validate_execution_revision(directory, old, new)
            for changed in ({**new, 'driver_num': 500},
                            {**new, 'code_sha256': {'collector.py': 'other'}}):
                with self.assertRaises(ValueError):
                    validate_execution_revision(directory, old, changed)
            receipt['validation']['passed'] = False
            save_json(Path(directory) / 'execution_revision.json', receipt)
            with self.assertRaises(ValueError):
                validate_execution_revision(directory, old, new)

    @patch('data_pipeline.collect_density.sleep')
    @patch('data_pipeline.collect_density.load_worker_pattern')
    def test_startup_stagger_only_applies_to_worker_initialization(self, load, sleep):
        import multiprocessing
        import data_pipeline.collect_density as collection
        old_pattern = collection._WORKER_PATTERN
        self.addCleanup(setattr, collection, '_WORKER_PATTERN', old_pattern)
        counter = multiprocessing.get_context('spawn').Value('i', 0)
        initialize_worker({}, counter, 2.0)
        sleep.assert_not_called()
        initialize_worker({}, counter, 2.0)
        sleep.assert_called_once_with(2.0)
        self.assertEqual(counter.value, 2)
        self.assertIs(collection._WORKER_PATTERN, load.return_value)

    def test_invalid_stagger_rejected_before_output_creation(self):
        for value in (-1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                run_density_collection(startup_stagger_seconds=value)


if __name__ == '__main__':
    unittest.main()
