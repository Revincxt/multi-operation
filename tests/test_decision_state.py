"""决策前状态的时间边界、邻区口径和离线/在线输入一致性。"""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.env.observation import (
    LOCAL_STATE_COLUMNS, NEIGHBOR_STATE_COLUMNS, STATE_COLUMNS,
    capture_decision_state, load_neighbor_adjacency, state_columns, state_metadata,
    validate_adjacency, validate_decision_states,
)
from data_pipeline.build_supervised_dataset import (
    aggregate_decision_metrics, build_history_samples, build_supervised_dataset,
)
from data_pipeline.collect_trajectories import run_trajectory_collection
from data_pipeline.validate_supervised_dataset import validate_supervised_dataset
from tests.test_supervised_dataset import _raw_tables


class DecisionStateTests(unittest.TestCase):
    def test_capture_is_read_only_and_sums_only_first_hop_neighbors(self):
        adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        sim = SimpleNamespace(
            grid_num=3, wait_requests=pd.DataFrame({'origin_grid_id': [0, 0, 2]}),
            driver_table=pd.DataFrame({'grid_id': [0, 1, 1, 2, 2, 0],
                                       'status': [0, 0, 1, 2, 4, 3]}))
        orders, drivers = sim.wait_requests.copy(), sim.driver_table.copy()
        state = capture_decision_state(sim, adjacency)
        local = np.array([[2, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 1]])
        np.testing.assert_array_equal(state[list(LOCAL_STATE_COLUMNS)], local)
        np.testing.assert_array_equal(state[list(NEIGHBOR_STATE_COLUMNS)], adjacency @ local)
        validate_decision_states(state.assign(decision_step=0), adjacency)
        pd.testing.assert_frame_equal(sim.wait_requests, orders)
        pd.testing.assert_frame_equal(sim.driver_table, drivers)
        sim.wait_requests.loc[0, 'origin_grid_id'] = 3
        with self.assertRaises(ValueError):
            capture_decision_state(sim, adjacency)

    def test_rejects_missing_topology_self_loops_and_partial_state(self):
        for invalid in (np.eye(2), [[0, 1], [0, 0]], [[0, 0.5], [0.5, 0]]):
            with self.assertRaises(ValueError):
                validate_adjacency(invalid, 2)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_neighbor_adjacency(directory, 2)
            self.assertEqual(load_neighbor_adjacency(directory, 1).tolist(), [[0]])
        with self.assertRaises(ValueError):
            state_columns([STATE_COLUMNS[0]])

    @staticmethod
    def state_tables():
        plan, minute, metadata = _raw_tables()
        for index, column in enumerate(LOCAL_STATE_COLUMNS):
            plan[column] = plan.decision_step * 10 + index
        for column in NEIGHBOR_STATE_COLUMNS:
            plan[column] = 0
        metadata.update(state_metadata(np.zeros((1, 1), dtype=int)))
        return plan, minute, metadata

    def test_current_features_come_from_snapshot_not_period_mean_or_target(self):
        plan, minute, metadata = self.state_tables()
        periods = aggregate_decision_metrics(plan, minute, metadata)
        samples = build_history_samples(periods, 1)
        np.testing.assert_array_equal(samples[list(STATE_COLUMNS)], plan.iloc[1:][list(STATE_COLUMNS)])
        self.assertFalse(any(f'target_{column}' in samples for column in STATE_COLUMNS))
        self.assertNotEqual(samples.iloc[0].start_idle_driver_num, samples.iloc[0].target_idle_driver_num)
        modified = minute.copy()
        modified.loc[modified.decision_step == 1, 'idle_driver_num'] = 999
        changed = build_history_samples(aggregate_decision_metrics(plan, modified, metadata), 1)
        pd.testing.assert_series_equal(samples.iloc[0][list(STATE_COLUMNS)],
                                       changed.iloc[0][list(STATE_COLUMNS)])
        bad = plan.copy()
        bad.loc[0, NEIGHBOR_STATE_COLUMNS[0]] = 1
        with self.assertRaises(ValueError):
            aggregate_decision_metrics(bad, minute, metadata)

    def test_builder_validator_and_old_new_isolation(self):
        plan, minute, metadata = self.state_tables()
        plan['price_multiplier'] = 1.0
        plan['matching_radius_km'] = 1.0
        plan['joint_action_id'] = 0
        plan['is_baseline_action'] = True
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'raw' / 'run_0'
            raw.mkdir(parents=True)
            plan.to_csv(raw / 'action_plan.csv', index=False)
            minute.to_csv(raw / 'grid_minute_metrics.csv', index=False)
            (raw / 'manifest.json').write_text(json.dumps(metadata))
            dataset, manifest = build_supervised_dataset(root / 'raw', root / 'dataset', history_steps=1)
            self.assertTrue(set(STATE_COLUMNS).issubset(manifest['feature_columns']))
            self.assertEqual(manifest['state_feature_columns'], list(STATE_COLUMNS))
            self.assertTrue(validate_supervised_dataset(root / 'dataset')['passed'])
            _, joint_manifest = build_supervised_dataset(
                root / 'raw', root / 'joint_dataset', history_steps=1,
                joint_action_context=True)
            self.assertTrue(joint_manifest['joint_action_context'])
            joint_report = validate_supervised_dataset(root / 'joint_dataset')
            self.assertTrue(joint_report['passed'])
            self.assertTrue(joint_report['checks']['joint_action_context_matches_each_grid'])
            dataset.loc[0, NEIGHBOR_STATE_COLUMNS[0]] = 1
            dataset.to_csv(root / 'dataset/supervised_dataset.csv', index=False)
            report = validate_supervised_dataset(root / 'dataset')
            self.assertFalse(report['checks']['pre_action_state_and_neighbor_sums_are_valid'])
            old_plan, old_minute, old_metadata = _raw_tables('run_old')
            old = root / 'raw/run_old'
            old.mkdir()
            old_plan.to_csv(old / 'action_plan.csv', index=False)
            old_minute.to_csv(old / 'grid_minute_metrics.csv', index=False)
            (old / 'manifest.json').write_text(json.dumps(old_metadata))
            with self.assertRaises(ValueError):
                build_supervised_dataset(root / 'raw', root / 'mixed', history_steps=1)

    @patch('data_pipeline.collect_trajectories.check_collection_inputs')
    @patch('data_pipeline.collect_trajectories.SimulatorPattern')
    def test_collector_snapshots_before_first_action_and_cancellation(self, pattern, check):
        pattern.return_value.request_all = {0: [], 86399: []}
        pattern.return_value.driver_info = pd.DataFrame({
            'driver_id': [1], 'start_time': [18000], 'end_time': [36000],
            'lng': [-73.99], 'lat': [40.75], 'grid_id': [0]})
        captures = []

        def record(sim, adjacency):
            # 有意让动作前排队数区别于动作后候选数：超时订单会在 step 内被删掉。
            if sim.current_step == 2:
                sim.wait_requests = pd.DataFrame({column: [0] for column in sim.request_columns})
                sim.wait_requests['origin_grid_id'] = 0
                sim.wait_requests['wait_time'] = 1000
                sim.wait_requests['maximum_wait_time'] = 1
            state = capture_decision_state(sim, adjacency)
            captures.append((sim.current_step, state.copy()))
            return state

        with tempfile.TemporaryDirectory() as directory, \
                patch('data_pipeline.collect_trajectories.capture_decision_state', side_effect=record):
            result = run_trajectory_collection(
                '2015-05-18', grid_num=1, driver_num=1, simulation_steps=4, decision_freq=2,
                price_action_values=[1.0], matching_radius_values=[1.0],
                data_dir=directory, output_dir=Path(directory) / 'raw')
            self.assertEqual([step for step, _ in captures], [0, 2])
            self.assertEqual(result['action_plan'].start_waiting_request_num.tolist(), [0, 1])
            self.assertEqual(result['action_plan'].start_idle_driver_num.tolist(), [1, 1])
            minute = result['minute_metrics'].set_index('minute_step')
            self.assertEqual(minute.loc[2, 'candidate_request_num'], 0)


class StateModelInputTests(unittest.TestCase):
    def test_new_context_reaches_both_models_and_checkpoint_preprocessor(self):
        from baseline.data import load_dataset
        from baseline.sequence import CONTEXT_COLUMNS, SequencePreprocessor
        from baseline.train_lstm import train_lstm
        from baseline.train_timesnet import train_timesnet
        from tests.test_lstm import _dataset, _write_dataset
        frame = _dataset()
        for index, column in enumerate(STATE_COLUMNS):
            frame[column] = frame.decision_step + index
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_dataset(root / 'dataset', frame)
            path = root / 'dataset/dataset_manifest.json'
            manifest = json.loads(path.read_text())
            manifest['feature_columns'] += list(STATE_COLUMNS)
            path.write_text(json.dumps(manifest))
            splits, processor, _ = load_dataset(root / 'dataset', chunk_size=3)
            self.assertEqual(processor.context_columns, list(CONTEXT_COLUMNS + STATE_COLUMNS))
            self.assertEqual(splits['train'].arrays['context'].shape[1], 12)
            restored = SequencePreprocessor.from_state_dict(processor.state_dict())
            self.assertEqual(restored.context_columns, processor.context_columns)
            _, context, _ = restored.transform_inputs(frame)
            actual = context * processor.scalers['context']['scale'] + processor.scalers['context']['mean']
            np.testing.assert_allclose(actual[:, 4:], frame[list(STATE_COLUMNS)], atol=1e-5)
            for trainer in (train_lstm, train_timesnet):
                config = trainer(root / 'dataset', root / 'unused', prepare_only=True,
                                 hidden_size=8, batch_size=4, threads=1)
                self.assertEqual(config['model']['context_size'], 12)
                self.assertEqual(config['forward_check']['output_shape'], [4, 6])
            self.assertFalse((root / 'unused').exists())

    def test_online_candidates_use_same_current_state_for_each_action(self):
        from baseline.compare_policies import candidate_features
        from baseline.sequence import CONTEXT_COLUMNS
        from data_pipeline.build_supervised_dataset import PERIOD_METRIC_COLUMNS, HISTORY_ACTION_COLUMNS
        plan, minute, metadata = DecisionStateTests.state_tables()
        periods = aggregate_decision_metrics(plan, minute, metadata)
        processor = SimpleNamespace(grid_ids=[0], history_steps=1,
                                    history_features=PERIOD_METRIC_COLUMNS + HISTORY_ACTION_COLUMNS,
                                    context_columns=CONTEXT_COLUMNS + STATE_COLUMNS)
        current = plan.loc[plan.decision_step == 2, ['grid_id', *STATE_COLUMNS]]
        candidates = candidate_features(periods[periods.decision_step < 2], processor,
                                        2, 18240, 2, [1.0, 1.6], [1.0, 2.25], current)
        offline = build_history_samples(periods, 1).iloc[-1]
        for column in STATE_COLUMNS:
            self.assertTrue(candidates[column].eq(offline[column]).all())
        self.assertFalse(any(column.startswith('target_') for column in candidates))
        with self.assertRaises(ValueError):
            candidate_features(periods[periods.decision_step < 2], processor,
                               2, 18240, 2, [1.0], [1.0])


if __name__ == '__main__':
    unittest.main()
