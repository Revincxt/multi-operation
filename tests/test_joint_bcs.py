"""联合动作上下文下的一轮贪心与多轮 BCS 行为。"""

from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from baseline.compare_policies import joint_action_search
from baseline.compare_bcs import _validate_collection
from data_pipeline.collect_density import config_signature
from baseline.sequence import CONTEXT_COLUMNS
from data_pipeline.build_supervised_dataset import (
    HISTORY_ACTION_COLUMNS, PERIOD_METRIC_COLUMNS, aggregate_decision_metrics,
)
from src.env.actions import GridAction
from src.utils.joint_context import joint_action_columns
from tests.test_supervised_dataset import _raw_tables


class InteractionPredictor:
    """构造具有跨网格交互和坐标局部最优的可控预测器。"""

    def __init__(self, periods):
        self.preprocessor = SimpleNamespace(
            grid_ids=[0, 1], history_steps=1,
            history_features=list(PERIOD_METRIC_COLUMNS + HISTORY_ACTION_COLUMNS),
            context_columns=list(CONTEXT_COLUMNS) + list(joint_action_columns([0, 1])))
        self.periods = periods

    def predict(self, frame):
        lookup = np.array([[0.0, 5.0, 0.0],
                           [3.0, 4.0, 0.0],
                           [2.0, 10.0, 0.0]])
        prices = {1.0: 0, 2.0: 1, 3.0: 2}
        first = frame["joint_price_grid_0"].map(prices).to_numpy(dtype=int)
        second = frame["joint_price_grid_1"].map(prices).to_numpy(dtype=int)
        total = lookup[first, second]
        return pd.DataFrame({"target_platform_revenue": total / 2}, index=frame.index)


class JointBCSTests(unittest.TestCase):
    def test_collection_accepts_verified_revision_but_rejects_unapproved_changes(self):
        source = {"scenario": "low", "run_count": 10, "seed_start": 42,
                  "batch_size": 10, "data_dir": "data", "date": "2015-05-18",
                  "start_time": "05:00", "simulation_steps": 300,
                  "code_sha256": {"src/env/simulator_env.py": "original"}}
        approved = {"src/env/simulator_env.py": "verified"}
        current = {**source, "code_sha256": {**approved, "src/utils/joint_context.py": "new"}}
        revision = {"collection_config_sha256": config_signature(source),
                    "code_sha256": approved, "validation": {"passed": True}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "execution_revision.json"
            path.write_text(json.dumps(revision))
            with patch("baseline.compare_bcs.make_config", return_value=current):
                self.assertEqual(_validate_collection(temporary, source), config_signature(source))
            changed = {**current, "code_sha256": {
                **current["code_sha256"], "src/env/simulator_env.py": "unverified"}}
            for invalid in (changed, {**current, "simulation_steps": 301}):
                with patch("baseline.compare_bcs.make_config", return_value=invalid):
                    with self.assertRaises(ValueError):
                        _validate_collection(temporary, source)
            revision["validation"]["passed"] = False
            path.write_text(json.dumps(revision))
            with patch("baseline.compare_bcs.make_config", return_value=current):
                with self.assertRaises(ValueError):
                    _validate_collection(temporary, source)

    def test_iterative_bcs_improves_on_one_pass_greedy_with_interactions(self):
        plan, minute, metadata = _raw_tables()
        second_plan, second_minute = plan.copy(), minute.copy()
        second_plan["grid_id"], second_minute["grid_id"] = 1, 1
        periods = aggregate_decision_metrics(
            pd.concat([plan, second_plan], ignore_index=True),
            pd.concat([minute, second_minute], ignore_index=True), metadata)
        predictor = InteractionPredictor(periods)
        history = periods[periods.decision_step < 2]
        initial = {0: GridAction(1.0, 1.0), 1: GridAction(1.0, 1.0)}

        greedy, _, greedy_report = joint_action_search(
            predictor, history, 2, 18240, 2, [1.0, 2.0, 3.0], [1.0], None,
            initial_actions=initial, max_sweeps=1)
        bcs, _, bcs_report = joint_action_search(
            predictor, history, 2, 18240, 2, [1.0, 2.0, 3.0], [1.0], None,
            initial_actions=initial, max_sweeps=5)

        self.assertEqual([greedy[grid].price_multiplier for grid in (0, 1)], [2.0, 2.0])
        self.assertEqual([bcs[grid].price_multiplier for grid in (0, 1)], [3.0, 2.0])
        self.assertEqual(greedy_report["predicted_total_platform_revenue"], 4.0)
        self.assertEqual(bcs_report["predicted_total_platform_revenue"], 10.0)
        self.assertTrue(bcs_report["converged"])
        self.assertGreater(bcs_report["sweeps"], greedy_report["sweeps"])


if __name__ == "__main__":
    unittest.main()
