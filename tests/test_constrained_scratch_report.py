"""检查从零训练报告的封存测试门槛、时序标签及验证集口径。"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from constrained_rl import report, scratch_report


class ScratchReportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.plan = {"constraints": {"service_floor": .8, "wait_limit_seconds_per_demand": 195.},
                     "expected_test_policies": list(report.LABELS), "seeds": {"test": [101, 103]}}
        self.write_plan()
        self.plots = patch.object(report, "plot_test")
        self.test_plot = self.plots.start()
        self.addCleanup(self.plots.stop)

    def write_plan(self):
        (self.output / "protocol.json").write_text(json.dumps(self.plan))

    def evaluation(self):
        rows = []
        for policy in self.plan["expected_test_policies"]:
            for seed, demand, covered, wait in ((101, 10, 5, 100), (103, 90, 81, 200)):
                rows.append({"phase": "test", "policy": policy, "iteration": 6250, "seed": seed,
                             "cohort_demand": demand, "unique_matched": covered,
                             "queue_wait_seconds": demand*wait,
                             "platform_revenue": seed*10 + (7 if policy == "constrained" else 0)})
        return pd.DataFrame(rows)

    def test_final_report_uses_protocol_and_paired_estimands(self):
        self.evaluation().to_csv(self.output / "evaluation.csv", index=False)
        result = scratch_report.run(self.output)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["test_scenarios"], 2)
        self.assertIn("From-scratch", result["scope"])
        self.assertIn("not their initialization", result["initialization"])
        summary = pd.read_csv(self.output / "policy_summary.csv")
        value = summary[(summary.policy == "constrained") & (summary.metric == "demand_satisfaction")].iloc[0]
        self.assertAlmostEqual(value.estimate, .86)
        self.assertEqual(value.aggregation, "ratio_of_sums")
        paired = pd.read_csv(self.output / "paired_statistics.csv")
        value = paired[(paired.policy == "constrained") & (paired.reference == "fixed_f3")
                       & (paired.metric == "platform_revenue")].iloc[0]
        self.assertAlmostEqual(value.difference, 7.)
        self.assertAlmostEqual(value.ci95_low, 7.)
        self.assertAlmostEqual(value.ci95_high, 7.)
        self.test_plot.assert_called_once()

    def test_missing_wrong_duplicate_or_extra_test_scenarios_do_not_report(self):
        frame = self.evaluation()
        wrong = frame.copy()
        wrong.loc[0, "seed"] = 999
        variants = [frame.iloc[1:], wrong, pd.concat([frame, frame.iloc[:1]]),
                    frame[frame.policy != "frozen_ar_ppo"]]
        for candidate in variants:
            with self.subTest(rows=len(candidate)):
                candidate.to_csv(self.output / "evaluation.csv", index=False)
                self.assertEqual(scratch_report.run(self.output)["status"], "test_incomplete")
                self.assertFalse((self.output / "report.json").exists())
        self.test_plot.assert_not_called()

    def test_sealed_test_seed_list_is_required(self):
        del self.plan["seeds"]
        self.write_plan()
        with self.assertRaises(ValueError):
            scratch_report.run(self.output)

    def test_validation_uses_logged_episode_count_and_total_denominators(self):
        frame = self.evaluation()
        frame["phase"] = "validation"
        frame["iteration"] = 8
        training = pd.DataFrame([{"method": method, "iteration": 8, "episodes": 128}
                                 for method in scratch_report.METHODS])
        result = scratch_report.validation_history(frame, training)
        self.assertEqual(set(result.method), set(scratch_report.METHODS))
        self.assertEqual(set(result.episodes), {128})
        for row in result.itertuples():
            self.assertAlmostEqual(row.demand_satisfaction, .86)
            self.assertAlmostEqual(row.wait_per_demand_seconds, 190.)

    def test_single_update_training_plot_has_visible_points_and_correct_axes(self):
        frame = pd.DataFrame([{
            "method": method, "iteration": 1, "episodes": 16, "revenue_mean_usd": 7000,
            "demand_satisfaction": .81, "wait_per_demand_seconds": 190., "joint_kl": .01,
            "lambda_service": .1, "lambda_wait": .2, "cost_service_mean": -.01,
            "cost_wait_mean": -.05} for method in scratch_report.METHODS])
        captured = []
        with patch.object(report, "save_figure", side_effect=lambda fig, path: captured.append(fig)):
            scratch_report.plot_training(frame, self.plan["constraints"], self.output)
        self.assertEqual(len(captured), 1)
        self.assertTrue(all(ax.get_xlabel() == "Training episodes" for ax in captured[0].axes))
        self.assertEqual(captured[0].axes[0].lines[0].get_marker(), "o")
        self.assertFalse(captured[0].axes[4].yaxis.get_major_formatter().get_useOffset())


if __name__ == "__main__":
    unittest.main()
