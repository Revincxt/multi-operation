"""检验配对统计和服务率约束的研究口径。"""

import unittest

import numpy as np
import pandas as pd

from constrained_rl.report import feasibility, summarize


class ConstrainedReportTests(unittest.TestCase):
    def frame(self):
        rows = []
        for policy in ("fixed_f3", "revenue_control", "constrained"):
            for seed, demand, matches in ((1, 10., 5.), (2, 90., 81.)):
                rows.append({"policy": policy, "seed": seed, "cohort_demand": demand,
                    "unique_matched": matches, "queue_wait_seconds": demand*(100 if seed == 1 else 200),
                    "demand_satisfaction": matches/demand,
                    "wait_per_demand_seconds": 100 if seed == 1 else 200,
                    "platform_revenue": seed*100 + (10 if policy == "constrained" else 0)})
        return pd.DataFrame(rows)

    def test_service_and_wait_estimates_keep_denominators(self):
        summary, _ = summarize(self.frame(), draws=100)
        values = summary[summary.policy == "fixed_f3"].set_index("metric")
        self.assertAlmostEqual(values.loc["demand_satisfaction", "estimate"], .86)
        self.assertAlmostEqual(values.loc["wait_per_demand_seconds", "estimate"], 190.)
        self.assertNotAlmostEqual(values.loc["demand_satisfaction", "estimate"], .7)

    def test_paired_revenue_difference_is_invariant_to_seed_reordering(self):
        first, paired = summarize(self.frame().sample(frac=1., random_state=31), draws=200)
        selected = paired[(paired.policy == "constrained") & (paired.reference == "fixed_f3")
                          & (paired.metric == "platform_revenue")].iloc[0]
        self.assertAlmostEqual(selected.difference, 10.)
        self.assertAlmostEqual(selected.ci95_low, 10.)
        self.assertAlmostEqual(selected.ci95_high, 10.)
        self.assertEqual(selected.favorable_scenarios, 2)
        _, other = summarize(self.frame(), draws=200)
        pd.testing.assert_frame_equal(paired, other)

    def test_incomplete_pairs_and_zero_demand_are_rejected(self):
        with self.assertRaises(ValueError):
            summarize(self.frame().iloc[1:], draws=100)
        frame = self.frame()
        frame.loc[0, "cohort_demand"] = 0
        with self.assertRaises(ValueError):
            summarize(frame, draws=100)

    def test_point_feasibility_does_not_claim_confident_feasibility(self):
        summary, _ = summarize(self.frame(), draws=1000)
        rows = feasibility(summary, {"service_floor": .8, "wait_limit_seconds_per_demand": 195.})
        for row in rows:
            self.assertTrue(row["point_feasible"])
            self.assertFalse(row["service_ci_above_floor"])
            self.assertFalse(row["wait_ci_below_limit"])


if __name__ == "__main__":
    unittest.main()
