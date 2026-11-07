import unittest
from unittest.mock import patch

import numpy as np

from data_pipeline.collect_trajectories import (
    action_mapping_for_decision,
    build_joint_action_plan,
)
from src.env.actions import DEFAULT_MATCHING_RADII_KM, DEFAULT_PRICE_MULTIPLIERS, GridAction
from src.env.exploration import ExplorationPlanner


def _mock_masks(grid_num):
    identity = np.eye(grid_num, dtype=bool)
    neighbors = [{g} for g in range(grid_num)]
    conflicts = [set() for _ in range(grid_num)]
    return identity, identity, identity, neighbors, neighbors, conflicts


class JointExplorationTests(unittest.TestCase):
    def test_uniform_mode_keeps_full_support_and_allows_repeated_actions(self):
        plan = build_joint_action_plan(
            np.random.default_rng(0), grid_num=110, simulation_steps=100, decision_freq=5)
        expected = {(price, radius) for price in DEFAULT_PRICE_MULTIPLIERS
                    for radius in DEFAULT_MATCHING_RADII_KM}
        self.assertEqual(len(expected), 110)
        self.assertEqual(plan.attrs['sampling_mode'], 'uniform')
        self.assertTrue(plan.attrs['action_sampling']['with_replacement'])
        self.assertEqual(set(plan[['price_multiplier', 'matching_radius_km']]
                             .itertuples(index=False, name=None)), expected)
        self.assertTrue(plan.groupby('decision_step').joint_action_id.nunique().lt(110).all())

    def test_preferred_mode_holds_joint_vectors_for_three_decisions(self):
        plan = build_joint_action_plan(np.random.default_rng(1), 6, 30, 5)
        self.assertEqual(plan.attrs['sampling_mode'], 'preferred')
        self.assertTrue(plan.price_multiplier.between(1.4, 1.8).all())
        self.assertTrue(plan.matching_radius_km.between(1.5, 2.5).all())
        for block in (0, 3):
            expected = plan.loc[plan.decision_step == block, 'joint_action_id'].to_numpy()
            for step in range(block, block + 3):
                np.testing.assert_array_equal(
                    plan.loc[plan.decision_step == step, 'joint_action_id'].to_numpy(), expected)

    def test_reference_mode_can_collect_whole_city_f3_for_a_full_run(self):
        plan = build_joint_action_plan(np.random.default_rng(9), 6, 300, 5)
        self.assertEqual(plan.attrs['sampling_mode'], 'reference')
        self.assertEqual(plan.attrs['reference_joint_action'], [1.6, 2.0])
        self.assertTrue(plan.price_multiplier.eq(1.6).all())
        self.assertTrue(plan.matching_radius_km.eq(2.0).all())
        self.assertEqual(plan.joint_action_id.nunique(), 1)

    def test_mixture_modes_have_expected_run_frequencies(self):
        modes = [build_joint_action_plan(np.random.default_rng(seed), 6, 1, 1)
                 .attrs['sampling_mode'] for seed in range(1000)]
        for name, expected in (('preferred', 0.6), ('uniform', 0.2), ('reference', 0.2)):
            self.assertAlmostEqual(modes.count(name) / len(modes), expected, delta=0.06)

    def test_default_spatial_plan_uses_new_candidates_and_uniform_weights(self):
        with patch('src.env.exploration._load_or_build_masks', return_value=_mock_masks(4)):
            planner = ExplorationPlanner(np.zeros((4, 4), dtype=int), grid_num=4)
        self.assertEqual(set(planner.price_action_weights), set(DEFAULT_PRICE_MULTIPLIERS) - {1.0})
        self.assertEqual(set(planner.matching_radius_weights), set(DEFAULT_MATCHING_RADII_KM))
        np.testing.assert_allclose(list(planner.price_action_weights.values()), 0.1)
        np.testing.assert_allclose(list(planner.matching_radius_weights.values()), 0.1)

    def test_spatial_plan_samples_and_records_both_action_dimensions(self):
        grid_num = 4
        with patch(
            'src.env.exploration._load_or_build_masks',
            return_value=_mock_masks(grid_num),
        ):
            planner = ExplorationPlanner(
                np.zeros((grid_num, grid_num), dtype=int),
                grid_num=grid_num,
                treatment_minutes=2,
                price_action_weights={1.3: 1.0},
                matching_radius_weights={2.0: 1.0},
            )

        events, metadata = planner.build_plan(
            np.random.default_rng(7), n_events=2,
            simulation_steps=20, margin_minutes=2)

        self.assertEqual(len(events), 2)
        self.assertTrue(all(
            event['exploration_price_multiplier'] == 1.3
            and event['matching_radius_km'] == 2.0
            for event in events))
        self.assertEqual(metadata['joint_action_sampling'], 'independent_product')
        self.assertEqual(metadata['price_action_weights'], {1.3: 1.0})
        self.assertEqual(metadata['matching_radius_weights'], {2.0: 1.0})

    def test_supervised_plan_is_deterministic_complete_and_with_replacement(self):
        kwargs = dict(
            grid_num=9,
            simulation_steps=5,
            decision_freq=2,
            price_action_values=[1.0, 1.3],
            matching_radius_values=[1.0, 2.0],
        )

        first = build_joint_action_plan(np.random.default_rng(11), **kwargs)
        second = build_joint_action_plan(np.random.default_rng(11), **kwargs)

        self.assertTrue(first.equals(second))
        self.assertEqual(len(first), 3 * 9)
        self.assertFalse(first.duplicated(['decision_step', 'grid_id']).any())
        self.assertEqual(
            first.groupby('decision_step')['grid_id'].nunique().tolist(),
            [9, 9, 9],
        )
        for _, period in first.groupby('decision_step'):
            counts = period.groupby(
                ['price_multiplier', 'matching_radius_km']).size()
            self.assertLessEqual(len(counts), 4)
            self.assertGreater(int(counts.max()), 1)
        self.assertEqual(first.attrs['sampling_mode'], 'uniform')
        self.assertEqual(
            first['decision_end_minute_step_exclusive'].max(), 5)

    def test_supervised_plan_resolves_complete_grid_mapping(self):
        plan = build_joint_action_plan(
            np.random.default_rng(3),
            grid_num=4,
            simulation_steps=2,
            decision_freq=2,
            price_action_values=[1.0, 1.3],
            matching_radius_values=[1.0, 2.0],
        )

        actions = action_mapping_for_decision(plan, 0, grid_num=4)

        self.assertEqual(set(actions), {0, 1, 2, 3})
        self.assertTrue(all(isinstance(action, GridAction)
                            for action in actions.values()))

    def test_supervised_plan_rejects_duplicate_action_values(self):
        with self.assertRaises(ValueError):
            build_joint_action_plan(
                np.random.default_rng(3),
                grid_num=4,
                simulation_steps=2,
                price_action_values=[1.0, 1.0],
                matching_radius_values=[1.0],
            )

    def test_invalid_action_weights_are_rejected(self):
        grid_num = 2
        with patch(
            'src.env.exploration._load_or_build_masks',
            return_value=_mock_masks(grid_num),
        ):
            with self.assertRaisesRegex(ValueError, 'price_action_weights'):
                ExplorationPlanner(
                    np.zeros((grid_num, grid_num), dtype=int),
                    grid_num=grid_num,
                    price_action_weights={1.3: 0.0},
                )


if __name__ == '__main__':
    unittest.main()
