import unittest

import numpy as np
import pandas as pd

from src.env.actions import (
    DEFAULT_MATCHING_RADII_KM, DEFAULT_PRICE_MULTIPLIERS, GridAction, uniform_grid_actions,
)
from src.env.simulator_env import Simulator
from src.utils.utilities import order_dispatch


class GridActionTests(unittest.TestCase):
    def test_retired_behavior_switches_are_rejected_even_when_false(self):
        for name in ('post_match_cancellation', 'driver_supply_response', 'repo_mode'):
            for value in (False, True, None):
                with self.subTest(name=name, value=value), self.assertRaises(TypeError):
                    Simulator(grid_num=1, grid_action=uniform_grid_actions(1, 1.0, 1.0),
                              **{name: value})

    @staticmethod
    def _state_for(sim):
        sim.time = 18000
        sim.current_step = 0
        sim.evaluate_df = pd.DataFrame({
            'origin_grid_id': np.arange(sim.grid_num),
        })

    def test_mapping_resolves_one_action_per_grid(self):
        sim = Simulator(
            grid_num=3,
            grid_action={
                0: GridAction(1.4, 0.5),
                1: GridAction(1.0, 1.0),
                2: GridAction(1.6, 2.0),
            },
        )
        self._state_for(sim)

        action_table = sim.current_grid_actions()

        np.testing.assert_allclose(
            action_table['price_multiplier'], [1.4, 1.0, 1.6])
        np.testing.assert_allclose(
            action_table['matching_radius_km'], [0.5, 1.0, 2.0])

    def test_missing_grid_action_is_rejected(self):
        sim = Simulator(
            grid_num=2,
            grid_action={0: GridAction(1.0, 1.0)},
        )
        self._state_for(sim)

        with self.assertRaisesRegex(ValueError, 'missing='):
            sim.current_grid_actions()

    def test_callable_can_change_actions_by_time(self):
        calls = []

        def action_fn(sim_time, grid_state):
            calls.append(sim_time)
            price = 1.7 if sim_time >= 60 else 1.0
            return {
                int(g): GridAction(price, 1.5)
                for g in grid_state['origin_grid_id']
            }

        sim = Simulator(grid_num=2, decision_freq=5, grid_action=action_fn)
        sim.time = 60
        sim.current_step = 1
        sim.evaluate_df = pd.DataFrame({'origin_grid_id': [0, 1]})

        first = sim.current_grid_actions()
        second = sim.current_grid_actions()

        np.testing.assert_allclose(first['price_multiplier'], [1.7, 1.7])
        np.testing.assert_allclose(first['matching_radius_km'], [1.5, 1.5])
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(calls, [60])

        sim.current_step = 5
        sim.time = 300
        sim.current_grid_actions()
        self.assertEqual(calls, [60, 300])

    def test_invalid_discrete_action_is_rejected(self):
        sim = Simulator(grid_num=1, grid_action={0: GridAction(1.2, 1.0)})
        self._state_for(sim)

        with self.assertRaisesRegex(ValueError, 'discrete action set'):
            sim.current_grid_actions()

    def test_new_defaults_cover_all_110_joint_actions(self):
        self.assertEqual(DEFAULT_PRICE_MULTIPLIERS,
                         (0.4, 1.0, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 4.0))
        self.assertEqual(DEFAULT_MATCHING_RADII_KM,
                         (0.5, 1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0))
        pairs = [(price, radius) for price in DEFAULT_PRICE_MULTIPLIERS
                 for radius in DEFAULT_MATCHING_RADII_KM]
        sim = Simulator(grid_num=110, grid_action={
            grid: GridAction(*pair) for grid, pair in enumerate(pairs)})
        self._state_for(sim)
        np.testing.assert_allclose(
            sim.current_grid_actions()[['price_multiplier', 'matching_radius_km']], pairs)

    def test_removed_values_are_not_default_candidates(self):
        for price, radius in ((0.7, 1.0), (1.3, 1.0), (2.2, 1.0), (1.0, 3.5)):
            with self.subTest(price=price, radius=radius):
                sim = Simulator(grid_num=1, grid_action={0: GridAction(price, radius)})
                self._state_for(sim)
                with self.assertRaises(ValueError):
                    sim.current_grid_actions()

    def test_row_level_action_sequence_is_rejected(self):
        with self.assertRaisesRegex(TypeError, 'complete grid mapping'):
            Simulator(grid_num=2, grid_action=[
                GridAction(1.0, 0.5), GridAction(1.0, 2.0)])


class DynamicRadiusDispatchTests(unittest.TestCase):
    def test_origin_grid_radius_controls_candidate_edges(self):
        # At the equator, 0.008 degrees longitude is approximately 0.89 km.
        requests = pd.DataFrame({
            'origin_lng': [0.000, 0.020],
            'origin_lat': [0.0, 0.0],
            'order_id': [10, 20],
            'weight': [1.0, 1.0],
            'origin_grid_id': [0, 1],
        })
        drivers = pd.DataFrame({
            'lng': [0.008, 0.028],
            'lat': [0.0, 0.0],
            'driver_id': [100, 200],
            'status': [0, 0],
        })

        matches, distances = order_dispatch(
            requests, drivers, matching_radius_by_grid=np.array([0.5, 1.0]))

        self.assertEqual([int(match[0]) for match in matches], [20])
        self.assertEqual([int(match[1]) for match in matches], [200])
        self.assertEqual(len(distances), 1)
        self.assertLess(distances[0], 1.0)

    def test_orders_from_same_grid_cannot_receive_different_radii(self):
        requests = pd.DataFrame({
            'origin_lng': [0.0, 0.0],
            'origin_lat': [0.0, 0.0],
            'order_id': [10, 20],
            'weight': [1.0, 1.0],
            'origin_grid_id': [0, 0],
        })
        drivers = pd.DataFrame({
            'lng': [0.008], 'lat': [0.0],
            'driver_id': [100], 'status': [0],
        })

        # The array is indexed by grid, so both orders use grid 0's 0.5 km
        # radius.  The second array element is grid 1's action, not order 2's.
        matches, _ = order_dispatch(
            requests, drivers, matching_radius_by_grid=np.array([0.5, 1.0]))

        self.assertEqual(matches, [])

    def test_scalar_radius_is_rejected(self):
        requests = pd.DataFrame({
            'origin_lng': [0.0], 'origin_lat': [0.0],
            'order_id': [10], 'weight': [1.0], 'origin_grid_id': [0],
        })
        drivers = pd.DataFrame({
            'lng': [0.008], 'lat': [0.0],
            'driver_id': [100], 'status': [0],
        })

        with self.assertRaisesRegex(ValueError, 'one-dimensional'):
            order_dispatch(requests, drivers, 1.0)


class MigrationTests(unittest.TestCase):
    def test_uniform_helper_builds_complete_grid_mapping(self):
        actions = uniform_grid_actions(3, 1.3, 1.5)

        self.assertEqual(set(actions), {0, 1, 2})
        self.assertTrue(all(action == GridAction(1.3, 1.5)
                            for action in actions.values()))

    def test_legacy_constructor_arguments_are_rejected_with_guidance(self):
        with self.assertRaisesRegex(TypeError, 'use grid_action'):
            Simulator(
                grid_num=1,
                grid_action=uniform_grid_actions(1),
                price_multiplier=1.0,
            )

    def test_driver_num_controls_initialized_supply(self):
        driver_info = pd.DataFrame({
            'driver_id': [1, 2, 3],
            'start_time': [18000, 18000, 18000],
            'end_time': [36000, 36000, 36000],
            'lng': [-73.99, -73.98, -73.97],
            'lat': [40.75, 40.76, 40.77],
            'grid_id': [0, 0, 0],
        })
        sim = Simulator(
            grid_num=1,
            driver_num=2,
            grid_action=uniform_grid_actions(1),
        )

        sim.reset(
            seed=42,
            given_data=True,
            request_databases={},
            driver_info=driver_info,
        )

        self.assertEqual(len(sim.driver_table), 2)


if __name__ == '__main__':
    unittest.main()
