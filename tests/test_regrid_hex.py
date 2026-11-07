"""六边形重划分、输入隔离和采集入口回归测试。"""

import importlib.util
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_pipeline.collect_trajectories import check_collection_inputs, run_trajectory_collection
from src.env.simulator_pattern import SimulatorPattern, input_data_paths

HAS_GEOMETRY = importlib.util.find_spec('shapely') is not None
if HAS_GEOMETRY:
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union
    from data_pipeline.regrid_hex import (
        LocalProjection, assign_grids, build_layout, hexagon,
        layout_tables, remap_drivers, remap_orders,
    )


@unittest.skipUnless(HAS_GEOMETRY, 'Install requirements-grid.txt for spatial tests')
class HexLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.region = Polygon([(-3, -8), (-1, -9), (5, 7), (3, 9), (0, 3)])
        cls.radius, cls.phase, cls.cells = build_layout(cls.region, 6)
        cls.projection = LocalProjection(-74.0, 40.75)
        cls.centers = np.array([cell['center'] for cell in cls.cells])

    def test_exactly_six_regular_hexagons_preserve_irregular_footprint(self):
        self.assertEqual(len(self.cells), 6)
        combined = unary_union([cell['region'] for cell in self.cells])
        self.assertLess(self.region.symmetric_difference(combined).area, 1e-7)
        self.assertAlmostEqual(sum(cell['region'].area for cell in self.cells), combined.area)
        for cell in self.cells:
            vertices = np.asarray(cell['full'].exterior.coords)
            self.assertEqual(len(vertices), 7)
            np.testing.assert_allclose(np.linalg.norm(np.diff(vertices, axis=0), axis=1), self.radius)

    def test_assignments_match_polygon_interiors(self):
        points = np.array([[cell['region'].representative_point().x,
                            cell['region'].representative_point().y] for cell in self.cells])
        assigned = assign_grids(self.projection.inverse(points), self.projection, self.centers)
        np.testing.assert_array_equal(assigned, np.arange(6))
        with self.assertRaises(ValueError):
            assign_grids([[np.nan, 40.75]], self.projection, self.centers)

    def test_adjacency_is_symmetric_and_reposition_points_are_inside(self):
        info, adjacency = layout_tables(self.cells, self.projection)
        np.testing.assert_array_equal(adjacency, adjacency.T)
        np.testing.assert_array_equal(np.diag(adjacency), np.zeros(6))
        xy = self.projection.forward(info[['center_lon', 'center_lat']].to_numpy())
        for cell, point in zip(self.cells, xy):
            self.assertTrue(cell['region'].buffer(1e-8).covers(Point(point)))

    def test_remap_preserves_orders_coordinates_and_outside_points(self):
        first = self.cells[0]['region'].representative_point()
        last = self.cells[-1]['region'].representative_point()
        p0, p1, outside = self.projection.inverse([[first.x, first.y], [last.x, last.y], [100, 100]])
        rows = [[10, 1, p0[1], p0[0], 2, p1[1], p1[0], 2.5, 18000, 30, 31, [], [], 123, 4.5, 0],
                [20, 1, outside[1], outside[0], 2, p0[1], p0[0], 3.0, 18001, 32, 33, [], [], 144, 5, 0]]
        original = {18000: rows, 18001: []}
        remapped, counts = remap_orders(original, self.projection, self.centers, self.region)
        self.assertEqual(set(original), set(remapped))
        self.assertEqual(counts['retained_order_count'], 2)
        self.assertEqual(counts['origin_outside_original_region'], 1)
        self.assertEqual(remapped[18000][0][9:11], [0, 5])
        self.assertEqual(original[18000][0][9:11], [30, 31])
        for old, new in zip(rows, remapped[18000]):
            self.assertEqual(old[:9], new[:9])
            self.assertEqual(old[11:], new[11:])
            self.assertTrue(0 <= new[9] < 6 and 0 <= new[10] < 6)
        drivers = pd.DataFrame({'driver_id': [1], 'lng': [p0[0]], 'lat': [p0[1]],
                                'grid_id': [30], 'target_loc_lng': [p1[0]],
                                'target_loc_lat': [p1[1]], 'target_grid_id': [31]})
        result = remap_drivers(drivers, self.projection, self.centers)
        self.assertEqual(result['grid_id'].tolist(), [0])
        self.assertEqual(result['target_grid_id'].tolist(), [5])
        pd.testing.assert_frame_equal(result.drop(columns=['grid_id', 'target_grid_id']),
                                      drivers.drop(columns=['grid_id', 'target_grid_id']))
        self.assertEqual(drivers['grid_id'].tolist(), [30])


class GridInputTests(unittest.TestCase):
    @staticmethod
    def prepare(directory, grid_num=6):
        root = Path(directory)
        manifest = {'schema_version': 'hex_grid_inputs_v1', 'grid_num': grid_num,
                    'dates': ['2015-05-18'], 'grid_geometry_sha256': 'test_geometry',
                    'grid_shape': 'hexagonal_clipped_to_original_region', 'radius_km': 4.0}
        (root / 'grid_manifest.json').write_text(json.dumps(manifest))
        with (root / f'orders_grid{grid_num}_2015-05-18.pkl').open('wb') as file:
            pickle.dump({0: [], 86399: []}, file)
        with (root / 'drivers.pickle').open('wb') as file:
            pickle.dump(pd.DataFrame({'driver_id': [1], 'lng': [-74.0], 'lat': [40.75],
                                      'start_time': [18000], 'end_time': [36000], 'grid_id': [0]}), file)
        return root

    def test_legacy_paths_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            orders, drivers = input_data_paths('2015-05-18', directory)
            self.assertEqual(orders.name, 'orders_grid35_2015-05-18.pkl')
            self.assertEqual(drivers.name, 'drivers_grid35_1000.pickle')

    def test_prepared_inputs_validate_grid_count_and_date(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.prepare(directory)
            check_collection_inputs(['2015-05-18'], 6, root)
            data = SimulatorPattern('2015-05-18', root)
            self.assertEqual(len(data.driver_info), 1)
            with self.assertRaises(ValueError):
                check_collection_inputs(['2015-05-18'], 263, root)
            with self.assertRaises(FileNotFoundError):
                check_collection_inputs(['2015-05-19'], 6, root)

    def test_collector_supports_six_grids_and_generic_reposition_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.prepare(directory)
            adjacency = np.ones((6, 6), dtype=int) - np.eye(6, dtype=int)
            pd.DataFrame(adjacency).to_csv(root / 'grid_adjacency.csv', index=False)
            pd.DataFrame({'center_lon': [-74.0] * 6, 'center_lat': [40.75] * 6}).to_csv(
                root / 'grid_info.csv', index=False)
            with patch.object(SimulatorPattern, 'load_mapping') as mapping:
                result = run_trajectory_collection(
                    '2015-05-18', grid_num=6, driver_num=1, simulation_steps=2,
                    decision_freq=1, price_action_values=[1.0], matching_radius_values=[1.0],
                    data_dir=root, output_dir=root / 'output', enable_reposition=True)
                mapping.assert_not_called()
            self.assertEqual(len(result['minute_metrics']), 12)
            self.assertEqual(result['manifest']['grid_geometry_sha256'], 'test_geometry')
            self.assertEqual(result['minute_metrics']['idle_driver_num'].sum(), 2)


if __name__ == '__main__':
    unittest.main()
