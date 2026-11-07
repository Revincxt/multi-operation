"""网格对比图只读取已有边界，检查覆盖范围、坐标变换和内部空隙。"""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


HAS_GEOMETRY = (importlib.util.find_spec("shapely") is not None
                and importlib.util.find_spec("matplotlib") is not None)
if HAS_GEOMETRY:
    from shapely import transform
    from shapely.geometry import Polygon, mapping
    from data_pipeline.regrid_hex import LocalProjection, hexagon, load_region
    from diagnostics.plot_grid_partition import load_comparison, polygon_path, plot_grid_partition


@unittest.skipUnless(HAS_GEOMETRY, "Install requirements-grid.txt for spatial figures")
class GridPartitionFigureTests(unittest.TestCase):
    def _write_layout(self, root):
        projection = LocalProjection(-74.0, 40.75)
        rows = []
        for grid_id, center in enumerate(((0, 0), (np.sqrt(3), 0))):
            polygon = hexagon(np.asarray(center), 1.0)
            coordinates = projection.inverse(np.asarray(polygon.exterior.coords)[:6])
            lon, lat = projection.inverse(center)
            row = {"grid_id": grid_id, "center_lon": lon, "center_lat": lat}
            for vertex, (x, y) in enumerate(coordinates, start=1):
                row.update({f"vertex_{vertex}_lon": x, f"vertex_{vertex}_lat": y})
            rows.append(row)
        source = root / "original.csv"
        pd.DataFrame(rows).to_csv(source, index=False)
        region, cells, projection = load_region(source)
        for name, polygons in (("grid_regions", cells), ("hex_lattice", cells), ("study_region", [region])):
            features = [{"type": "Feature", "properties": {"grid_id": index},
                         "geometry": mapping(transform(polygon, projection.inverse))}
                        for index, polygon in enumerate(polygons)]
            (root / f"{name}.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))
        manifest = {"source_grid_num": 2, "grid_num": 2, "radius_km": 1.0,
                    "projection": {"method": "local_equirectangular_km",
                                   "longitude": projection.longitude, "latitude": projection.latitude}}
        (root / "grid_manifest.json").write_text(json.dumps(manifest))
        return source

    def test_saved_geometries_and_vector_exports_preserve_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._write_layout(root)
            inputs = {path: path.read_bytes() for path in root.iterdir()}
            result = plot_grid_partition(source, root)
            self.assertEqual(result["grid_ids"], [0, 1])
            self.assertLess(result["symmetric_difference_km2"], 1e-7)
            output = root / "paper_figures"
            self.assertTrue((output / "fig_grid_partition_comparison.pdf").read_bytes().startswith(b"%PDF-"))
            self.assertIn("<svg", (output / "fig_grid_partition_comparison.svg").read_text())
            self.assertTrue((output / "fig_grid_partition_comparison.png").read_bytes().startswith(b"\x89PNG"))
            for path, original in inputs.items():
                self.assertEqual(path.read_bytes(), original)

    def test_missing_region_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._write_layout(root)
            path = root / "grid_regions.geojson"
            data = json.loads(path.read_text())
            data["features"].pop()
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_comparison(source, root)

    def test_holes_are_preserved_with_opposite_winding(self):
        polygon = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)],
                          holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]])
        path = polygon_path(polygon)
        rings = path.to_polygons()
        self.assertEqual(len(rings), 2)
        areas = [0.5 * np.sum(ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1])
                 for ring in rings]
        self.assertGreater(areas[0], 0)
        self.assertLess(areas[1], 0)
        self.assertAlmostEqual(sum(areas), polygon.area)


if __name__ == "__main__":
    unittest.main()
