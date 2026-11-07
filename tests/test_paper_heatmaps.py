"""论文版热力图保留配对扫描口径、共享色标和未定义值。"""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from diagnostics.plot_density_heatmaps import prepare_metrics
from tests.test_density_heatmaps import example_data


HAS_PLOTTING = importlib.util.find_spec("matplotlib") is not None
if HAS_PLOTTING:
    from diagnostics.plot_paper_heatmaps import draw_paper_heatmap, plot_paper_heatmaps
    import matplotlib.pyplot as plt


@unittest.skipUnless(HAS_PLOTTING, "matplotlib not installed")
class PaperHeatmapTests(unittest.TestCase):
    def test_axes_reference_units_and_preference_directions(self):
        frame, config = example_data()
        data = prepare_metrics(frame, config).query("scenario == 'low'")
        fig, axes = plt.subplots(1, 3)
        try:
            request = draw_paper_heatmap(fig, axes[0], data, "request_rate", config["prices"],
                                         config["radii"], (0, 1.2), 1)
            wait = draw_paper_heatmap(fig, axes[1], data, "mean_wait_to_match_seconds", config["prices"],
                                      config["radii"], (60, 180), 4)
            demand = draw_paper_heatmap(fig, axes[2], data, "baseline_request_num", config["prices"],
                                        config["radii"], (0, 100), 0)
            np.testing.assert_allclose(request.get_array(), 1.2)
            self.assertEqual(request.norm.vmax, 1.2)
            self.assertEqual(len(axes[0].patches), 1)
            self.assertEqual(wait.cmap.name, "viridis_r")
            self.assertEqual(demand.cmap.name, "viridis")
            self.assertIn(r"\uparrow", axes[0].get_title(loc="left"))
            self.assertIn(r"\downarrow", axes[1].get_title(loc="left"))
            self.assertIn("=", axes[2].get_title(loc="left"))
            self.assertFalse(request.get_rasterized())
            self.assertEqual([label.get_text() for label in axes[1].get_xticklabels()], ["2", "3"])
            self.assertEqual(len(axes[2].texts), len(config["prices"]) * len(config["radii"]))
            self.assertTrue(all(text.get_text() == "100" for text in axes[2].texts))
            self.assertTrue(all(text.get_bbox_patch() is None for text in axes[2].texts))
        finally:
            plt.close(fig)

    def test_bubble_counts_are_annotated_at_the_correct_cells(self):
        frame, config = example_data()
        data = prepare_metrics(frame, config).query("scenario == 'low'").copy()
        # 直接检查绘图，不依赖固定需求的校验假设，缺失值不能标成零。
        data.loc[data.metric == "baseline_request_num", "value"] = [4974, 9964, 24903, np.nan]
        fig, ax = plt.subplots()
        try:
            im = draw_paper_heatmap(fig, ax, data, "baseline_request_num", config["prices"],
                                    config["radii"], (0, 24903), 0)
            self.assertEqual({text.get_position(): text.get_text() for text in ax.texts},
                             {(0, 0): "4974", (1, 0): "9964", (0, 1): "24903", (1, 1): "—"})
            self.assertEqual([text.get_color() for text in ax.texts],
                             ["white", "white", "#222222", "#444444"])
            self.assertTrue(np.ma.getmaskarray(im.get_array())[1, 1])
            self.assertEqual(im.get_clim(), (0, 24903))
        finally:
            plt.close(fig)

    def test_missing_wait_is_masked_not_replaced_by_zero(self):
        frame, config = example_data()
        frame["matched_request_num"] = 0
        frame["order_completion_rate"] = 0.0
        frame["mean_wait_to_match_seconds"] = np.nan
        data = prepare_metrics(frame, config).query("scenario == 'low'")
        fig, ax = plt.subplots()
        try:
            im = draw_paper_heatmap(fig, ax, data, "mean_wait_to_match_seconds", config["prices"],
                                    config["radii"], (0, 1), 4)
            self.assertTrue(np.ma.getmaskarray(im.get_array()).all())
        finally:
            plt.close(fig)

    def test_five_digit_counts_fit_inside_the_paper_layout(self):
        from diagnostics.plot_paper_heatmaps import _font_family

        prices = [0.4, 0.7, 1, 1.3, 1.5, 1.6, 1.9, 2.2, 2.5, 2.8, 3.1, 3.4, 3.7, 4]
        radii = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
        data = pd.DataFrame([
            {"metric": "baseline_request_num", "price_multiplier": price,
             "matching_radius_km": radius, "value": 24903}
            for price in prices for radius in radii])
        with plt.rc_context({"font.family": _font_family()}):
            fig, axes = plt.subplots(2, 3, figsize=(7.16, 6.1), dpi=600)
            try:
                fig.subplots_adjust(left=0.072, right=0.92, bottom=0.145, top=0.865,
                                    hspace=0.26, wspace=0.50)
                ax = axes[0, 0]
                draw_paper_heatmap(fig, ax, data, "baseline_request_num", prices, radii, (0, 24903), 0)
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                for text in ax.texts:
                    column, row = text.get_position()
                    left, bottom = ax.transData.transform((column - 0.5, row - 0.5))
                    right, top = ax.transData.transform((column + 0.5, row + 0.5))
                    extent = text.get_window_extent(renderer)
                    self.assertLess(extent.width, (right - left) * 0.86)
                    self.assertGreater(extent.x0, left)
                    self.assertLess(extent.x1, right)
                    self.assertGreater(extent.y0, bottom)
                    self.assertLess(extent.y1, top)
            finally:
                plt.close(fig)

    def test_exports_shared_scale_figures_without_modifying_source(self):
        frame, config = example_data()
        config.update(date="2015-05-18", start_time="05:00", simulation_steps=300)
        report = {"status": "complete", "config": config, "scenario_settings": {
            scenario: {"driver_num": count, "actual_sampled_orders": 101}
            for scenario, count in (("low", 250), ("high", 1000))}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame.to_csv(root / "screening_runs.csv", index=False)
            (root / "manifest.json").write_text(json.dumps(report))
            before = {path: path.read_bytes() for path in root.iterdir()}
            result = plot_paper_heatmaps(root)
            output = root / "six_metric_heatmaps" / "paper_figures"
            self.assertEqual(result["phase"], "screening")
            self.assertEqual(result["color_limits_shared_across_scenarios"]["request_rate"], (0, 1.2))
            self.assertEqual(result["interpolation"], "none")
            self.assertTrue((output / "all_scenarios_six_metrics.pdf").read_bytes().startswith(b"%PDF-"))
            for scenario in config["scenarios"]:
                pdf = (output / f"{scenario}_six_metrics.pdf").read_bytes()
                self.assertTrue(pdf.startswith(b"%PDF-"))
                self.assertFalse(b"/Subtype /Image" in pdf)
                self.assertIn("<svg", (output / f"{scenario}_six_metrics.svg").read_text())
                self.assertTrue((output / f"{scenario}_six_metrics.png").read_bytes().startswith(b"\x89PNG"))
            expected = prepare_metrics(frame, config)
            pd.testing.assert_frame_equal(pd.read_csv(output / "heatmap_values.csv"), expected)
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
