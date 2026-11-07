"""当前入口路径、废弃参数文档与采集兼容修订的回归检查。"""

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from baseline import compare_models, compare_policies
from baseline.data import DEFAULT_INPUT
from data_pipeline.collect_density import config_signature


class ProjectMaintenanceTests(unittest.TestCase):
    def test_training_default_uses_current_snapshot(self):
        self.assertEqual(DEFAULT_INPUT, "datasets_grid6_low_mixed47k_20261029")

    def test_comparison_clis_require_explicit_existing_artifact_selection(self):
        for module, function in ((compare_models, "compare_models"),
                                 (compare_policies, "compare_policies")):
            with self.subTest(module=module.__name__), patch("sys.argv", [module.__name__]), \
                    patch.object(module, function) as run, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    module.main()
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()

    def test_missing_checkpoint_rejected_before_loading_large_dataset(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(compare_models, "load_dataset") as load:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                compare_models.compare_models(root / "data", root / "lstm", root / "timesnet",
                                              root / "output")
            load.assert_not_called()
            self.assertFalse((root / "output").exists())

    def test_revision_is_exact_and_does_not_allow_configuration_changes(self):
        source = {"scenario": "low", "run_count": 10, "seed_start": 42, "batch_size": 1000,
                  "data_dir": "input", "date": "2015-05-18", "start_time": "05:00",
                  "simulation_steps": 300, "code_sha256": {"src/simulator.py": "before"}}
        current = {**source, "code_sha256": {"src/simulator.py": "after"}}
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(compare_policies, "make_config", return_value=current):
            root = Path(temporary)
            with self.assertRaises(ValueError):
                compare_policies.check_collection_revision(root, source)
            revision = {"collection_config_sha256": config_signature(source),
                        "code_sha256": current["code_sha256"], "validation": {"passed": True}}
            (root / "execution_revision.json").write_text(json.dumps(revision))
            compare_policies.check_collection_revision(root, source)
            current["simulation_steps"] = 301
            with self.assertRaises(ValueError):
                compare_policies.check_collection_revision(root, source)
            current["simulation_steps"] = 300
            current["code_sha256"] = {"src/simulator.py": "unverified"}
            with self.assertRaises(ValueError):
                compare_policies.check_collection_revision(root, source)

    def test_canonical_documentation_example_uses_current_actions(self):
        root = Path(__file__).resolve().parents[1]
        document = (root / "docs/price_response_models.md").read_text()
        example, = re.findall(r"```python\n(.*?)```", document, flags=re.S)
        namespace = {}
        exec(compile(example, "price_response_models.md", "exec"), namespace)
        actions = namespace["simulator"].current_grid_actions()
        self.assertEqual(len(actions), 6)
        self.assertTrue(actions["price_multiplier"].eq(1.5).all())
        self.assertTrue(actions["matching_radius_km"].eq(3.0).all())


if __name__ == "__main__":
    unittest.main()
