"""从零训练协议的种子隔离、校准来源和冻结文件约束。"""

import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from constrained_rl import scratch_protocol
from offline_rl.data import sha256


class ScratchProtocolTests(unittest.TestCase):
    def setUp(self):
        self.previous = Path.cwd()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        os.chdir(self.temporary.name)
        self.addCleanup(os.chdir, self.previous)
        root = scratch_protocol.REFERENCE.parent
        root.mkdir(parents=True)
        contents = {
            "checkpoints/ar_ppo.pt": b"reference checkpoint, never deserialize",
            "models/old/progress.json": b'{"status":"paused"}',
            "models/old/latest.pt": b"old weights",
            str(root / "progress.json"): b'{"status":"complete"}',
            str(root / "revenue_control/latest.pt"): b"pilot control weights",
            str(root / "constrained/latest.pt"): b"pilot constrained weights",
            "my_data/orders.pkl": b"orders",
            "online_rl/policy.py": b"# frozen original\n",
            "constrained_rl/environment.py": b"# new metrics\n",
            "constrained_rl/scratch_protocol.py": b"# scratch protocol\n",
            "constrained_rl/nested/worker.py": b"# nested source\n",
        }
        for name, content in contents.items():
            path = Path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        with (root / "evaluation.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["phase", "policy", "cohort_demand",
                "unique_matched", "queue_wait_seconds", "platform_revenue"])
            writer.writeheader()
            writer.writerows([
                dict(phase="calibration", policy="fixed_f3", cohort_demand=100,
                     unique_matched=60, queue_wait_seconds=6000, platform_revenue=1000),
                dict(phase="calibration", policy="fixed_f3", cohort_demand=300,
                     unique_matched=240, queue_wait_seconds=18000, platform_revenue=2000),
                dict(phase="test", policy="fixed_f3", cohort_demand=100,
                     unique_matched=99, queue_wait_seconds=10, platform_revenue=9999)])
        self.reference = {"config": {"grid_num": 6}, "observation_metadata": {"state_size": 660},
            "model": {"grid_count": 6}, "seeds": {"calibration": [3700000011, 3700000022]},
            "algorithm": {"learning_rate": 1e-4, "reward_scale": 1500., "gamma": 1.,
                "cost_warmup_epochs": 50, "cost_warmup_learning_rate": .001},
            "constraints": {"service_floor": .75, "wait_limit_seconds_per_demand": 60.,
                "demand_scale": 200., "wait_scale_seconds": 100.},
            "initial_checkpoint": "checkpoints/ar_ppo.pt",
            "initial_sha256": sha256("checkpoints/ar_ppo.pt"),
            "data_sha256": {"my_data/orders.pkl": sha256("my_data/orders.pkl")},
            "protected_files_sha256": {name: sha256(name)
                for name in ("models/old/progress.json", "models/old/latest.pt")},
            "constraint_rule": "total counts", "cost_rule": "additive costs",
            "coverage_rule": "unique parents", "execution": "ancestral sampling", "sources": []}
        scratch_protocol.REFERENCE.write_text(json.dumps(self.reference))
        source_patch = patch.object(scratch_protocol, "source_hashes",
            side_effect=lambda: {"online_rl/policy.py": sha256("online_rl/policy.py")})
        source_patch.start()
        self.addCleanup(source_patch.stop)

    def test_calibration_reuse_and_random_initialization_contract(self):
        with patch("torch.load", side_effect=AssertionError("must not deserialize weights")):
            plan = scratch_protocol.make_protocol("main")
            loaded = scratch_protocol.load_protocol("main/protocol.json")
        self.assertEqual(loaded, plan)
        for key in ("config", "model", "observation_metadata", "constraints"):
            self.assertEqual(plan[key], self.reference[key])
        self.assertNotIn("initial_checkpoint", plan)
        self.assertEqual(plan["algorithm"]["learning_rate"], 3e-4)
        self.assertEqual(plan["algorithm"]["reward_scale"], 1500.)
        self.assertFalse(any(key.startswith("cost_warmup") for key in plan["algorithm"]))
        provenance = plan["baseline_calibration_provenance"]
        self.assertEqual((provenance["cohort_demand"], provenance["unique_matched"]), (400, 300))
        self.assertEqual(provenance["queue_wait_seconds"], 24000)
        self.assertEqual(provenance["episodes"], 2)
        self.assertEqual(plan["training_seed"], 412)
        self.assertEqual(plan["methods"], ["revenue_control", "constrained"])
        self.assertEqual(set(plan["expected_test_policies"]),
                         {"fixed_f3", "frozen_ar_ppo", "revenue_control", "constrained"})
        self.assertEqual(plan["training_episodes"], 100000)
        self.assertEqual(plan["episodes_per_update"], 16)
        self.assertEqual(plan["validation_interval_updates"], 64)
        self.assertEqual(plan["early_validation_updates"], [1, 4, 16])
        self.assertEqual(len(plan["seeds"]["validation"]), 32)
        self.assertEqual(len(plan["seeds"]["test"]), 100)

    def test_seed_namespaces_deterministic_and_disjoint_from_smoke(self):
        main = scratch_protocol.make_protocol("main", episodes=80)
        again = scratch_protocol.make_protocol("again", episodes=80)
        smoke = scratch_protocol.make_protocol("smoke", episodes=32, training_seed=413, smoke=True)
        self.assertEqual(main["seeds"], again["seeds"])
        all_seeds = []
        for plan, offset in ((main, 0), (smoke, 60_000_000)):
            for name, low in (("train", 3_800_000_000), ("validation", 3_820_000_000),
                              ("test", 3_840_000_000)):
                values = plan["seeds"][name]
                self.assertTrue(all(low + offset <= seed < low + offset + 10_000_000 for seed in values))
                all_seeds.extend(values)
        self.assertEqual(len(all_seeds), len(set(all_seeds)))
        self.assertEqual(smoke["stage"], "smoke")
        self.assertEqual(smoke["training_seed"], 413)
        self.assertEqual(smoke["validation_interval_updates"], 1)
        self.assertEqual(len(smoke["seeds"]["validation"]), 2)
        self.assertEqual(len(smoke["seeds"]["test"]), 2)

    def test_source_snapshot_covers_nested_new_code_and_detects_changes(self):
        plan = scratch_protocol.make_protocol("run", episodes=16)
        self.assertIn("constrained_rl/nested/worker.py", plan["source_sha256"])
        for name, digest in plan["source_sha256"].items():
            self.assertEqual(sha256(Path("run/source_snapshot") / name), digest)
        target = Path("run/source_snapshot/constrained_rl/nested/worker.py")
        target.write_text("# changed snapshot\n")
        with self.assertRaises(ValueError):
            scratch_protocol.load_protocol("run/protocol.json")

    def test_loader_detects_source_data_reference_and_old_artifact_mutations(self):
        scratch_protocol.make_protocol("run", episodes=16)
        for name in ("online_rl/policy.py", "constrained_rl/environment.py", "my_data/orders.pkl",
                     str(scratch_protocol.REFERENCE), "checkpoints/ar_ppo.pt",
                     "models/old/progress.json", "models/old/latest.pt"):
            with self.subTest(path=name):
                path = Path(name)
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                with self.assertRaises(ValueError):
                    scratch_protocol.load_protocol("run/protocol.json")
                path.write_bytes(original)
        scratch_protocol.load_protocol("run/protocol.json")

    def test_rejects_inconsistent_calibration_and_overwrite(self):
        self.reference["constraints"]["service_floor"] = .70  # Mean of per-episode ratios.
        scratch_protocol.REFERENCE.write_text(json.dumps(self.reference))
        with self.assertRaises(ValueError):
            scratch_protocol.make_protocol("invalid", episodes=16)
        self.assertFalse(Path("invalid").exists())
        self.reference["constraints"]["service_floor"] = .75
        scratch_protocol.REFERENCE.write_text(json.dumps(self.reference))
        scratch_protocol.make_protocol("run", episodes=16)
        with self.assertRaises(FileExistsError):
            scratch_protocol.make_protocol("run", episodes=16)


if __name__ == "__main__":
    unittest.main()
