"""DL-MRD 的时间维注意力、样本独立性、上下文隔离与六输出检查。"""

from copy import deepcopy
import tempfile
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch

import torch

from baseline.dlmrd.model import DLMRDPredictor, GridDLMRD


class DLMRDModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(61)
        self.config = dict(history_size=24, grid_count=6, history_steps=4,
                           hidden_size=16, num_heads=4, num_layers=3, d_ff=32, context_size=28)
        self.model = GridDLMRD(**self.config).eval()
        self.history = torch.randn(5, 4, 24)
        self.context = torch.randn(5, 28)
        self.grids = torch.tensor([0, 1, 2, 3, 4])

    def test_six_independent_heads_receive_history_context_and_grid_gradients(self):
        history = self.history.clone().requires_grad_()
        context = self.context.clone().requires_grad_()
        output = self.model(history, context, self.grids)
        self.assertEqual(output.shape, (5, 6))
        output.square().mean().backward()
        for values in (history.grad, context.grad, self.model.history_embedding.weight.grad,
                       self.model.context_weight.grad, self.model.context_bias.grad,
                       self.model.grid_embedding.weight.grad):
            self.assertTrue(torch.isfinite(values).all())
            self.assertGreater(float(values.abs().sum()), 0.)
        self.assertTrue((context.grad.abs().sum(0) > 0).all())
        for head in self.model.output_heads:
            self.assertGreater(float(head.weight.grad.abs().sum()), 0.)
            self.assertGreater(float(head.bias.grad.abs().sum()), 0.)
        self.assertEqual(len({id(head.weight) for head in self.model.output_heads}), 6)

    def test_prediction_is_independent_of_batch_companions_and_permutations(self):
        with torch.inference_mode():
            together = self.model(self.history, self.context, self.grids)
            alone = torch.cat([self.model(self.history[i:i+1], self.context[i:i+1], self.grids[i:i+1])
                               for i in range(len(self.history))])
            order = torch.tensor([3, 0, 4, 1, 2])
            permuted = self.model(self.history[order], self.context[order], self.grids[order])
            perturbed = self.history.clone()
            perturbed[1:] += 100.
            companions_changed = self.model(perturbed, self.context, self.grids)
        torch.testing.assert_close(together, alone, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(permuted, together[order], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(companions_changed[:1], together[:1], rtol=1e-5, atol=1e-6)

    def test_history_order_changes_prediction(self):
        with torch.no_grad():
            forward = self.model(self.history, self.context, self.grids)
            reversed_order = self.model(self.history.flip(1), self.context, self.grids)
        self.assertGreater(float((forward - reversed_order).abs().max()), 1e-4)

    def test_future_history_cannot_change_earlier_encoded_positions(self):
        changed = self.history.clone()
        changed[:, 2:] += torch.randn_like(changed[:, 2:]) * 10
        with torch.no_grad():
            before = self.model.encode_history(self.history)
            after = self.model.encode_history(changed)
        torch.testing.assert_close(before[:, :2], after[:, :2], rtol=0, atol=0)
        self.assertGreater(float((before[:, 2:] - after[:, 2:]).abs().max()), 1e-4)
        history = self.history.clone().requires_grad_()
        self.model.encode_history(history)[:, :2].square().sum().backward()
        torch.testing.assert_close(history.grad[:, 2:], torch.zeros_like(history.grad[:, 2:]), rtol=0, atol=0)

    def test_context_and_grid_are_joined_after_temporal_attention(self):
        encoded = []
        handle = self.model.history_encoder.register_forward_hook(
            lambda module, inputs, output: encoded.append(output.detach().clone()))
        try:
            with torch.no_grad():
                first = self.model(self.history, self.context, self.grids)
                other = self.model(self.history, self.context + 2, self.grids.roll(1))
        finally:
            handle.remove()
        torch.testing.assert_close(encoded[0], encoded[1], rtol=0, atol=0)
        self.assertGreater(float((first - other).abs().max()), 1e-4)

    def test_outputs_can_exceed_minmax_training_range(self):
        with torch.no_grad():
            for index, head in enumerate(self.model.output_heads):
                head.weight.zero_()
                head.bias.fill_(-2. if index % 2 else 3.)
            output = self.model(self.history, self.context, self.grids)
        torch.testing.assert_close(output, torch.tensor([3., -2., 3., -2., 3., -2.]).expand(5, -1))

    def test_model_and_predictor_checkpoint_roundtrip_use_minmax_loader(self):
        module = types.ModuleType("baseline.dlmrd.data")
        preprocessor = object()
        module.MinMaxPreprocessor = Mock()
        module.MinMaxPreprocessor.from_state_dict.return_value = preprocessor
        checkpoint = {"schema_version": "grid_dlmrd_v1", "model_config": self.config,
                      "model_state_dict": deepcopy(self.model.state_dict()),
                      "preprocessor": {"normalization": "minmax"}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            torch.save(checkpoint, path)
            with patch.dict("sys.modules", {"baseline.dlmrd.data": module}):
                predictor = DLMRDPredictor.load(path)
                self.assertIs(predictor.preprocessor, preprocessor)
                module.MinMaxPreprocessor.from_state_dict.assert_called_once_with(checkpoint["preprocessor"])
                with torch.no_grad():
                    expected = self.model(self.history, self.context, self.grids)
                    actual = predictor.model(self.history, self.context, self.grids)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                checkpoint["schema_version"] = "grid_lstm_v1"
                torch.save(checkpoint, path)
                with self.assertRaises(ValueError):
                    DLMRDPredictor.load(path)

    def test_dimensions_and_input_shapes_are_checked(self):
        for key, value in (("history_size", 0), ("grid_count", 0), ("history_steps", 0),
                           ("context_size", 0), ("hidden_size", 15), ("num_heads", 0),
                           ("num_layers", 0), ("d_ff", False)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                GridDLMRD(**dict(self.config, **{key: value}))
        for history, context, grids in (
                (self.history[:, :3], self.context, self.grids),
                (self.history, self.context[:, :27], self.grids),
                (self.history, self.context, self.grids[:, None])):
            with self.assertRaises(ValueError):
                self.model(history, context, grids)


if __name__ == "__main__":
    unittest.main()
