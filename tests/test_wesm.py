import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise unittest.SkipTest("需要 baseline/requirements.txt") from error

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from baseline.losses import WESMLoss
from baseline.train_lstm import _run_epoch


class WESMLossTests(unittest.TestCase):
    def test_six_task_weights_follow_paper_formula(self):
        criterion = WESMLoss(6, gamma=0.5, history_steps=2, update_interval=1)
        old = torch.tensor([1., 2., 3., 4., 5., 6.])
        recent = torch.tensor([6., 5., 4., 3., 2., 1.])
        current = torch.tensor([1., 1., 1., 1., 1., 9.], requires_grad=True)
        criterion(old)
        criterion(recent)
        loss = criterion(current)
        smoothed = (current.detach() + 0.5 * recent + 0.25 * old) / 1.75
        expected_weights = smoothed / smoothed.sum()
        torch.testing.assert_close(criterion.weights, expected_weights)
        torch.testing.assert_close(loss, (expected_weights * current).sum())
        self.assertEqual(criterion.weights.argmax().item(), 5)
        loss.backward()
        # 权重是调度系数，不对权重公式自身反向传播。
        torch.testing.assert_close(current.grad, expected_weights)
        self.assertFalse(criterion.loss_history.requires_grad)

    def test_old_losses_expire_and_recent_loss_has_larger_influence(self):
        criterion = WESMLoss(2, gamma=0.5, history_steps=1, update_interval=1)
        criterion(torch.tensor([1000., 0.]))
        criterion(torch.tensor([8., 0.]))
        criterion(torch.tensor([0., 8.]))
        torch.testing.assert_close(criterion.weights, torch.tensor([1 / 3, 2 / 3]))

    def test_update_interval_warmup_and_zero_loss_fallback(self):
        criterion = WESMLoss(6, history_steps=0, update_interval=2)
        hard = torch.tensor([1., 1., 1., 1., 1., 5.])
        criterion(hard)
        torch.testing.assert_close(criterion.weights, torch.full((6,), 1 / 6))
        criterion(hard)
        torch.testing.assert_close(criterion.weights, hard / hard.sum())
        weights = criterion.weights.clone()
        criterion(torch.zeros(6))
        torch.testing.assert_close(criterion.weights, weights)
        self.assertEqual(criterion(torch.zeros(6)).item(), 0)
        torch.testing.assert_close(criterion.weights, torch.full((6,), 1 / 6))

    def test_state_roundtrip_and_eval_do_not_update_history(self):
        criterion = WESMLoss(6, update_interval=2)
        criterion(torch.arange(1., 7.))
        restored = WESMLoss(6, update_interval=2)
        restored.load_state_dict(criterion.state_dict())
        next_loss = torch.arange(6., 0., -1.)
        torch.testing.assert_close(criterion(next_loss), restored(next_loss))
        restored.eval()
        before = {key: value.clone() for key, value in restored.state_dict().items()}
        restored(next_loss * 1000)
        for key, value in restored.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_validation_uses_equal_weights_without_changing_strategy(self):
        class ZeroModel(nn.Module):
            def forward(self, history, context, grids):
                return torch.zeros(len(grids), 6)

        criterion = WESMLoss(6, update_interval=1)
        criterion(torch.tensor([1., 1., 1., 1., 1., 10.]))
        before = {key: value.clone() for key, value in criterion.state_dict().items()}
        targets = torch.arange(18.).reshape(3, 6)
        loader = DataLoader(TensorDataset(
            torch.zeros(3, 2, 1), torch.zeros(3, 4), torch.zeros(3, dtype=torch.int64),
            targets), batch_size=2)
        stats = _run_epoch(ZeroModel(), loader, "cpu", criterion=criterion)
        self.assertAlmostEqual(stats["mse"], targets.square().mean().item(), delta=1e-5)
        self.assertEqual(stats["mse"], stats["weighted_loss"])
        torch.testing.assert_close(torch.tensor(stats["mean_weights"]), torch.full((6,), 1 / 6))
        for key, value in criterion.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_rejects_invalid_configuration_and_losses(self):
        for kwargs in ({"task_count": 0}, {"task_count": 6, "gamma": 1.0},
                       {"task_count": 6, "gamma": float("nan")},
                       {"task_count": 6, "history_steps": -1},
                       {"task_count": 6, "update_interval": 0}):
            with self.assertRaises(ValueError):
                WESMLoss(**kwargs)
        criterion = WESMLoss(6)
        for values in (torch.zeros(4), torch.full((6,), -1.), torch.full((6,), float("nan"))):
            with self.assertRaises(ValueError):
                criterion(values)


if __name__ == "__main__":
    unittest.main()
