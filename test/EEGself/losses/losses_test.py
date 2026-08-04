import os
import sys
import types
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EEGself._testtools import get_device, make_grid

from selfeeg import losses

N, FEAT = 64, 128


def _phase_signal(phase=0.0, length=FEAT, rows=N):
    """Deterministic (seed-free) signal used for the numeric regression checks."""
    base = torch.sin(torch.linspace(0, 8 * torch.pi, length) + phase)
    return base + (torch.arange(rows) / rows).unsqueeze(1)


class TestLoss(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = get_device(
            probe=lambda dev: losses.barlow_loss(
                torch.randn(N, FEAT, device=dev), torch.randn(N, FEAT, device=dev)
            )
        )

    def _tensor_sets(self):
        """Yield a namespace of random tensors per device to be tested."""
        torch.manual_seed(1234)
        devices = ["cpu"] + ([self.device.type] if self.device.type != "cpu" else [])
        for dev in devices:
            yield types.SimpleNamespace(
                x=torch.randn(N, FEAT, device=dev),
                y=torch.randn(N, FEAT, device=dev),
                p=torch.randn(N, FEAT, device=dev),
                z=torch.randn(N, FEAT, device=dev),
                u=torch.randn(FEAT, 1024, device=dev),
            )

    def _assert_no_nan(self, grid, loss_fn):
        for args in grid:
            with self.subTest(**{k: v for k, v in args.items() if not torch.is_tensor(v)}):
                self.assertEqual(torch.isnan(loss_fn(**args)).sum().item(), 0)

    def test_barlow_loss(self):
        for t in self._tensor_sets():
            grid = make_grid(
                {"z1": [t.x], "z2": [t.y, None], "lambda_coeff": [0.005, 0.05, 0.5, 1]}
            )
            self._assert_no_nan(grid, losses.barlow_loss)
        loss = losses.barlow_loss(_phase_signal())
        self.assertAlmostEqual(loss.item(), 76.4043, delta=1e-3)

    def test_byol_loss(self):
        for t in self._tensor_sets():
            grid = make_grid(
                {"z1": [t.x], "z2": [t.y], "p1": [t.p], "p2": [t.z], "projections_norm": [True, False]}
            )
            self._assert_no_nan(grid, losses.byol_loss)
        loss = losses.byol_loss(
            _phase_signal(torch.pi / 6),
            _phase_signal(torch.pi / 4),
            _phase_signal(torch.pi / 3),
            _phase_signal(torch.pi / 5),
        )
        self.assertAlmostEqual(loss.item(), 0.0534, delta=1e-3)

    def test_simclr_loss(self):
        for t in self._tensor_sets():
            grid = make_grid(
                {"projections": [t.x], "temperature": [0.15, 0.5, 0.7], "projections_norm": [True, False]}
            )
            self._assert_no_nan(grid, losses.simclr_loss)
        loss = losses.simclr_loss(_phase_signal(torch.pi / 6))
        self.assertAlmostEqual(loss.item(), 9.04887, delta=1e-3)

    def test_simsiam_loss(self):
        for t in self._tensor_sets():
            grid = make_grid(
                {"z1": [t.x], "z2": [t.y], "p1": [t.p], "p2": [t.z], "projections_norm": [True, False]}
            )
            self._assert_no_nan(grid, losses.simsiam_loss)
        loss = losses.simsiam_loss(
            _phase_signal(torch.pi / 6),
            _phase_signal(torch.pi / 4),
            _phase_signal(torch.pi / 3),
            _phase_signal(torch.pi / 5),
        )
        self.assertAlmostEqual(loss.item(), -0.9867, delta=1e-3)

    def test_vicreg_loss(self):
        for t in self._tensor_sets():
            grid = make_grid(
                {"z1": [t.x], "z2": [t.y, None], "Lambda": [25, 10, 50], "Mu": [25, 5, 50], "Nu": [2, 1, 0.5]},
                max_comb=24,
            )
            self._assert_no_nan(grid, losses.vicreg_loss)
        loss = losses.vicreg_loss(_phase_signal())
        self.assertAlmostEqual(loss.item(), 21.4443, delta=1e-3)

    def test_moco_loss(self):
        for t in self._tensor_sets():
            grid = make_grid(
                {
                    "q": [t.x],
                    "k": [t.y],
                    "queue": [None, t.u],
                    "projections_norm": [True, False],
                    "temperature": [0.15, 0.5, 0.9],
                }
            )
            self._assert_no_nan(grid, losses.moco_loss)
        x = _phase_signal(torch.pi / 6)
        y = _phase_signal(torch.pi / 4)
        u = _phase_signal(torch.pi / 3, length=1024, rows=FEAT)
        self.assertAlmostEqual(losses.moco_loss(x, y, u).item(), 108.6667, delta=1e-3)
        self.assertAlmostEqual(losses.moco_loss(x, y).item(), 0.4952, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
