"""Portable sensor/command contracts for the three released checkpoints."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from kaleid import Kaleid, available_systems, system_info
from kaleid.predictor import _precision


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.models = {s: Kaleid(s, device="cpu") for s in available_systems()}

    def test_one_joint_forward_per_packet(self):
        for name, model in self.models.items():
            with self.subTest(system=name):
                calls = []
                hook = model._model.register_forward_hook(lambda *args: calls.append(1))
                try:
                    command = model.predict(model.demo_observation())
                finally:
                    hook.remove()
                self.assertEqual(calls, [1])
                self.assertEqual(command.shape, (model.info["dofs"],))
                self.assertTrue(np.isfinite(command).all())
                self.assertLessEqual(float(np.abs(command).max()), model.info["command_bound"])

    def test_batch_is_independent_and_preserves_input(self):
        for name, model in self.models.items():
            with self.subTest(system=name):
                observations = np.stack([model.demo_observation(1), model.demo_observation(2)])
                original = observations.copy()
                batch = model.predict(observations)
                separate = np.stack([model.predict(row) for row in observations])
                np.testing.assert_allclose(batch, separate, rtol=1e-4, atol=2e-5)
                np.testing.assert_array_equal(observations, original)

    def test_invalid_observations(self):
        for model in self.models.values():
            good = model.demo_observation()
            for value in (np.zeros((2,)), np.empty((0, *good.shape)),
                          np.full(good.shape, np.nan), np.full(good.shape, np.inf),
                          np.full(good.shape, 1e100), good.astype(complex) + 1j):
                with self.subTest(system=model.system, shape=value.shape):
                    with self.assertRaises(ValueError):
                        model.predict(value)

    def test_physical_units_and_body_order(self):
        scales = {"nikon": [.02,.02,.02,.04,.04],
                  "trepan": [.1,.1,.1,2.5/60,2.5/60],
                  "olympus": [.1,.1,.1,5/60,5/60]}
        for name, model in self.models.items():
            values = model.to_physical(np.ones(model.info["dofs"]))
            np.testing.assert_allclose(values.reshape(-1, 5),
                                       np.tile(scales[name], (model.info["dofs"]//5, 1)))
            with self.assertRaises(ValueError):
                model.to_physical(np.ones(5))

    def test_metadata_is_not_mutable_global_state(self):
        info = system_info("nikon")
        info["input_shape"][0] = 123
        self.assertEqual(system_info("nikon")["input_shape"], [3, 49])
        with self.assertRaises(ValueError):
            Kaleid("unknown")

    def test_cuda_context_preserves_backend_and_restores_flags(self):
        def flags():
            return (torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark,
                    torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32,
                    torch.backends.cuda.matmul.allow_tf32)
        before = flags()
        for system in available_systems():
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                with _precision(torch.device("cuda"), system):
                    self.assertEqual(flags()[:3], before[:3])
                    self.assertEqual(flags()[3:], (system == "olympus",) * 2)
                    raise RuntimeError("test failure")
            self.assertEqual(flags(), before)

    def test_nikon_grid_preserves_constant_samples_inside_support(self):
        grid = self.models["nikon"]._grid
        sums = grid.sum(-1)
        inside = sums > .5
        self.assertGreater(int(inside.sum()), 100)
        np.testing.assert_allclose(sums[inside], 1., atol=2e-7)
        np.testing.assert_allclose(sums[~inside], 0., atol=1e-7)

    def test_cli_file_roundtrip(self):
        model = self.models["olympus"]
        observation = model.demo_observation()
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp)/"input.npz", Path(temp)/"command.npz"
            np.savez(source, observations=observation)
            result = subprocess.run([sys.executable, "-m", "kaleid", "predict", "olympus",
                                     str(source), "--output", str(target), "--device", "cpu"],
                                    capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(result.stdout)["shape"], [35])
            with np.load(target, allow_pickle=False) as output:
                np.testing.assert_allclose(output["command_normalized"], model.predict(observation),
                                           rtol=1e-4, atol=2e-5)
                np.testing.assert_array_equal(output["units"], ["mm","mm","mm","deg","deg"])


if __name__ == "__main__":
    unittest.main()
