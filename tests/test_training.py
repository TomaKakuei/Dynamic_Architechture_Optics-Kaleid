"""Offline training contracts, optical gradients, and exact interrupted resume."""

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from kaleid import Kaleid
from kaleid.training.data import Dataset
from kaleid.training.model import build_model, forward
from kaleid.training.objectives import make_objective
from kaleid.training.runner import train
from kaleid.training.smoke import make_smoke_data


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_physical_gradients(self):
        # Compare automatic optical derivatives to a numerical direction in
        # the interior of the valid domain, away from visibility changes.
        for system, size in (("nikon", 35), ("trepan", 40), ("olympus", 35)):
            with self.subTest(system=system):
                objective = make_objective(system)
                q = torch.linspace(-0.01, 0.01, size, dtype=torch.float64)[
                    None
                ].requires_grad_()

                def physical(x):
                    if system == "nikon":
                        return objective.optics.metrics(x)["center"].sum()
                    if system == "trepan":
                        return objective.center.metrics(x)[0].sum()
                    return (
                        objective.center.high_order(x)[0].clamp_min(1e-20).sqrt().sum()
                    )

                value = physical(q)
                (grad,) = torch.autograd.grad(value, q)
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(float(grad.abs().max()), 1e-6)
                direction = torch.zeros_like(q)
                direction[0, int(grad.abs().argmax())] = 1
                step = 1e-4
                numeric = (
                    physical(q.detach() + step * direction)
                    - physical(q.detach() - step * direction)
                ) / (2 * step)
                torch.testing.assert_close(
                    (grad * direction).sum(), numeric, rtol=3e-3, atol=2e-5
                )

    def test_training_without_pretrained_parameters(self):
        for system in ("nikon", "trepan", "olympus"):
            with (
                self.subTest(system=system),
                tempfile.TemporaryDirectory() as temporary,
            ):
                path = make_smoke_data(system, temporary)
                dataset = Dataset(path, system)
                model, grid = build_model(system)
                objective = make_objective(system)
                x, q, maps = dataset.batch("train", [0, 1], "cpu")
                command, prediction = forward(system, model, x, grid, maps)
                loss, details = objective.loss(
                    command, q, prediction=prediction, model=model
                )
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)
                self.assertIn("physical", details)

    def test_resume_matches_uninterrupted_updates_and_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = make_smoke_data("nikon", root / "data")
            kwargs = dict(device="cpu", epochs=2, batch_size=2, seed=71)
            continuous = train(
                "nikon", manifest, root / "continuous", max_steps=2, **kwargs
            )
            train("nikon", manifest, root / "resumed", max_steps=1, **kwargs)
            resumed = train(
                "nikon", manifest, root / "resumed", resume=True, max_steps=1, **kwargs
            )
            self.assertEqual(continuous["updates"], resumed["updates"])
            a = torch.load(root / "continuous/resume.pt", weights_only=True)
            b = torch.load(root / "resumed/resume.pt", weights_only=True)
            for key in a["state_dict"]:
                torch.testing.assert_close(
                    a["state_dict"][key], b["state_dict"][key], rtol=0, atol=0
                )
            model = Kaleid("nikon", checkpoint=root / "resumed/last.pt", device="cpu")
            packet = (
                Dataset(manifest, "nikon").batch("validation", [0], "cpu")[0].numpy()
            )
            self.assertEqual(model.predict(packet).shape, (1, 35))
            done = train("nikon", manifest, root / "resumed", resume=True, **kwargs)
            self.assertEqual(done["status"], "complete")
            self.assertEqual(done["updates"], 4)
            history = json.loads((root / "resumed/history.json").read_text())
            self.assertEqual([row["epoch"] for row in history], [0, 1, 2])
            with self.assertRaises(FileExistsError):
                train("nikon", manifest, root / "resumed", **kwargs)
            with self.assertRaises(ValueError):
                train("nikon", manifest, root / "resumed", resume=True, epochs=3)

    def test_partition_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = make_smoke_data("trepan", temporary)
            value = json.loads(manifest.read_text())
            value["validation"] = value["train"]
            manifest.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "overlap"):
                Dataset(manifest, "trepan")

    def test_olympus_prior_schedule(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = make_smoke_data("olympus", temporary)
            root = Path(temporary)
            source = (root / "train.npz").read_bytes()
            for branch in ("front", "rear"):
                (root / f"{branch}.npz").write_bytes(source)
            value = json.loads(manifest.read_text())
            value["train"] += [dict(path=f"{b}.npz", kind=b) for b in ("front", "rear")]
            manifest.write_text(json.dumps(value))
            dataset = Dataset(manifest, "olympus")
            for epoch, branch in ((1, "front"), (2, "rear")):
                schedule = dataset.schedule(epoch, 101, 2)
                kinds = [dataset.rows["train"][batch[0]][2] for batch in schedule]
                self.assertEqual(kinds, ["joint", branch, "joint", branch])


if __name__ == "__main__":
    unittest.main()
