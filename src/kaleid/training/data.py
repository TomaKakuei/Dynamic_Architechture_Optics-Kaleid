"""Bounded-memory NPZ datasets with explicit training/validation partitions."""

from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
import torch
from ..predictor import system_info


class Dataset:
    def __init__(self, manifest, system, cache_size=3):
        self.path = Path(manifest).resolve()
        self.spec = json.loads(self.path.read_text(encoding="utf-8"))
        if self.spec.get("system") != system:
            raise ValueError("Dataset system does not match the training model")
        self.system, self.info = system, system_info(system)
        self.cache, self.cache_size = OrderedDict(), cache_size
        self.rows = {}
        used = {}
        for split in ("train", "validation"):
            rows = []
            for record in self.spec.get(split, []):
                record = {"path": record} if isinstance(record, str) else record
                path = (self.path.parent / record["path"]).resolve()
                with np.load(path, allow_pickle=False) as arrays:
                    targets = arrays[self._target_key(arrays)]
                    if (
                        targets.ndim != 2
                        or targets.shape[1] != self.info["dofs"]
                        or not np.isrealobj(targets)
                        or not np.isfinite(targets).all()
                    ):
                        raise ValueError(f"Invalid normalized pose labels in {path}")
                    count = len(targets)
                indices = record.get("indices", range(count))
                kind = record.get("kind", "joint")
                if kind not in ("joint", "front", "rear") or (
                    system != "olympus" and kind != "joint"
                ):
                    raise ValueError(f"Unsupported training cohort: {kind}")
                if split == "validation" and kind != "joint":
                    raise ValueError(
                        "Validate the whole joint system; validation kind must be joint"
                    )
                for index in indices:
                    if (
                        not isinstance(index, (int, np.integer))
                        or not 0 <= index < count
                    ):
                        raise ValueError(f"Invalid row index in {path}: {index}")
                    key = (path, int(index))
                    if key in used:
                        raise ValueError(
                            "Duplicate sample or train/validation row overlap"
                        )
                    used[key] = split
                    rows.append((path, int(index), kind))
            if not rows:
                raise ValueError(f"Empty {split} partition")
            self.rows[split] = rows
        kinds = {r[2] for r in self.rows["train"]}
        if "joint" not in kinds or ("front" in kinds) != ("rear" in kinds):
            raise ValueError(
                "Training needs joint observations; Olympus priors need both front and rear cohorts"
            )

    @staticmethod
    def _target_key(arrays):
        for key in ("targets", "targets_normalized", "q"):
            if key in arrays:
                return key
        raise ValueError(
            "NPZ must contain normalized initial poses: targets, targets_normalized or q"
        )

    def _read(self, path):
        if path not in self.cache:
            with np.load(path, allow_pickle=False) as z:
                observation_key = (
                    "observations"
                    if "observations" in z
                    else ("raw" if self.system == "nikon" else "speckles")
                )
                observations = z[observation_key].copy()
                targets = z[self._target_key(z)].copy()
                maps = (
                    z["maps"].copy() if self.system == "nikon" and "maps" in z else None
                )
            shape = (len(targets), *self.info["input_shape"])
            if (
                observations.shape != shape
                or not np.isrealobj(observations)
                or not np.isfinite(observations).all()
            ):
                raise ValueError(f"Invalid observations in {path}; expected {shape}")
            if maps is not None and (
                maps.shape != (len(targets), 3, 32, 32) or not np.isfinite(maps).all()
            ):
                raise ValueError(f"Invalid Nikon maps in {path}")
            self.cache[path] = observations, targets, maps
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(path)
        return self.cache[path]

    def batch(self, split, indices, device):
        observations, targets, maps = (
            [None] * len(indices),
            [None] * len(indices),
            [None] * len(indices),
        )
        by_file = {}
        for position, index in enumerate(indices):
            path, row, _ = self.rows[split][int(index)]
            by_file.setdefault(path, []).append((position, row))
        for path, selected in by_file.items():
            x, q, m = self._read(path)
            for position, row in selected:
                observations[position], targets[position] = x[row], q[row]
                maps[position] = None if m is None else m[row]
        if any(m is None for m in maps) and not all(m is None for m in maps):
            raise ValueError("Use maps in all Nikon files or omit maps in all files")
        return (
            torch.as_tensor(np.stack(observations).astype(np.float32), device=device),
            torch.as_tensor(np.stack(targets).astype(np.float64), device=device),
            (
                None
                if maps[0] is None
                else torch.as_tensor(np.stack(maps).astype(np.float32), device=device)
            ),
        )

    def schedule(self, epoch, seed, batch_size):
        rows = self.rows["train"]
        groups = {
            k: np.array([i for i, r in enumerate(rows) if r[2] == k])
            for k in ("joint", "front", "rear")
        }
        rng = np.random.default_rng(seed + epoch)

        def shuffled_blocks(indices):
            # Shuffle shards and then their rows. Avoid decompressing a large
            # camera shard separately for every randomly scattered sample.
            blocks = {}
            for index in indices:
                blocks.setdefault(rows[index][0], []).append(index)
            paths = list(blocks)
            return np.concatenate(
                [rng.permutation(blocks[paths[i]]) for i in rng.permutation(len(paths))]
            )

        joint = shuffled_blocks(groups["joint"])
        chunks = [joint[i : i + batch_size] for i in range(0, len(joint), batch_size)]
        if not len(groups["front"]):
            return chunks
        prior = groups["front" if epoch % 2 else "rear"]
        if len(prior) != len(joint):
            raise ValueError(
                "Olympus alternating priors must match the joint cohort size"
            )
        prior = shuffled_blocks(prior)
        priors = [prior[i : i + batch_size] for i in range(0, len(prior), batch_size)]
        return [item for pair in zip(chunks, priors) for item in pair]
