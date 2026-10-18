"""Portable offline training, validation selection, and resumable AdamW state."""

import json
import math
import os
from pathlib import Path
import time
import torch
import numpy as np
from .data import Dataset
from .model import build_model, export_bundle, forward
from .objectives import make_objective, aggregate, selection_key
from ..predictor import _precision

RECIPES = {
    "nikon": dict(
        batch_size=128,
        seed=202610051,
        base_lr=2e-5,
        new_lr=1e-4,
        min_lr=2e-6,
        validation_every=1,
    ),
    "trepan": dict(
        batch_size=32,
        seed=9102026,
        base_lr=3e-4,
        new_lr=3e-4,
        min_lr=3e-6,
        validation_every=1,
    ),
    "olympus": dict(
        batch_size=25,
        seed=2026101501,
        base_lr=2e-5,
        new_lr=1e-4,
        min_lr=2e-6,
        validation_every=5,
    ),
}


def save_torch(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def make_optimizer(system, model):
    recipe = RECIPES[system]
    old, new = [], []
    for name, parameter in model.named_parameters():
        is_old = (
            name.startswith(("front.", "rear."))
            if system == "olympus"
            else name.startswith("base.")
        )
        (old if is_old else new).append(parameter)
    return torch.optim.AdamW(
        [
            dict(params=old, lr=recipe["base_lr"], initial_lr=recipe["base_lr"]),
            dict(params=new, lr=recipe["new_lr"], initial_lr=recipe["new_lr"]),
        ],
        weight_decay=1e-4,
    )


@torch.no_grad()
def evaluate(system, model, grid, objective, data, device, batch_size=8):
    model.eval()
    values, validity = [], []
    for start in range(0, len(data.rows["validation"]), batch_size):
        indices = range(start, min(start + batch_size, len(data.rows["validation"])))
        x, q, maps = data.batch("validation", indices, device)
        command, _ = forward(system, model, x, grid, maps)
        w, valid = objective.metrics(command, q)
        values.append(w)
        validity.append(valid)
    return aggregate(np.concatenate(values), np.concatenate(validity))


def train(
    system,
    manifest,
    output,
    *,
    device="auto",
    epochs=50,
    batch_size=None,
    micro_batch=None,
    seed=None,
    calibration=None,
    init=None,
    resume=False,
    max_steps=None,
    save_every=20,
    validation_every=None,
):
    """Run training; max_steps limits additional updates in this invocation.

    Resume restores the next batch, optimizer, RNG, epoch accumulators and best
    validation score. Change epochs/data/seed/batch size only in a new run.
    """
    recipe = RECIPES[system]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    batch_size = recipe["batch_size"] if batch_size is None else batch_size
    micro_batch = batch_size if micro_batch is None else micro_batch
    seed = recipe["seed"] if seed is None else seed
    validation_every = (
        recipe["validation_every"] if validation_every is None else validation_every
    )
    if min(epochs, batch_size, micro_batch, save_every, validation_every) < 1 or (
        max_steps is not None and max_steps < 1
    ):
        raise ValueError("Epoch, batch, checkpoint and step limits must be positive")
    if resume and (init is not None or calibration is not None):
        raise ValueError(
            "Resume restores its own model/calibration; omit init and calibration"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "resume.pt"
    if not resume and any(
        (output / name).exists()
        for name in ("resume.pt", "best.pt", "last.pt", "history.json")
    ):
        raise FileExistsError(
            "Output contains a run; use --resume or choose a new directory"
        )
    data = Dataset(manifest, system)
    config = dict(
        system=system,
        epochs=epochs,
        batch_size=batch_size,
        micro_batch=micro_batch,
        seed=seed,
        validation_every=validation_every,
        dataset=data.spec,
        dataset_root=str(data.path.parent),
        recipe=recipe,
    )
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    previous = (
        torch.load(checkpoint, map_location="cpu", weights_only=True)
        if resume
        else None
    )
    if previous is not None and previous["config"] != config:
        raise ValueError("Resume configuration differs from the saved run")
    if previous is not None:
        # The portable deployment bundle also carries calibration buffers.
        model, grid = build_model(system, init=checkpoint, device=device)
    else:
        model, grid = build_model(
            system, calibration=calibration, init=init, device=device
        )
    objective, optimizer = make_objective(system, device), make_optimizer(system, model)
    history, best, epoch, cursor, updates, loss_sum, seen = [], None, 1, 0, 0, 0.0, 0
    if previous is not None:
        optimizer.load_state_dict(previous["optimizer"])
        history, best = previous["history"], previous["best"]
        epoch, cursor, updates = (
            previous["next_epoch"],
            previous["next_batch"],
            previous["updates"],
        )
        loss_sum, seen = previous["loss_sum"], previous["seen"]
        torch.set_rng_state(previous["cpu_rng"])
        if device.type == "cuda" and previous["cuda_rng"]:
            torch.cuda.set_rng_state_all(previous["cuda_rng"])
    started, initial_updates = time.perf_counter(), updates

    def checkpoint_run():
        state = export_bundle(system, model, grid, epoch - 1)
        state.update(
            config=config,
            optimizer=optimizer.state_dict(),
            history=history,
            best=best,
            next_epoch=epoch,
            next_batch=cursor,
            updates=updates,
            loss_sum=loss_sum,
            seen=seen,
            cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        )
        save_torch(checkpoint, state)
        save_json(output / "history.json", history)

    with _precision(device, system):
        if not history:
            metrics = evaluate(system, model, grid, objective, data, device)
            best = selection_key(metrics)
            history.append(dict(epoch=0, validation=metrics))
            save_torch(output / "best.pt", export_bundle(system, model, grid, 0))
            checkpoint_run()
        while epoch <= epochs:
            rate = 0.5 * (1 + math.cos(math.pi * (epoch - 1) / epochs))
            for group in optimizer.param_groups:
                group["lr"] = (
                    recipe["min_lr"] + (group["initial_lr"] - recipe["min_lr"]) * rate
                )
            schedule = data.schedule(epoch, seed, batch_size)
            model.train()
            for batch in range(cursor, len(schedule)):
                ids = schedule[batch]
                optimizer.zero_grad(set_to_none=True)
                batch_loss = 0.0
                for start in range(0, len(ids), micro_batch):
                    selected = ids[start : start + micro_batch]
                    x, q, maps = data.batch("train", selected, device)
                    command, prediction = forward(system, model, x, grid, maps)
                    loss, _ = objective.loss(
                        command, q, prediction=prediction, model=model
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            "Nonfinite loss; optimizer update was not committed"
                        )
                    (loss * (len(selected) / len(ids))).backward()
                    batch_loss += float(loss.detach()) * len(selected)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5.0, error_if_nonfinite=True
                )
                optimizer.step()
                updates += 1
                loss_sum += batch_loss
                seen += len(ids)
                cursor = batch + 1
                if updates % save_every == 0:
                    checkpoint_run()
                if max_steps is not None and updates - initial_updates >= max_steps:
                    checkpoint_run()
                    save_torch(
                        output / "last.pt",
                        export_bundle(system, model, grid, epoch - 1),
                    )
                    return dict(
                        status="paused",
                        epoch=epoch,
                        next_batch=cursor,
                        updates=updates,
                        mean_loss=loss_sum / seen,
                        seconds=time.perf_counter() - started,
                    )
            row = dict(epoch=epoch, mean_loss=loss_sum / seen, updates=updates)
            if epoch % validation_every == 0 or epoch == epochs:
                metrics = evaluate(system, model, grid, objective, data, device)
                row["validation"] = metrics
                key = selection_key(metrics)
                if key < tuple(best):
                    best = key
                    save_torch(
                        output / "best.pt", export_bundle(system, model, grid, epoch)
                    )
            history.append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
            save_torch(output / "last.pt", export_bundle(system, model, grid, epoch))
            epoch += 1
            cursor = 0
            loss_sum = 0.0
            seen = 0
            checkpoint_run()
    return dict(
        status="complete",
        updates=updates,
        seconds=time.perf_counter() - started,
        selected_epoch=torch.load(
            output / "best.pt", map_location="cpu", weights_only=True
        )["epoch"],
    )
