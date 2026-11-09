"""Observed-camera residual with full-render stochastic coordinate derivatives.

The predictor estimates the INITIAL state as -command. Render that estimate,
not initial+command. Only the observed noisy packet enters this loss; true
training poses do not enter this module. Each sample/epoch uses one uniform
field and one uniform pose coordinate. A central difference, multiplied by40,
estimates the gradient of the nine-field image MSE. This avoids360 camera
derivatives per example, preserves the full native raster, and includes pupil
boundary/phase/carrier/calibration responses in the selected derivative.
"""
from pathlib import Path
import json
import torch
import numpy as np
from camera import Camera, WaveFit
from optics import Optics, numpy_trace
from common import HERE, SPEC, clock


class SelectedWaveFit:
    """Same WaveFit contract, evaluate only the sampled physical field."""
    def __init__(self):
        self.template = WaveFit()
        t = self.template
        self.engines = {}
        self.t = t

    def engine(self, fi):
        if fi not in self.engines:
            t = self.t
            self.engines[fi] = (
                Optics(xy=t.xy, fields=[fi]), Optics(xy=t.probe, fields=[fi]),
                Optics(xy=np.concatenate([t.unit*r for r in t.radial]), fields=[fi]))
        return self.engines[fi]

    @torch.no_grad()
    def fit(self, q, fi):
        t = self.t
        optics, check, bound = self.engine(fi)
        tensor = torch.tensor(np.asarray(q)[None], dtype=torch.float64)
        phase, valid = optics.phase(tensor)
        truth, ok = check.phase(tensor)
        _, bv = bound.phase(tensor)
        phase, valid = phase[0, 0].numpy(), valid[0, 0].numpy()
        truth, ok = truth[0, 0].numpy(), ok[0, 0].numpy()
        bv = bv[0, 0].numpy().reshape(len(t.radial), 256)
        if not bv[0].all() or np.any(np.diff(bv.astype(int), axis=0) > 0):
            raise ValueError('Unsupported estimated pupil topology; no case resampling')
        boundary = np.ones(256)
        if not bv[-1].all():
            lo = np.max(np.where(bv, t.radial[:, None], 0), axis=0)
            hi = np.min(np.where(~bv, t.radial[:, None], 1), axis=0)
            for _ in range(18):
                mid = (lo+hi)/2
                trace = numpy_trace(t.unit*mid[:, None], SPEC['angles_rad'][fi], q)
                lo = np.where(trace[3], mid, lo)
                hi = np.where(trace[3], hi, mid)
            boundary = lo
        fraction = float(np.mean(boundary**2-.05**2)/(1-.05**2))
        if fraction < .98:
            raise ValueError(f'Estimated camera pupil area below original98%: {fraction}')
        coef = t.pinv@phase if valid.all() else np.linalg.lstsq(t.design[valid], phase[valid], rcond=1e-12)[0]
        residual = t.design[valid]@coef-phase[valid]
        error = t.check_design[ok]@coef-truth[ok]
        error -= error.mean()
        er, fr = float(np.sqrt(np.mean(error**2))), float(np.sqrt(np.mean(residual**2)))
        if max(er, fr) > .0003:
            raise RuntimeError(f'Original continuous phase-fit tolerance failed: {fr},{er}')
        matrix = np.zeros((t.degree+1, t.degree+1))
        for k, (i, j) in enumerate(t.terms):
            matrix[i, j] = coef[k]
        linear = np.linalg.lstsq(np.c_[np.ones(valid.sum()), t.xy[valid]], phase[valid], rcond=None)[0]
        return dict(coefficients=matrix, boundary=boundary, carrier=linear[1:],
                    fit_rms_waves=fr, independent_fit_rms_waves=er, pupil_survival_fraction=fraction)


class ImageConsistency:
    def __init__(self, out):
        self.out = Path(out)
        calibration = json.loads((HERE/'camera_calibration.json').read_text(encoding='utf-8'))
        self.camera = Camera(calibration=calibration)
        self.fit = SelectedWaveFit()
        self.h = .001
        self.calls, self.seconds, self.high_resolution_calls = 0, 0., 0
        self.axis_counts = np.zeros(40, dtype=int)
        self.field_counts = np.zeros(9, dtype=int)

    def expected(self, estimate, field):
        rep = self.fit.fit(estimate, field)
        slope = self.camera.phase_slope(rep)
        if slope > 1800:
            raise ValueError('Estimated camera bandwidth exceeds original support')
        grid = 2048 if slope <= 450 else (4096 if slope <= 900 else 8192)
        intensity = (self.camera.integrated_irradiance(rep, 2048) if grid == 2048 else
                     self.camera.reference_irradiance(rep, grid, carrier_shift=True))
        self.calls += 1
        self.high_resolution_calls += int(grid > 2048)
        expected = np.maximum(intensity*self.camera.gain_e*self.camera.prnu, 0)
        return 4*np.clip(expected+self.camera.dsnu, 0, 10000)+100

    def evaluate(self, command, observed, ids, epoch):
        start = clock()
        estimates = -command.detach().double().cpu().numpy()
        data = observed.detach().cpu().numpy()
        values, gradients = [], np.zeros_like(estimates)
        for i, case in enumerate(ids):
            # Same draws across both image-loss arms; independent of outcomes.
            rng = np.random.default_rng(202610310000 + int(epoch)*1000000 + int(case)*17)
            field, axis = int(rng.integers(9)), int(rng.integers(40))
            self.field_counts[field] += 1; self.axis_counts[axis] += 1
            delta = np.zeros(40); delta[axis] = self.h
            try:
                plus = self.expected(estimates[i]+delta, field)
                minus = self.expected(estimates[i]-delta, field)
                fp = float(np.mean(((plus-data[i, field])/1024.)**2))
                fm = float(np.mean(((minus-data[i, field])/1024.)**2))
            except Exception as exc:
                from durable_io import atomic_json
                atomic_json(self.out/'image_loss_failure.json', dict(case=int(case), epoch=int(epoch),
                    field=field, axis=axis, estimate=estimates[i].tolist(), error=repr(exc)))
                raise
            values.append((fp+fm)/2)
            # q_estimate=-command; multiply by40 for the uniform coordinate draw.
            gradients[i, axis] = -40*(fp-fm)/(2*self.h)
        self.seconds += clock()-start
        return (command.new_tensor(values), command.new_tensor(gradients))

    def diagnostics(self):
        return dict(camera_calls=self.calls, camera_seconds=self.seconds,
                    high_resolution_calls=self.high_resolution_calls,
                    field_counts=self.field_counts.tolist(), axis_counts=self.axis_counts.tolist())

    def __call__(self, command, observed, ids, epoch):
        return _ImageLoss.apply(command, observed, ids, epoch, self)


class _ImageLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, command, observed, ids, epoch, engine):
        value, gradient = engine.evaluate(command, observed, ids, epoch)
        ctx.save_for_backward(gradient)
        return value

    @staticmethod
    def backward(ctx, grad_output):
        gradient, = ctx.saved_tensors
        return grad_output[:, None]*gradient, None, None, None, None
