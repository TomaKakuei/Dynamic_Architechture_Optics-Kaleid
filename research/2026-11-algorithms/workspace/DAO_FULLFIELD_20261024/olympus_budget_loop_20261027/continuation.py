"""Continue saved N endpoints with a wider, bounded model-only search."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
NROOT = HERE.parent / 'olympus_secant_20261027'
KROOT = HERE.parent / 'olympus_constrained_solver_20261026'
sys.path.insert(0, str(NROOT))
from secant import Refiner, common, np, torch, sigma_from_adu
from qp import solve_local


class BudgetRefiner(Refiner):
    def __init__(self):
        super().__init__('hybrid')
        self.precision = np.linalg.inv(self.cov) / 128**2
        self.cache = HERE / 'responses'
        self.cache.mkdir(exist_ok=True)

    @torch.no_grad()
    def predict(self, observed, prior, start, k_start, case, source_tag,
                last_cycle, center_limit, eligible, steps=4, field_weights=None):
        x = np.asarray(start, dtype=np.float64).copy()
        prior = np.asarray(prior, dtype=np.float64)
        k_start = np.asarray(k_start, dtype=np.float64)
        history = []; calls = 0; probes = 0; tick = common.clock()
        output = dict(truth_used_in_inference=False, new_acquisitions=0,
                      additional_steps_limit=steps, search_rank=16,
                      gains=[1., .5, .25, .125, .0625], center_limit=center_limit)

        def finish(reason, **kw):
            return dict(command=-x, steps=history, seconds=common.clock()-tick,
                        actual_camera_calls=calls, new_finite_difference_probes=probes,
                        stop_reason=reason, **output, **kw)

        if not eligible or last_cycle is None:
            return finish('no_active_N_response_retained_N')
        obs = torch.as_tensor(observed, device='cuda').float()
        sigma = sigma_from_adu(obs)
        weights = np.ones(9) if field_weights is None else np.asarray(field_weights, dtype=float)
        if weights.shape != (9,) or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError('Expected nine positive finite field weights')
        weights = weights * 9 / weights.sum()
        field_scale = torch.as_tensor(np.sqrt(weights), device='cuda').double()[:, None, None]
        output['field_weights_normalized_sum9'] = weights.tolist()
        P = torch.as_tensor(self.precision, device='cuda').double()

        def render(q):
            nonlocal calls
            calls += 1
            return self.camera.evaluate(q)[0]

        def residual(im):
            return (((im-obs)/sigma/128).double()*field_scale).flatten()

        def merit(im, q):
            d = torch.as_tensor(q-prior, device='cuda').double()
            return float(residual(im).square().sum()+d@P@d)

        def secant(J, change, coordinate):
            z = torch.as_tensor(coordinate, device='cuda').double()
            norm = float(z@z)
            if norm < 1e-16:
                return J, 0.
            error = change-J@z
            return J+error[:, None]*z[None, :]/norm, float(error.norm()/change.norm().clamp_min(1e-12))

        def coordinates(B, delta):
            z = np.linalg.lstsq(B, delta, rcond=None)[0]
            if np.linalg.norm(B@z-delta) > 1e-7:
                raise ValueError('Saved move outside its response subspace')
            return z

        try:
            path = KROOT/'responses'/source_tag/f'{case}_{last_cycle}.npz'
            if not path.exists():
                path = KROOT/'responses'/'development_j_central8'/f'{case}_{last_cycle}.npz'
            with np.load(path) as z:
                q0 = z['prior'].copy(); basis = z['basis'].copy()
                image0 = torch.as_tensor(z['image'], device='cuda')
                raw = torch.stack([torch.as_tensor(z[f'jac_{j}'], device='cuda') for j in range(8)], dim=-1)
            if np.array_equal(q0, k_start):
                k_image = image0
            else:
                with np.load(NROOT/'images'/f'{case}.npz') as z:
                    if not np.array_equal(z['estimate'], k_start):
                        raise ValueError('Saved K terminal image mismatch')
                    k_image = torch.as_tensor(z['image'], device='cuda')
            J = ((raw/sigma[..., None]/128).double()*field_scale[..., None]).flatten(0, -2)
            J, k_error = secant(J, residual(k_image)-residual(image0), coordinates(basis[:, :8], k_start-q0))
            with np.load(NROOT/'extra_response'/f'{case}.npz') as z:
                if not np.array_equal(z['estimate'], k_start):
                    raise ValueError('Saved N extra response mismatch')
                extra = [((torch.as_tensor(z[f'jac_{j}'], device='cuda')/sigma/128).double()*field_scale).flatten() for j in range(8, 12)]
            J = torch.cat([J, torch.stack(extra, dim=-1)], dim=1)

            dest = self.cache/f'{case}.npz'
            known = {}
            if dest.exists():
                with np.load(dest) as z:
                    known = {k: z[k].copy() for k in z.files}
                if not np.array_equal(known['estimate'], x):
                    raise ValueError('Continuation cache endpoint mismatch')
            else:
                current = k_image if np.array_equal(x, k_start) else render(x)
                known = dict(estimate=x.copy(), image=current.cpu().numpy())
                np.savez_compressed(dest, **known)
            current = torch.as_tensor(known['image'], device='cuda')
            # The archived N final Jacobian was not saved. Transfer the existing
            # 12 directions to its endpoint with one aggregate secant update.
            # This is a continuation approximation, not an exact N-J replay.
            J, n_error = secant(J, residual(current)-residual(k_image), coordinates(basis[:, :12], x-k_start))
            columns = []
            for j in range(12, 16):
                name = f'jac_{j}'; direction = basis[:, j]
                if name not in known:
                    mask = np.abs(direction) > 1e-12
                    h = min(.005, .9*float(((1-np.abs(x[mask]))/np.abs(direction[mask])).min()))
                    if h < 1e-6:
                        raise ValueError('Insufficient symmetric probe space')
                    plus = render(x+h*direction); minus = render(x-h*direction)
                    probes += 2
                    known[name] = ((plus-minus)/(2*h)).cpu().numpy()
                    known[f'h_{j}'] = h
                    np.savez_compressed(dest, **known)
                columns.append(((torch.as_tensor(known[name], device='cuda')/sigma/128).double()*field_scale).flatten())
            J = torch.cat([J, torch.stack(columns, dim=-1)], dim=1)
            B = basis[:, :16]; bt = torch.as_tensor(B, device='cuda').double()
            cost = merit(current, x); initial = cost
            gain_floor = 2*B.shape[1]/128**2
            for cycle in range(steps):
                r = residual(current); d = torch.as_tensor(x-prior, device='cuda').double()
                Q = J.T@J+bt.T@P@bt; g = J.T@r+bt.T@P@d
                dz, info = solve_local(Q.cpu().numpy(), g.cpu().numpy(), B, x, prior,
                                       self.center_gram, .2, center_limit)
                before = cost; accepted = False; trials = []; correction = None
                if info['feasible'] and info['predicted_gain'] > gain_floor:
                    for gain in output['gains']:
                        u = torch.as_tensor(gain*dz, device='cuda').double()
                        q = x+B@(gain*dz)
                        if not self.camera.camera.engine.geometry.path(q)['safe']:
                            trials.append(dict(gain=gain, geometry_rejected=True)); continue
                        if not self.camera.camera.engine.geometry.path(prior-q, start=prior)['safe']:
                            trials.append(dict(gain=gain, correction_path_rejected=True)); continue
                        try:
                            im = render(q); value = merit(im, q)
                        except (ValueError, RuntimeError) as exc:
                            trials.append(dict(gain=gain, error=str(exc))); continue
                        predicted = float((r+J@u).square().sum()+(d+bt@u)@P@(d+bt@u))
                        rho = (before-value)/max(before-predicted, 1e-12)
                        trials.append(dict(gain=gain, cost=value, predicted_cost=predicted, agreement=rho))
                        if value < before and rho >= .1:
                            J, correction = secant(J, residual(im)-r, gain*dz)
                            x = q; current = im; cost = value; accepted = True
                            break
                history.append(dict(step=cycle+1, before=before, after=cost, accepted=accepted,
                                    solver=info, trials=trials, secant_relative_correction=correction))
                if not accepted:
                    break
            return finish('bounded_wider_continuation', initial_image_cost=initial,
                          final_image_cost=cost, K_secant_correction=k_error, N_secant_correction=n_error)
        except (ValueError, RuntimeError, OSError) as exc:
            return finish('model_error_retained_last_estimate', error=str(exc))
