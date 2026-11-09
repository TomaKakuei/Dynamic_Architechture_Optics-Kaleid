"""Two bounded neural-consistency loops on one unchanged KLA camera packet.

No initial true pose, hidden wavefront, PSF target or evaluation score is an
input to this controller. The original calibrated forward camera is reused.
"""
from pathlib import Path
import sys, json
ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[1]
SOURCE=WORKSPACE/'Adaptive Optics/ADAPTIVE_OPTICS_V15P/kla_joint_sensorless_20261008'
INCUMBENT=ROOT.parent/'kla_objective_alignment_20261028/training_compare'
sys.path.insert(0,str(SOURCE))
from common import np, clock, stamp, atomic, awake
from camera import Camera, WaveFit, torch
from geometry import Geometry
from network import restore
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False


class PredictorLoop:
    def __init__(self):
        calibration=json.loads((SOURCE/'camera_calibration.json').read_text())
        self.camera=Camera(calibration=calibration)
        self.fit=WaveFit();self.geometry=Geometry()
        self.model,_=restore(INCUMBENT/'old_wfe_stable/candidate.pt')
        self.cache=ROOT/'model_images';self.cache.mkdir(exist_ok=True)

    @torch.no_grad()
    def render(self,estimate,case):
        folder=self.cache/str(case);folder.mkdir(exist_ok=True)
        paths=sorted(folder.glob('render_*.npz'))
        for p in paths:
            with np.load(p) as z:
                if np.array_equal(z['estimate'],estimate):
                    self.hits+=1
                    return torch.as_tensor(z['adu'].copy(),device='cuda'),z['grids'].tolist()
        self.calls+=1;reps=self.fit.fit(estimate);intensities=[];grids=[]
        for rep in reps:
            slope=self.camera.phase_slope(rep)
            if slope>1800:raise ValueError('Original camera bandwidth limit exceeded; retain last estimate')
            grid=2048 if slope<=450 else (4096 if slope<=900 else 8192)
            grids.append(grid)
            irradiance=(self.camera.integrated_irradiance(rep,2048) if grid==2048
                        else self.camera.reference_irradiance(rep,grid,carrier_shift=True))
            intensities.append(irradiance)
        # Original known exposure/PRNU/DSNU/ADU. No per-frame normalization,
        # latent acquisition seed or new sensor noise realization is used.
        expected=np.maximum(np.asarray(intensities)*self.camera.gain_e*self.camera.prnu,0)
        adu=(4*np.clip(expected+self.camera.dsnu,0,10000)+100).astype(np.float32)
        path=folder/f'render_{len(paths):03d}.npz'
        np.savez_compressed(path,estimate=estimate,adu=adu,grids=np.asarray(grids))
        return torch.as_tensor(adu,device='cuda'),grids

    def bounded_ray(self,x,d,prior):
        maximum=float(np.abs(d).max())
        if not np.isfinite(d).all() or maximum<1e-10:return None
        scale=min(1.,.05/maximum)
        low=np.maximum(-1.,prior-.1)-x;high=np.minimum(1.,prior+.1)-x
        positive=d>1e-15;negative=d < -1e-15
        if positive.any():scale=min(scale,float((high[positive]/d[positive]).min()))
        if negative.any():scale=min(scale,float((low[negative]/d[negative]).min()))
        if scale<=1e-9:return None
        q=x+scale*d
        if not self.geometry.path(q):return None
        # Estimated correction-path check; actual path is scored offline only.
        if not self.geometry.path(prior-q,prior):return None
        return q

    @torch.no_grad()
    def predict(self,observed,prior,case):
        """g(y) is the archived prior; g(F(x)) uses the SAME frozen network."""
        start=clock();self.calls=0;self.hits=0;network_calls=0
        prior=np.asarray(prior,dtype=np.float64);x=prior.copy();history=[];states=[]
        measured=torch.as_tensor(observed,device='cuda').float()
        dark=torch.as_tensor(100+4*self.camera.dsnu,device='cuda').float()
        variance=4*(measured-dark).clamp_min(0)+16*1.2**2+1/12
        def merit(image):return float(((image-measured).square()/variance).mean())
        def finish(reason,**extra):
            while len(states)<2:states.append(x.copy())
            return dict(command=(-x).tolist(),commands=[(-q).tolist() for q in states],steps=history,
                seconds=clock()-start,actual_camera_calls=self.calls,cache_hits=self.hits,
                network_calls=network_calls,stop_reason=reason,max_departure=float(abs(x-prior).max()),
                truth_in_inference=False,new_acquisitions=0,max_updates=2,**extra)
        try:
            image,grids=self.render(x,case);cost=merit(image);initial=cost
        except (ValueError,RuntimeError) as error:
            return finish('initial_model_error_retained_prior',error=str(error))
        for cycle in range(2):
            if cost<=1.05:
                history.append(dict(cycle=cycle+1,action='noise_level_stop',cost=cost));break
            predicted=-self.model.command(image[None])[0].double().cpu().numpy();network_calls+=1
            direction=prior-predicted;before=cost;best=None;trials=[]
            for gain in (1.,.5,-.5):
                proposal=self.bounded_ray(x,gain*direction,prior)
                if proposal is None or float(abs(proposal-x).max())<1e-9:
                    trials.append(dict(gain=gain,rejected='known_bounds_or_estimated_path'));continue
                try:
                    candidate,candidate_grids=self.render(proposal,case);value=merit(candidate)
                except (ValueError,RuntimeError) as error:
                    trials.append(dict(gain=gain,rejected='camera_model',error=str(error)));continue
                trials.append(dict(gain=gain,merit=value,grids=candidate_grids))
                if np.isfinite(value) and value<cost:
                    cost=value;best=(proposal,candidate,gain)
            accepted=best is not None
            if accepted:x,image,gain=best
            history.append(dict(cycle=cycle+1,before=before,after=cost,accepted=accepted,
                selected_gain=best[2] if accepted else None,trials=trials))
            states.append(x.copy())
            if not accepted:break
        return finish('fixed_two_prediction_loops',initial_merit=initial,final_merit=cost)
