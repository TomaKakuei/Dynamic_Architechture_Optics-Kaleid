"""Matched local loss diagnosis; zero training, no deployment or feedback.

Truth is used only for this offline diagnostic and simulated original capture.
Every camera finite difference uses the complete original renderer, including
the changing pupil boundary. Nothing changes in the source experiment.
"""
from pathlib import Path
import sys, os, json, time, argparse, traceback
ENV = Path(r'C:/Users/arthu/anaconda3/envs/nn_310')
sys.path.append(str(ENV/'Lib/site-packages'))
_dll = os.add_dll_directory(str(ENV/'Library/bin'))
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
WORK=ROOT/'DAO_FULLFIELD_20261024'
SOURCE=ROOT/'Adaptive Optics/ADAPTIVE_OPTICS_V15P/kla_joint_sensorless_20261008'
sys.path.insert(0,str(SOURCE))
from common import np, clock, awake, stamp
from camera import Camera, WaveFit, torch
from geometry import Geometry
from optics import Optics
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.writing')
    tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    for attempt in range(20):
        try: os.replace(tmp,p); return
        except PermissionError:
            if attempt==19: raise
            time.sleep(.1)
def state(stage,**kw): save(HERE/'state.json',dict(stage=stage,pid=os.getpid(),updated=stamp(),**kw))
STEPS=[.0001,.0003,.001]
FD=.0001

def prepare():
    if (HERE/'selection.json').exists(): return read(HERE/'selection.json')
    source=WORK/'kla_feedback_loop150_20261028/cases'
    ids=sorted(int(p.stem) for p in source.glob('*.json'))
    chosen=[]
    for cohort in range(3):
        group=[i for i in ids if i//10000==cohort]
        assert len(group)==50
        chosen += [group[j] for j in [0,16,32,49]]
    result=dict(cases=chosen,selection='Before new loss calculations: four equally spaced case IDs per cohort from the existing complete150.',
        baseline='Latest frozen old_wfe_stable E4 one-shot correction, recovered from saved first feedback endpoint minus its recorded first action.',
        count=12,steps=STEPS,camera_fd=FD,
        objectives=dict(wavefront='Mean squared absolute PT-removed wavefront at corrected endpoint, defocus retained; no log/bounding or Q term.',
            pose='Mean squared normalized pose-estimation error.',
            image='Mean original-camera expected-ADU squared residual divided by fixed observed shot/read variance.'),
        selection_of_step='Report all three fixed caps. Optional own-loss choice includes unchanged baseline; Q never selects a step.',
        truth_scope='Offline loss diagnosis only; not an input-only inference method.',
        capture='Reconstruct the original first noisy packet using original pose, camera/calibration and exact dataset seed; do not add feedback.',
        training_updates=0,network_inference_calls=0,automatic_training=False,automatic_deployment=False,
        scope='Reused development/validation cases, not independent validation; 12-case pilot, not all600/3000.')
    save(HERE/'selection.json',result)
    return result

class Study:
    def __init__(self):
        self.camera=Camera(calibration=read(SOURCE/'camera_calibration.json'))
        self.fit=WaveFit();self.geometry=Geometry()
        self.optics={65:Optics(grid=65,device='cuda'),129:Optics(grid=129,device='cuda')}
        with np.load(SOURCE/'training/joint_validation3000.npz') as z:
            self.truth={int(i):q.copy() for i,q in zip(z['indices'],z['initial'])}
        self.times=[]
    @torch.no_grad()
    def irradiance(self,q):
        tick=clock();reps=self.fit.fit(q);values=[];grids=[]
        for rep in reps:
            slope=self.camera.phase_slope(rep)
            if slope>1800: raise ValueError('Original camera bandwidth limit exceeded; do not alter grid or drop case')
            grid=2048 if slope<=450 else (4096 if slope<=900 else 8192)
            grids.append(grid)
            values.append(self.camera.integrated_irradiance(rep,2048) if grid==2048 else self.camera.reference_irradiance(rep,grid,carrier_shift=True))
        self.times.append(clock()-tick)
        return np.asarray(values),grids
    def expected(self,q):
        values,grids=self.irradiance(q)
        electrons=np.maximum(values*self.camera.gain_e*self.camera.prnu,0)
        return (4*np.clip(electrons+self.camera.dsnu,0,10000)+100).astype(np.float32),grids
    @torch.no_grad()
    def physical(self,truth,x):
        endpoint=truth-x
        q=torch.tensor(endpoint[None],device='cuda',dtype=torch.float64)
        a=self.optics[129].metrics(q)
        wrms=a['wrms'][0].cpu().numpy()
        return dict(wfe_mse=float((wrms**2).mean()),wrms_mean=float(wrms.mean()),wrms_worst=float(wrms.max()),
            wrms_fields=wrms.tolist(),all9_below07=bool((wrms<.07).all()),
            q2000=float(a['score'][0]),valid=bool(a['valid'][0]) and self.geometry.path(endpoint,truth),
            pose_mse=float(np.mean((x-truth)**2)))
    def wave_gradient(self,truth,x):
        t=torch.tensor(x[None],device='cuda',dtype=torch.float64,requires_grad=True)
        q=torch.as_tensor(truth[None],device='cuda',dtype=torch.float64)-t
        res,valid=self.optics[65].residual(q)
        w=valid.double();loss=((res.square()*w).sum(-1)/w.sum(-1)).mean()
        grad=torch.autograd.grad(loss,t)[0][0].detach().cpu().numpy()
        return grad,float(loss.detach())
    def data(self,case):
        folder=HERE/'cases'/str(case);folder.mkdir(parents=True,exist_ok=True)
        row=read(WORK/'kla_feedback_loop150_20261028/cases'/f'{case}.json')
        truth=self.truth[case]
        first=np.asarray(row['endpoints'][0])-np.asarray(row['steps'][0]['command'])
        x=truth-first
        path=folder/'packet.npz'
        if path.exists():
            with np.load(path) as z:
                assert np.array_equal(z['truth'],truth) and np.array_equal(z['estimate'],x)
                observed=z['observed'].copy()
        else:
            intensity,grids=self.irradiance(truth)
            seed=193300000+(case//10000)*100000+case%10000
            observed=self.camera.noisy(intensity,seed)
            np.savez_compressed(path,truth=truth,estimate=x,observed=observed,grids=np.asarray(grids),seed=seed)
        measured=observed.astype(float)
        variance=4*np.maximum(measured-(100+4*self.camera.dsnu),0)+16*1.2**2+1/12
        def image_loss(estimate):
            expected,_=self.expected(estimate)
            return float(np.mean((expected-measured)**2/variance))
        return folder,truth,x,image_loss

def cosine(a,b):return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

def run(preflight=False):
    if (HERE/'result.json').exists():print('Already completed; existing result retained.');return
    plan=prepare();state('initializing',cases=plan['cases']);tick=clock();s=Study()
    for case_index,case in enumerate(plan['cases']):
        folder=HERE/'cases'/str(case)
        if (folder/'result.json').exists():continue
        if (HERE/'STOP_REQUESTED.json').exists():state('stopped',case=case);return
        state('case_setup',case=case,case_index=case_index)
        folder,truth,x,image_loss=s.data(case)
        bpath=folder/'baseline.json'
        if bpath.exists():baseline=read(bpath)
        else:
            baseline=s.physical(truth,x);baseline['image_loss']=image_loss(x)
            save(bpath,baseline)
        gpath=folder/'image_fd.json'
        values=read(gpath) if gpath.exists() else {}
        for axis in range(40):
            if str(axis) in values:continue
            direction=np.zeros(40);direction[axis]=FD
            plus=image_loss(x+direction);minus=image_loss(x-direction)
            values[str(axis)]=dict(plus=plus,minus=minus,gradient=(plus-minus)/(2*FD))
            save(gpath,values)
            mean_render=float(np.median(s.times))
            completed=sum((HERE/'cases'/str(i)/'result.json').exists() for i in plan['cases'])
            # Each remaining case uses~92 complete camera renders: capture,
            # baseline,80 coordinate differences and9 matched finite trials.
            remaining=((len(plan['cases'])-completed-1)*92+(40-len(values))*2+9)*mean_render
            state('image_gradient',case=case,case_index=case_index,gradient_axes=len(values),
                camera_seconds_median=mean_render,estimated_remaining_seconds=remaining,
                elapsed_this_invocation=clock()-tick)
            print(json.dumps(dict(case=case,axis=axis,render_seconds=mean_render,remaining_seconds=remaining)),flush=True)
            if preflight and len(values)>=2:
                state('preflight_complete',case=case,gradient_axes=len(values),camera_seconds_median=mean_render,
                    estimated_remaining_seconds=remaining,active_seconds=clock()-tick)
                return
            if (HERE/'STOP_REQUESTED.json').exists():state('stopped',case=case,axis=axis);return
        grads={}
        grads['wavefront'],training_wave_loss=s.wave_gradient(truth,x)
        grads['pose']=2*(x-truth)/40
        grads['image']=np.asarray([values[str(i)]['gradient'] for i in range(40)])
        assert all(np.isfinite(g).all() and np.linalg.norm(g)>0 for g in grads.values())
        checkpath=folder/'image_direction_check.json'
        if checkpath.exists():image_check=read(checkpath)
        else:
            direction=grads['image']/np.max(abs(grads['image']))
            h=1e-5
            finite=(image_loss(x+h*direction)-image_loss(x-h*direction))/(2*h)
            predicted=float(grads['image']@direction)
            image_check=dict(h=h,finite_derivative=finite,gradient_derivative=predicted,
                relative_difference=abs(finite-predicted)/max(abs(finite),abs(predicted),1e-12))
            save(checkpath,image_check)
        result=dict(case=case,baseline=baseline,training_grid65_wfe_mse=training_wave_loss,
            image_direction_check=image_check,
            gradient_alignment={a+'__'+b:cosine(grads[a],grads[b]) for a,b in [('wavefront','pose'),('wavefront','image'),('pose','image')]},trials={})
        for name,gradient in grads.items():
            result['trials'][name]=[]
            for cap in STEPS:
                path=folder/f'{name}_{cap:g}.json'
                if path.exists():trial=read(path)
                else:
                    proposed=x-cap*gradient/np.max(abs(gradient))
                    trial=s.physical(truth,proposed);trial['image_loss']=image_loss(proposed)
                    trial['cap']=cap;trial['max_coordinate_change']=float(np.max(abs(proposed-x)))
                    save(path,trial)
                result['trials'][name].append(trial)
        save(folder/'result.json',result)
        np.savez(folder/'gradients.npz',**grads,truth=truth,estimate=x)
        state('case_complete',case=case,completed=case_index+1,total=len(plan['cases']))
        print(json.dumps(dict(case=case,finished=True)),flush=True)
    summarize(plan)
    state('complete',cases=len(plan['cases']),active_seconds_this_invocation=clock()-tick,assistant_interpretation='pending')

def summarize(plan):
    rows=[read(HERE/'cases'/str(case)/'result.json') for case in plan['cases']]
    def metric(values):
        return dict(count=len(values),mean_wrms=float(np.mean([v['wrms_mean'] for v in values])),
            mean_worst_wrms=float(np.mean([v['wrms_worst'] for v in values])),
            mean_wfe_mse=float(np.mean([v['wfe_mse'] for v in values])),
            mean_pose_mse=float(np.mean([v['pose_mse'] for v in values])),
            mean_image_loss=float(np.mean([v['image_loss'] for v in values])),
            mean_q2000=float(np.mean([v['q2000'] for v in values])),
            all9_below07=sum(v['valid'] and v['all9_below07'] for v in values),
            q_passes=sum(v['valid'] and v['q2000']>=.45 for v in values),invalid=sum(not v['valid'] for v in values))
    baseline=[r['baseline'] for r in rows];output=dict(scope=plan,baseline=metric(baseline),fixed_steps={},own_loss_selected={},alignment={})
    keys={'wavefront':'wfe_mse','pose':'pose_mse','image':'image_loss'}
    for name,key in keys.items():
        output['fixed_steps'][name]={}
        for j,cap in enumerate(STEPS):
            vals=[r['trials'][name][j] for r in rows]
            stats=metric(vals)
            for loss in ['wfe_mse','pose_mse','image_loss']:
                stats[loss+'_improved']=sum(v[loss]<b[loss] for v,b in zip(vals,baseline))
            stats['q_improved']=sum(v['q2000']>b['q2000'] for v,b in zip(vals,baseline))
            output['fixed_steps'][name][str(cap)]=stats
        chosen=[min([r['baseline']]+r['trials'][name],key=lambda v:v[key]) for r in rows]
        output['own_loss_selected'][name]=metric(chosen)
    for name in rows[0]['gradient_alignment']:
        vals=[r['gradient_alignment'][name] for r in rows]
        output['alignment'][name]=dict(mean=float(np.mean(vals)),median=float(np.median(vals)),opposed=sum(v<0 for v in vals))
    output['image_gradient_direction_check_relative_errors']=[r['image_direction_check']['relative_difference'] for r in rows]
    save(HERE/'result.json',output)
    lines=['KLA three-loss local diagnosis: program-generated results; assistant interpretation pending.',
        '12 fixed reused cases, zero training. Original complete camera renderer,40D central differences,3 fixed step caps.',
        'Wavefront and pose gradients use offline truth. This is not a deployable one-shot recovery result.',
        json.dumps(output,indent=2,ensure_ascii=False)]
    (HERE/'GENERATED_REPORT.txt').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');a=parser.parse_args()
    import msvcrt
    lock=open(HERE/'run.lock','a+b')
    if lock.tell()==0:lock.write(b'0');lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    awake()
    try:run(a.preflight)
    except Exception as e:
        state('failed',error=repr(e),traceback=traceback.format_exc());raise
    finally:awake(False)
