"""Simulated measured-feedback loop; controller sees only camera packets.

This explicitly uses up to THREE observations, not one-shot inference.
Hidden state belongs only to the simulated plant and offline scorer.
"""
from pathlib import Path
import sys, os, json, traceback
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'kla_prediction_loop_20261028'
SOURCE=ROOT.parents[1]/'Adaptive Optics/ADAPTIVE_OPTICS_V15P/kla_joint_sensorless_20261008'
INCUMBENT=ROOT.parent/'kla_objective_alignment_20261028/training_compare'
sys.path.insert(0,str(SOURCE))
from common import np, atomic, stamp, clock, awake
from camera import Camera, WaveFit, torch
from network import restore
from optics import Optics
from geometry import Geometry
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def state(stage,**kw):atomic(ROOT/'state.json',dict(stage=stage,pid=os.getpid(),updated=stamp(),**kw))
def stopped():return (ROOT/'STOP_REQUESTED.json').exists()

class ImageController:
    """No simulator, hidden pose or optical metric is available to this class."""
    def __init__(self):self.model,_=restore(INCUMBENT/'old_wfe_stable/candidate.pt')
    @torch.no_grad()
    def propose(self,noisy_adu):
        images=torch.tensor(noisy_adu[None],device='cuda',dtype=torch.float32)
        raw=self.model.command(images)[0].double().cpu().numpy()
        scale=min(1.,.05/max(float(np.abs(raw).max()),1e-30))
        return raw*scale,dict(raw_max=float(np.abs(raw).max()),fixed_bound_scale=scale)

class SimulatedPlant:
    """Physical-model environment: truth generates observations, never actions."""
    def __init__(self):
        self.camera=Camera(calibration=read(SOURCE/'camera_calibration.json'))
        self.fit=WaveFit();self.geometry=Geometry()
    def observe(self,physical_state,seed):
        reps=self.fit.fit(physical_state)
        return self.camera.capture(reps,seed,2048)

def summary(rows):
    output=dict(completed=len(rows),total=50,complete=len(rows)==50,metrics={},paired={},
        mode='Simulated measured-feedback,up to3camera packets; not one-shot',
        observed_packets_by_stage={'one_shot':1,'feedback1':2,'feedback2':3},
        controller_truth_input=False,training_updates=0,automatic_full3000=False,
        seconds=sum(r['active_seconds'] for r in rows),new_captures=sum(r['new_captures'] for r in rows),
        capture_failures=[r['case'] for r in rows if r.get('capture_error')],
        scope='Same fixed50 as original computational-loop pilot; reused validation selection,not independent test')
    flags={}
    for arm in ('one_shot','feedback1','feedback2'):
        s=np.asarray([r['scores'][arm]['score'] for r in rows]);v=np.asarray([r['scores'][arm]['valid'] for r in rows]);p=v&(s>=.45);flags[arm]=p
        output['metrics'][arm]=dict(passes=int(p.sum()),rate=float(p.mean()),invalid=int((~v).sum()),
            q_mean=float(s.mean()),wrms_mean=float(np.mean([r['scores'][arm]['wrms_mean'] for r in rows])))
    for left,right in (('one_shot','feedback1'),('one_shot','feedback2'),('feedback1','feedback2')):
        a,b=flags[left],flags[right]
        output['paired'][left+'_to_'+right]=dict(gained=[rows[i]['case'] for i in np.where(~a&b)[0]],lost=[rows[i]['case'] for i in np.where(a&~b)[0]])
    return output

def main():
    if (ROOT/'result.json').exists():print((ROOT/'result.json').read_text());return
    if stopped():state('stopped');return
    tick=clock();state('initializing')
    selection=read(BASE/'selection.json');atomic(ROOT/'selection.json',selection)
    with np.load(SOURCE/'training/joint_validation3000.npz') as z:initial=z['initial'].copy();ids=z['indices'].copy()
    with np.load(INCUMBENT/'selected_validation3000_grid129.npz') as z:command=z['command'].copy();baseline_wrms=z['wrms'].copy()
    with np.load(INCUMBENT/'selected_refined_scores.npz') as z:baseline_score=z['score'].copy();baseline_valid=z['valid'].copy()
    atomic(ROOT/'protocol.json',dict(cases=50,selection='Exact same50 as computational-loop pilot',
        checkpoint=str(INCUMBENT/'old_wfe_stable/candidate.pt'),max_additional_feedback_steps=2,
        max_total_observed_packets=3,first_observation_and_command='Reuse archived one-shot noisy packet and correction',
        feedback='Apply prior correction, acquire new independent noisy packet at the resulting physical state, run same frozen neural controller',
        action='Neural correction times min(1,0.05/maxabs(raw)); fixed bound before any score access',
        candidate_selection='None; one fixed action per feedback packet, no true-score gains or oracle rejection',
        scoring='Only after both proposed feedback actions have been fixed; original Q2000,129/257 with65disagreement; complete actual paths',
        physical_safety='Paths evaluated offline; failed simulated observation keeps last state and records failure rather than silently dropping case',
        camera='Original stochastic noisy fixed-exposure9field camera,original2048/4096/8192 bandwidth policy',
        observation_noise_seed='New fixed study seed1933290000+case*10+cycle; hidden environment only,not controller input',
        training_updates=0,truth_input_to_controller=False,one_shot=False,automatic_full3000=False))
    controller=ImageController();plant=SimulatedPlant();metric={n:Optics(grid=n,device='cuda') for n in (65,129,257)}
    (ROOT/'cases').mkdir(exist_ok=True);(ROOT/'observations').mkdir(exist_ok=True)
    @torch.no_grad()
    def score(endpoint,path_valid):
        q=torch.tensor(endpoint[None],device='cuda',dtype=torch.float64)
        a=metric[129].metrics(q);b=metric[65].metrics(q);s=float(a['score'][0]);s65=float(b['score'][0]);valid=bool(a['valid'][0] and b['valid'][0] and path_valid);grid=129
        if abs(s-.45)<.02 or (s>=.45)!=(s65>=.45):
            c=metric[257].metrics(q);s=float(c['score'][0]);valid=valid and bool(c['valid'][0]);grid=257
        return dict(score=s,valid=valid,wrms_mean=float(a['wrms'].mean()),grid=grid)
    records=[]
    for row in selection['rows']:
        case=int(ids[row]);dest=ROOT/'cases'/f'{case}.json'
        if dest.exists():records.append(read(dest));continue
        if stopped():state('stopped',completed=len(records),total=50);return
        start=clock();state('feedback_case',completed=len(records),total=50,case=case)
        # Environment starts after the archived first command; the controller
        # receives only newly generated camera observations below.
        physical=initial[row]+command[row];first=physical.copy();steps=[];endpoints=[];captured=0;failure=None
        for cycle in range(2):
            observation_path=ROOT/'observations'/f'{case}_{cycle}.npz'
            try:
                if observation_path.exists():
                    with np.load(observation_path) as z:
                        if not np.array_equal(z['state'],physical):raise ValueError('Cached observation belongs to different plant state')
                        obs=z['adu'].copy()
                else:
                    obs=plant.observe(physical,1933290000+case*10+cycle)
                    np.savez_compressed(observation_path,state=physical,adu=obs)
                captured+=1
            except (ValueError,RuntimeError) as error:
                failure=dict(cycle=cycle+1,error=str(error));break
            action,info=controller.propose(obs)
            physical=physical+action;endpoints.append(physical.copy())
            steps.append(dict(cycle=cycle+1,command=action.tolist(),**info))
        while len(endpoints)<2:endpoints.append(physical.copy())
        # No recovery score is calculated until all actions above are fixed.
        scores={'one_shot':dict(score=float(baseline_score[row]),valid=bool(baseline_valid[row]),wrms_mean=float(baseline_wrms[row].mean()),grid='saved')}
        previous=first;path_valid=plant.geometry.path(first,initial[row])
        for k,arm in enumerate(('feedback1','feedback2')):
            endpoint=endpoints[k];path_valid=path_valid and plant.geometry.path(endpoint,previous)
            scores[arm]=score(endpoint,path_valid);previous=endpoint
        record=dict(case=case,row=row,scores=scores,steps=steps,endpoints=[e.tolist() for e in endpoints],
            new_captures=captured,capture_error=failure,active_seconds=clock()-start)
        atomic(dest,record);records.append(record);report=summary(records);atomic(ROOT/'partial_summary.json',report)
        eta=float(np.mean([r['active_seconds'] for r in records[-10:]]))*(50-len(records))
        state('case_complete',completed=len(records),total=50,last_case=case,estimated_remaining_seconds=eta)
        print(json.dumps(dict(completed=len(records),case=case,passes={k:v['passes'] for k,v in report['metrics'].items()},seconds=record['active_seconds'])),flush=True)
    report=summary(records);atomic(ROOT/'result.json',report)
    state('complete',completed=50,total=50,active_seconds=clock()-tick,assistant_interpretation='pending')
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    awake()
    try:main()
    except Exception as error:state('failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:awake(False)
