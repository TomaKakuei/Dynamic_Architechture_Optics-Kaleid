"""Interpret saved loop trajectories without new network/optical evaluations."""
from pathlib import Path
import json, sys, numpy as np
ROOT=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
rows=[read(p) for p in sorted((ROOT/'cases').glob('*.json'))]
changes=[];accepted=[0,0];reductions=[];gains={}
for r in rows:
    before=r['metrics']['one_shot'];after=r['metrics']['loop2'];info=r['inference']
    passed_before=before['valid'] and before['score']>=.45
    passed_after=after['valid'] and after['score']>=.45
    if 'initial_merit' in info:
        reduction=(info['initial_merit']-info['final_merit'])/max(info['initial_merit'],1e-30)
        reductions.append(reduction)
    else:reduction=None
    for k,s in enumerate(info['steps']):
        if s.get('accepted'):
            accepted[k]+=1;key=str(s['selected_gain']);gains[key]=gains.get(key,0)+1
    if passed_before!=passed_after:
        changes.append(dict(case=r['case'],change='gain' if passed_after else 'loss',q_before=before['score'],
            q_after=after['score'],relative_image_merit_reduction=reduction,max_departure=info['max_departure']))
result=dict(count=len(rows),threshold_changes=changes,accepted_updates_by_cycle=accepted,selected_gains=gains,
    median_relative_image_merit_reduction=float(np.median(reductions)),
    cases_Q_improved=sum(r['metrics']['loop2']['score']>r['metrics']['one_shot']['score'] for r in rows),
    cases_Q_worsened=sum(r['metrics']['loop2']['score']<r['metrics']['one_shot']['score'] for r in rows),
    cases_unchanged=sum(r['metrics']['loop2']['score']==r['metrics']['one_shot']['score'] for r in rows),
    scope='Saved trajectory analysis, not additional inference or replay')
(ROOT/'trajectory_analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
