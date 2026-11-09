"""Read already generated results only; no optical/model replay."""
from pathlib import Path
import json,statistics,datetime
ROOT=Path(__file__).resolve().parent
summary={}
for tag in ['development_m_secant2','development_n_expand4','paired32_n_expand4']:
    path=ROOT/tag/'result.json'
    if not path.exists():continue
    r=json.loads(path.read_text())
    rows=[json.loads(p.read_text()) for p in path.parent.glob('case_*.json')]
    active=[x for x in rows if x['actual_camera_calls']>0]
    r['mean_extra_seconds']=statistics.mean(x['seconds'] for x in rows)
    r['active_cases']=len(active)
    r['mean_extra_seconds_active']=statistics.mean(x['seconds'] for x in active) if active else 0
    summary[tag]=r
(ROOT/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
lines=['Olympus secant continuation and complementary-direction experiment',
       'Generated '+datetime.datetime.now(datetime.timezone.utc).isoformat(),'',
       'Prior K32 finished25/32 fullfield, versus F23/32 and pure network20/32.',
       'K gained7488 and7472, zero lost passes, center30/32 and invalid0 unchanged.',
       'Seven K failures: five still above image-cost9 after3 steps; two below9',
       'but still fail edge recovery. Model-image fit alone is not the recovery criterion.', '',
       'M: reuse K last8 directions with rank-one secant model-response updates,',
       'maximum2 constrained continuation steps, no new finite-difference probes.',
       'Stop using predicted local gain scale2*rank/128^2, not global image cost9.',
       'M did not pass the registered development gate: no recovery gain and slightly',
       'worse edge mean. M paired32 is not run.', '',
       'N: retain those8 responses, add4 complementary known optical-risk basis directions',
       'with central complete-camera differences h=.005 at the current K estimate.',
       'Then at most2 jointly constrained steps, with secant response updates.',
       'Original K final center-trust bound, prior radius.2 and per-step.05 retained.',
       'Selection to refine depends only on K accepted-step history, never on labels.',
       'Actual original noisy-image merit selects accepted steps; truth scores only the',
       'fixed final command. One existing packet, frozen neural weights, no new acquisition.',
       'Recovery remains center ABS plus signed edge DIFF, all9<.07 and original validity.', '',
       'These are previously observed12/32 exploratory cases, not fresh independent tests.',
       'No combined full500 result; original pure-network317/500 remains unchanged.',
       'All timings below are incremental after archived K and reused response caches;',
       'they exclude previous stages, neural inference, initialization and offline scoring.',
       'Means of fields include finite scored values even for invalid cases; invalids never pass.', '']
for tag,r in summary.items():
    lines.extend([tag,
        '  Fullfield K{} -> candidate{} / {}; gained{}, lost{}'.format(r['previous']['full'],r['candidate']['full'],r['cases'],r['gained'],r['lost']),
        '  Center {} -> {}; invalid {} -> {}; center lost{}; new invalid{}'.format(r['previous']['center'],r['candidate']['center'],r['previous']['invalid'],r['candidate']['invalid'],r['center_lost'],r['new_invalid']),
        '  Edge mean {:.9f} -> {:.9f} waves'.format(r['previous']['edge_mean'],r['candidate']['edge_mean']),
        '  Extra time {:.3f}s total / {:.3f}s per case / {:.3f}s per active case'.format(r['additional_seconds'],r['mean_extra_seconds'],r['mean_extra_seconds_active']),
        '  {} additional camera calls, {} finite-difference probes, model errors{}'.format(r['additional_camera_calls'],r['new_finite_difference_probes'],r['model_errors']),''])
lines.extend(['N development rescued7463: worst-field.0759789 ->.0360990, while old8-direction',
              'continuation alone could not move it. This supports an insufficient search subspace',
              'on this case, not a universal observability conclusion.',
              'No original weights/defaults overwritten; negative candidates preserved.',
              'Read state.json/HANDOFF.txt for current running/completed status.'])
(ROOT/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({tag:dict(cases=r['cases'],full=r['candidate']['full'],gained=r['gained'],lost=r['lost']) for tag,r in summary.items()},indent=2))
