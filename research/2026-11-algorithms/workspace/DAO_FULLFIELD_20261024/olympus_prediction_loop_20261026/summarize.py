"""Summarize saved trial records only; no inference, tracing, or replay."""
import json, pathlib, statistics, datetime

ROOT=pathlib.Path(__file__).resolve().parent
tags=['development_e_prediction','development_f_model_loop','paired32_f_model_loop',
      'development_g_local','paired32_g_local','development_h_adaptive','paired32_h_adaptive']
summary={}
for tag in tags:
    path=ROOT/tag/'result.json'
    if not path.exists():continue
    result=json.loads(path.read_text())
    rows=[json.loads(p.read_text()) for p in sorted(path.parent.glob('case_*.json'))]
    passed=lambda r,name:r[name]['valid'] and max(r[name]['fields'])<.07
    result['gained_ids']=[r['case'] for r in rows if passed(r,'candidate') and not passed(r,'previous')]
    result['lost_ids']=[r['case'] for r in rows if passed(r,'previous') and not passed(r,'candidate')]
    result['additional_mean_seconds']=statistics.mean(r['seconds'] for r in rows)
    result['additional_median_seconds']=statistics.median(r['seconds'] for r in rows)
    result['active_cases']=sum(bool(r.get('steps')) and any(s.get('trials') for s in r['steps']) for r in rows)
    result['accepted_extra_direction_steps']=sum(
        any(t.get('source') in ('local_extra_direction','quadratic_interpolation','bracket') and
            t.get('cost')==s.get('after') for t in s.get('trials',[]))
        for r in rows for s in r.get('steps',[]) if s.get('accepted'))
    result['center_trust_expansions']=sum(s.get('next_center_limit',0)>s.get('center_limit',0)
        for r in rows for s in r.get('steps',[]))
    summary[tag]=result
(ROOT/'comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
lines=['Olympus: local search and model prediction loops',
       'Saved-results interpretation; no additional inference or independent replay.',
       'Updated '+datetime.datetime.now(datetime.timezone.utc).isoformat(),'',
       'Same existing noisy nine-field observation packet. Frozen 4-epoch FP32 network.',
       'Center ABS, signed edge DIFF, all9 fields <.07 waves plus original validity.',
       'Truth is used only to score the fixed command, never to select steps.',
       'Development12 is deliberately mixed and previously observed; paired32 also',
       'uses previously observed cases. Neither is a new independent test.',
       'Original pure-network500 remains317/500; no combined full500 result.',
       'Means include all finite scored fields, including invalid cases; invalid cases never pass.',
       'All times below are EXTRA model-loop/search time after saved previous results;',
       'they do not measure cold end-to-end neural + D + loop inference.', '',
       'E: g(y)-g(F(x)) frozen neural self-consistency proposal, two bounded cycles.',
       'F: two fresh8D full-camera linearizations; gains1/.5/1.5; optional +/- extra directions.',
       'F total departure.2, nominal center trust.01waves, full-resolution aperture model.',
       'G: after F, two extra directions tangent to active center trust; bracket +/- .025',
       'then quadratic-interpolated trial; only actual original image-merit decrease accepted.',
       'G preserves F radius/center limits; maximum7 extra full-camera evaluations.', '']
lines.extend(['H: maximum3 fresh-response cycles; increase center trust .01 -> .02 -> .04',
    'only after an improving in-subspace trial has actual/predicted image-merit gain >=.75',
    'and the previous center bound is active. No extra-axis fixed-step probes.',
    'H uses cached first linearizations from F; its timings are cache-assisted.',
    'G initially stopped on NumPy boolean JSON serialization, repaired with bool conversion.',
    'Already completed records retained. Reported G seconds exclude the unsaved failed attempt.', ''])
for tag,r in summary.items():
    lines.extend([tag,
        '  Fullfield: neural {}/{}; previous {}/{}; candidate {}/{}; gained{}, lost{}'.format(
            r['network']['fullfield_pass'],r['cases'],r['previous']['fullfield_pass'],r['cases'],
            r['candidate']['fullfield_pass'],r['cases'],r['gained_over_previous'],r['lost_previous']),
        '  Center: {} -> {}; invalid: {} -> {}; new invalid{}'.format(
            r['previous']['center_pass'],r['candidate']['center_pass'],r['previous']['invalid'],
            r['candidate']['invalid'],r['new_invalid']),
        '  Edge mean: {:.9f} -> {:.9f} waves'.format(r['previous']['edge_mean'],r['candidate']['edge_mean']),
        '  Incremental total {:.3f}s; mean {:.3f}s/case; {} camera / {} network calls'.format(
            r['additional_seconds'],r['additional_mean_seconds'],r['additional_model_calls'],r['additional_network_calls']),
        '  Newly recovered IDs: {}; lost IDs: {}; accepted extra-direction steps: {}'.format(
            r['gained_ids'],r['lost_ids'],r['accepted_extra_direction_steps']),
        '  Actual center trust expansions: {}'.format(r['center_trust_expansions']),''])
lines.extend(['Assistant interpretation:',
    'E provided no new recovery. F saved one development case but paired32 remained23/32.',
    'F reduced paired32 mean edge error another7.696percent at10.921s additional mean time.',
    'Fixed extra-direction probes were never selected in either F cohort.',
    'G produced accepted finer local steps but no recovery gain over F and slightly worse',
    'development edge mean; its preregistered paired32 extension gate failed.',
    'H also gave no further recovery over F; the center expansion condition never triggered.',
    'H therefore does NOT establish whether a genuinely relaxed center bound would help.',
    'The nominal center-bound explanation remains unproven. Do not widen scoring gates.',
    'Example7354: F image merit312.699 ->303.123 while worst physical field.52825 ->.55596.',
    'This demonstrates non-monotone transfer from image fit to recovery on that case;',
    'it does not establish an information limit or failure of all local methods.',
    'No new variant is promoted to the default. G/H paired32 extensions were not run.', ''])
lines.extend(['Limitations: improving the image fit does not guarantee improving physical recovery.',
              'The nominal center constraint is a model proxy, not an actual endpoint guarantee.',
              'No scoring threshold was widened; invalid cases remain failures.',
              'No original model overwritten, deployment, new observations, or training.',
              'Development F protocol gain metadata was inherited incorrectly from E; the actual',
              'saved trials and implementation used1/.5/1.5. Writer fixed before paired32 F.',
              'All negative variants and results remain saved.'])
(ROOT/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
