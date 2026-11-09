# Subsequent DAO / Kaleid algorithm research

Source snapshot of the algorithm work following package `2026.10.17.1`, through
the completed November 1 KLA residual experiment. This update contains Python
source, configuration, protocols, and text result records. No checkpoints,
NPZ/NPY arrays, observation datasets, or paper assets are included.

The installed `kaleid` and `kaleid-train` interfaces keep their existing behavior.
Publication of a research candidate does not replace the default model.

## Research directions

| Direction | Source | Recorded scope and status |
| --- | --- | --- |
| Nikon full-packet refinement | [candidate.py](workspace/DAO_FULLFIELD_20261024/nikon_bottleneck_20261026/candidate.py) | Four original model steps plus four steps fitting all 540 observations from the same packet. Recorded 5,978/6,000 full-field passes on the original validation cases; a research candidate with model computation. |
| Olympus constrained and secant refinement | [qp.py](workspace/DAO_FULLFIELD_20261024/olympus_constrained_solver_20261026/qp.py), [secant.py](workspace/DAO_FULLFIELD_20261024/olympus_secant_20261027/secant.py), [search.py](workspace/DAO_FULLFIELD_20261024/olympus_local_ablation150_20261027/search.py) | Original N: 373/500 versus network 317/500. User-locked N_line: 113/150 versus N 112/150 on the same subset. N_line was not evaluated on all 500. |
| KLA simulated measurement feedback | [run.py](workspace/DAO_FULLFIELD_20261024/kla_feedback_full3000_20261028/run.py) | Recorded 1,684, 2,056, then 2,234 passes on the same 3,000 cases using one, two, then three observed packets. Final comparison: 568 gains, 18 losses. |
| KLA model-image residual correction | [models.py](workspace/DAO_FULLFIELD_20261024/kla_full27000_residual_20261101/models.py) | Frozen prior trained on 27,000 examples plus a separately trained bounded residual head. Recorded 124 to 134 passes on the same reused 200 selection cases, with 14 gains and 4 losses. One measured packet plus model rendering; no independent confirmation or deployment. |
| Full-field training and attribution | [DAO_FULLFIELD_20261024](workspace/DAO_FULLFIELD_20261024) | Nikon/Trepan objectives, neural and sensor variants, Olympus repair, and KLA loss/input/spatial-head ablations. Unsuccessful candidates and diagnostic results are retained. |

Each direction retains its own task, metric, measurement budget, and evaluation
set. These counts are reported from the archived experiment records; publishing
source is not a new recovery experiment, independent confirmation, or deployment.
The newest KLA record's historical `assistant_interpretation: pending` field is
preserved. The Olympus user lock is documented in
[LOCK.json](workspace/DAO_FULLFIELD_20261024/olympus_version_lock_20261028/LOCK.json).

## Using the source

`workspace/` preserves the original relative directory layout, including small
source and JSON calibration dependencies. These are research sources, not newly
packaged deployment APIs. The existing scripts identify their required local
checkpoints, observation sets, initialization files, camera arrays, and response
caches; use the corresponding original assets in a separately prepared workspace.
Some runners are specific to Windows/CUDA and start long experiments when run.

Algorithm Python files are copied unchanged. Absolute workspace paths in JSON
records are made relative, and local workspace prefixes in text records are
redacted; numerical fields are preserved. `SOURCE_MANIFEST.json` lists source
and published SHA-256 hashes and each transformation.

## Verify the source snapshot

From the repository root:

```sh
python research/2026-11-algorithms/verify_snapshot.py
python -m unittest discover -s tests -v
```

The snapshot check verifies text-file hashes, Python syntax, JSON parsing, and
the absence of binary data/weights. It does not import research runners, load
checkpoints, or replay experiments. The existing package tests cover its public
inference/training interfaces with small synthetic fixtures.
