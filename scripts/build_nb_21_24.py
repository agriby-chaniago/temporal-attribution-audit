"""Build the prospective 20-seed extension; never execute scientific cells.

Notebook 21: design/inputs/schedules. 22: NewHandPD. 23: UCI-SST. 24: joint report.
Scientific additions live in the generated notebooks, emitted by this builder.
Legacy builders are parsed as text, never imported or executed.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "plans/attention_audit_replication20_20261019"
CONFIG = {
    "format_version": 1, "design_date": "2026-10-19",
    "total_seeds": list(range(20)), "discovery_seeds": [0, 1, 2],
    "architectures": ["gru", "mamba2"], "n_folds": 5, "split_seed": 42,
    "schedule_seed": 2101, "schedule_offsets": [-2, -1, 1, 2, 10],
    "bootstrap_seed": 2102, "n_boot": 5000, "confidence": 0.95,
    "placebo_seed": 2103, "placebo_candidates": 199, "placebo_jsd_caliper": 0.025,
    "match_seed": 2104, "intervention_seed": 1600,
    "model_config": {"in_channels": 6, "d_model": 128, "patch_size": 7,
                     "n_layers": 3, "dropout": 0.30, "d_dir": 64,
                     "pooling": "attention"},
    "training_config": {"epochs": 35, "batch_size": 8, "lr": 0.0003},
    "fractions": [0.05, 0.10, 0.20, 0.30], "patch": 7, "fs": 100.0,
    "random_repeats": 20, "donor_repeats": 5, "donor_block_patches": 32,
    "rms_relative_tolerance": 0.05, "rms_absolute_tolerance": 1e-8,
    "max_candidates": 4096, "candidate_batch": 128, "dose_sensitivity_caliper": 0.05,
    "normalization_floor": 1e-8, "min_ci_subjects": 10,
    "newhandpd_cache": "data/cache/newhandpd_spiral_fs100.npz",
    "uci395_cache": "data/cache/uci395_fs100.npz", "uci_primary_task": 0,
    "discovery_run": "results/e2/20261012T050124548170Z",
    "discovery_swap": "results/attention_ranking_swap/20261017T002514477657Z",
    "public_preregistration": False,
    "analysis_status": "prospectively_specified_extension_of_known_exploratory_results",
}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def old_function_sources(number, names):
    """Extract only literal code/function definitions; no legacy execution."""
    raw = (ROOT / f"scripts/build_nb_{number}.py").read_text()
    tree = ast.parse(raw)
    snippets = []
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        # Notebook-20's methods are returned as a literal source string.
        if fn.name == "swap_methods_source":
            snippets.append(ast.literal_eval(fn.body[0].value))
        if fn.name in {"build_notebook", "_finish_notebook"}:
            for node in fn.body:
                if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name)
                        and node.value.func.id == "code"):
                    for argument in node.value.args:
                        if isinstance(argument,ast.Constant) and isinstance(argument.value,str):
                            snippets.append(argument.value)
    found = {}
    for source in snippets:
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                if node.name in found:
                    raise ValueError("Duplicate legacy function " + node.name)
                found[node.name] = ast.get_source_segment(source, node)
    if set(found) != set(names):
        raise ValueError(f"Missing notebook-{number} functions: {set(names)-set(found)}")
    return "\n\n".join(found[name] for name in names)


def schedule_rows():
    rows = []
    for cohort in ("newhandpd", "uci395"):
        for arm in CONFIG["architectures"]:
            key = json.dumps([CONFIG["schedule_seed"], cohort, arm]).encode()
            seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
            ring = CONFIG["total_seeds"].copy()
            random.Random(seed).shuffle(ring)
            for i, target in enumerate(ring):
                for offset in CONFIG["schedule_offsets"]:
                    source = ring[(i + offset) % len(ring)]
                    rows.append(dict(cohort=cohort, architecture=arm, target_seed=target,
                        source_seed=source, ring_position=i, offset=offset,
                        generator_seed=seed,
                        primary_pair=(cohort == "uci395" or (target >= 3 and source >= 3))))
    for cohort in ("newhandpd", "uci395"):
        for arm in CONFIG["architectures"]:
            g = [r for r in rows if r["cohort"] == cohort and r["architecture"] == arm]
            edges = {(r["target_seed"], r["source_seed"]) for r in g}
            assert len(g) == len(edges) == 100
            assert all(a != b and (b, a) in edges for a, b in edges)
            for seed in range(20):
                assert sum(r["target_seed"] == seed for r in g) == 5
                assert sum(r["source_seed"] == seed for r in g) == 5
            if cohort == "newhandpd":
                assert all(2 <= sum(r["target_seed"] == seed and r["primary_pair"]
                                   for r in g) <= 5 for seed in range(3, 20))
    return rows


COMMON = r'''
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
from contextlib import contextmanager
from importlib.metadata import version
import csv, gzip, json, os, pickle, platform, socket, sys, time
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from IPython.display import display

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
            if (p/'src/e2_checkpoints.py').is_file())
sys.path.insert(0, str(ROOT/'src'))
from audit_e0_e1 import AlignmentError, cache_manifest, file_sha256
from e2_checkpoints import spec_digest
PLAN = ROOT/'plans/attention_audit_replication20_20261019'
LOCK = json.loads((PLAN/'design_lock.json').read_text())
C = LOCK['configuration']
if file_sha256(PLAN/'PROTOCOL.md') != LOCK['protocol_sha256']:
    raise AlignmentError('protocol changed after design lock; rebuild before new outcomes')
for relative, expected in LOCK['inputs'].items():
    if file_sha256(ROOT/relative) != expected:
        raise AlignmentError('pinned input/source changed: '+relative)
if file_sha256(PLAN/'directed_seed_schedule.csv') != LOCK['schedule_sha256']:
    raise AlignmentError('transfer schedule differs from design lock')
schedule_all = pd.read_csv(PLAN/'directed_seed_schedule.csv')
schedule_all['primary_pair'] = schedule_all.primary_pair.astype(str).str.lower().map({'true': True, 'false': False})
if schedule_all.primary_pair.isna().any(): raise AlignmentError('invalid schedule boolean')


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, path)


def atomic_pickle(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp.'+str(os.getpid()))
    blob=gzip.compress(pickle.dumps(value,protocol=5),compresslevel=3)
    with tmp.open('wb') as stream:pickle.dump(dict(format='gzip_pickle_v1',sha256=sha256(blob).hexdigest(),payload=blob),stream,protocol=5)
    os.replace(tmp, path)


def read_local_pickle(path):
    with Path(path).open('rb') as stream:envelope=pickle.load(stream)
    if envelope['format']!='gzip_pickle_v1' or sha256(envelope['payload']).hexdigest()!=envelope['sha256']:
        raise AlignmentError('cached payload checksum mismatch: '+str(path))
    return pickle.loads(gzip.decompress(envelope['payload']))


def notebook_code_digest(path):
    doc=json.loads(Path(path).read_text())
    chunks=[]
    for cell in doc['cells']:
        if cell['cell_type']=='code':
            value=cell['source'];chunks.append(''.join(value) if isinstance(value,list) else value)
    return sha256('\n'.join(chunks).encode()).hexdigest()


def atomic_csv(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp.'+str(os.getpid()))
    data.to_csv(tmp, index=False); os.replace(tmp, path)


@contextmanager
def stage_lock(directory, stage):
    path = directory/(stage+'.active.lock')
    directory.mkdir(parents=True, exist_ok=True)
    if path.exists():
        owner = json.loads(path.read_text())
        if owner['host'] != socket.gethostname():
            raise RuntimeError('stage lock belongs to another host; inspect before recovering: '+str(path))
        try: os.kill(int(owner['pid']), 0)
        except ProcessLookupError:
            path.rename(path.with_name(path.name+'.stale.'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
        else:
            raise RuntimeError('stage already running in PID '+str(owner['pid'])+'; no concurrent Run All')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, 'w') as stream:
        json.dump(dict(pid=os.getpid(), host=socket.gethostname()), stream)
    try: yield
    except BaseException as error:
        if 'manifest' in globals() and 'OUT' in globals():
            manifest.update(completion_status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                            failed_stage=stage,error_type=type(error).__name__,error_message=str(error))
            atomic_json(OUT/'analysis_manifest.json',manifest)
        raise
    finally:
        if path.exists(): path.unlink()


def prepare_cohort(cohort):
    cache = ROOT/C[cohort+'_cache']
    records = cache_manifest(cache, cohort, C['patch'], C['fs'])
    if cohort == 'uci395':
        records = records.loc[records.task == C['uci_primary_task']].copy()
    records = records.rename(columns={'idx': 'cache_idx'}).reset_index(drop=True)
    records['idx'] = np.arange(len(records))
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(splitter.split(np.zeros(len(records)), records.label, records.subject))
    folds = np.full(len(records), -1, int)
    for fold, (train, test) in enumerate(splits):
        if set(records.subject.iloc[train]) & set(records.subject.iloc[test]):
            raise AlignmentError('subject leakage in fold')
        if set(records.label.iloc[test]) != {0, 1}:
            raise AlignmentError('held-out fold lacks a class; no automatic split change')
        folds[test] = fold
    if (folds < 0).any(): raise AlignmentError('incomplete folds')
    records['fold'] = folds
    if cohort == 'newhandpd':
        old = pd.read_csv(ROOT/C['discovery_run']/'record_manifest.csv')
        if len(old) != len(records) or not np.array_equal(old.record_id, records.record_id):
            raise AlignmentError('NewHandPD records differ from discovery bank')
        field = 'fold_actual' if 'fold_actual' in old else 'fold'
        if not np.array_equal(old[field].to_numpy(int), folds):
            raise AlignmentError('NewHandPD fold identities differ')
        if records.subject.nunique() != 66 or len(records) != 264:
            raise AlignmentError('NewHandPD bank denominator differs')
    return records, splits
'''


BANK_METHODS = r'''
preflight=PLAN/'preflight/completion.json'
if not preflight.is_file():raise FileNotFoundError('run notebook 21 before training: '+str(preflight))
preflight_info=json.loads(preflight.read_text())
if preflight_info['completion_status']!='complete' or preflight_info['design_id']!=LOCK['design_id']:
    raise AlignmentError('preflight belongs to an incomplete/different design')
import torch
from e2_checkpoints import fold_spec, get_fold_model, FrozenPredictor
from audit_e2 import donor_background, apply_mask, keyed_seed
from preprocessing import muat_cache
from newhandpd import muat_cache_bisp
import matplotlib.pyplot as plt

if not torch.cuda.is_available(): raise RuntimeError('CUDA project environment required; no CPU training fallback')
DEVICE = 'cuda'
VERSIONS = {p: version(p) for p in ('numpy','pandas','scipy','scikit-learn','torch',
                                   'mamba-ssm','causal-conv1d','triton','einops')}
VERSIONS['cuda_runtime'] = str(torch.version.cuda)
runtime = dict(versions=VERSIONS, device=torch.cuda.get_device_name(0),
    capability=list(torch.cuda.get_device_capability(0)),
    matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
    cudnn_tf32=torch.backends.cudnn.allow_tf32,
    cudnn_benchmark=torch.backends.cudnn.benchmark,
    cudnn_deterministic=torch.backends.cudnn.deterministic,
    deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
RUNTIME_ID = spec_digest(runtime)
DESIGN_ID = LOCK['design_id']
OUT = ROOT/'results/attention_replication20'/COHORT/DESIGN_ID/RUNTIME_ID
FIG = ROOT/'figures/attention_replication20'/COHORT/DESIGN_ID/RUNTIME_ID
MODEL_STORE = ROOT/'results/attention_replication20_models'/COHORT
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
SESSION = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
SCIENTIFIC_SOURCE_DIGEST=notebook_code_digest(NOTEBOOK)
manifest = dict(cohort=COHORT, design_id=DESIGN_ID, runtime_id=RUNTIME_ID,
    completion_status='initializing', session=SESSION, analysis_status=C['analysis_status'],
    public_preregistration=False, old_preregistration_changed=False,
    target_seeds=C['total_seeds'], n_expected_checkpoints=200, runtime=runtime,
    source_notebook=str(NOTEBOOK.relative_to(ROOT)), protocol_sha256=LOCK['protocol_sha256'])
atomic_json(OUT/'analysis_manifest.json', manifest)
atomic_json(OUT/'sessions'/('start_'+SESSION+'.json'), manifest)
records, splits = prepare_cohort(COHORT)
cache = ROOT/C[COHORT+'_cache']
raw_records = muat_cache_bisp(cache) if COHORT == 'newhandpd' else muat_cache(cache)
recordings = [raw_records[int(i)] for i in records.cache_idx]
del raw_records
for rec, row in zip(recordings, records.itertuples(index=False)):
    if (str(rec.subjek) != row.subject or rec.kelompok not in {'HC','PD'}
            or rec.label != row.label or rec.tugas != row.task or rec.fs != C['fs']
            or rec.kanal.shape != (row.n_samples, 6) or not np.isfinite(rec.kanal).all()):
        raise AlignmentError('cache/record manifest mismatch: '+row.record_id)
atomic_csv(OUT/'record_manifest.csv', records)
schedule = schedule_all.loc[schedule_all.cohort == COHORT].copy()
atomic_csv(OUT/'directed_seed_schedule.csv', schedule)
ARMS, SEEDS = tuple(C['architectures']), tuple(C['total_seeds'])
PATCH, FRACTIONS = C['patch'], tuple(C['fractions'])
MASKERS = ('train_median','train_donor')
METRICS = ('abs_delta_logit','signed_margin_drop','predicted_class_prob_drop')
DONOR_REPEATS, RANDOM_REPEATS = C['donor_repeats'], C['random_repeats']
MATCH_SEED = C['match_seed']
RMS_REL_TOL, RMS_ABS_TOL = C['rms_relative_tolerance'], C['rms_absolute_tolerance']
MAX_CANDIDATES_PER_BUDGET, CANDIDATE_BATCH = C['max_candidates'], C['candidate_batch']
PLACEBO_SEED, PLACEBO_CANDIDATES = C['placebo_seed'], C['placebo_candidates']
PLACEBO_JSD_CALIPER = C['placebo_jsd_caliper']
PRIMARY_FLOOR = C['normalization_floor']
recipe = dict(intervention_seed=C['intervention_seed'])
timing_rows = []
cache_hash = file_sha256(cache)
old_models = pd.read_csv(ROOT/C['discovery_run']/'model_inventory.csv')


def timed_event(phase, seconds, **identity):
    row = dict(phase=phase, seconds=float(seconds), session=SESSION, **identity)
    path = OUT/'timings.jsonl'
    with path.open('a') as stream: stream.write(json.dumps(row)+'\n')


def model_for(arm, seed, fold):
    train, test = splits[fold]
    if COHORT == 'newhandpd' and seed in C['discovery_seeds']:
        row = old_models.loc[(old_models.architecture==arm)&(old_models.seed==seed)&(old_models.fold==fold)]
        if len(row) != 1: raise AlignmentError('discovery checkpoint identity is not unique')
        row = row.iloc[0]
        metadata = json.loads((ROOT/row.checkpoint).with_name('metadata.json').read_text())
        spec = metadata['spec']
        for relative,expected in spec['sources'].items():
            if file_sha256(ROOT/relative)!=expected:raise AlignmentError('discovery training source changed: '+relative)
        if spec['versions']!=VERSIONS:
            raise AlignmentError('training environment differs from discovery recipe; use a separately versioned bank')
        if (spec['architecture'] != arm or spec['seed'] != seed or spec['fold'] != fold
                or spec['cache_sha256'] != cache_hash
                or spec['model_config'] != C['model_config'] or spec['training_config'] != C['training_config']
                or spec['split']['train_indices'] != list(map(int,train))
                or spec['split']['test_indices'] != list(map(int,test))):
            raise AlignmentError('discovery checkpoint recipe/split mismatch')
        model, norm, info = get_fold_model(ROOT, spec, recordings, DEVICE, allow_training=False)
        if info['model_id'] != row.model_id or info['checkpoint_sha256'] != row.checkpoint_sha256:
            raise AlignmentError('discovery checkpoint hash mismatch')
        return model, norm, info, spec
    spec = fold_spec(ROOT, cache_hash, records, train, test, arm, seed, fold,
                     C['model_config'], C['training_config'], VERSIONS)
    spec.update(cohort=COHORT, primary_task=0 if COHORT=='uci395' else 'spirals',
                prospective_recipe_version=1)
    identity = spec_digest(spec)
    destination = MODEL_STORE/'checkpoints/e2'/identity
    if destination.exists():
        model, norm, info = get_fold_model(MODEL_STORE,spec,recordings,DEVICE,allow_training=False)
    else:
        # Publish the entire completed checkpoint directory atomically. Incomplete
        # staging folders remain quarantined and never look like reusable models.
        staging = MODEL_STORE/'staging'/(identity+'_'+SESSION)
        started = time.perf_counter()
        model, norm, info = get_fold_model(staging,spec,recordings,DEVICE,allow_training=True)
        timed_event('training',time.perf_counter()-started,architecture=arm,seed=seed,fold=fold)
        destination.parent.mkdir(parents=True,exist_ok=True)
        (staging/'checkpoints/e2'/identity).rename(destination)
        info['checkpoint'] = str(destination/'model.pt')
    return model, norm, info, spec


def prediction_path(model_id, idx):
    return OUT/'predictions'/model_id/(str(int(idx))+'.pkl')


def load_prediction(model_id, idx):
    path = prediction_path(model_id,idx)
    if not path.is_file(): raise FileNotFoundError(path)
    p=read_local_pickle(path)
    rec = records.iloc[int(idx)]
    if (p['design_id'] != DESIGN_ID or p['runtime_id'] != RUNTIME_ID
            or p['model_id'] != model_id or p['record_id'] != rec.record_id
            or p['subject'] != rec.subject or int(p['fold']) != rec.fold
            or int(p['label']) != rec.label or len(p['alpha']) != rec.n_patches):
        raise AlignmentError('prediction identity mismatch: '+str(path))
    a = np.asarray(p['alpha'],float)
    if (not np.isfinite(a).all() or (a<0).any() or not np.isclose(a.sum(),1,atol=1e-4,rtol=0)
            or not np.isfinite(p['logit'])): raise AlignmentError('invalid prediction/map')
    return p


def make_training_libraries(x, donors, rec):
    test_subjects = set(records.subject.iloc[splits[int(rec.fold)][1]])
    if any(s in test_subjects for _,s,_ in donors): raise AlignmentError('test subject in donor pool')
    libs = [dict(masker='train_median',donor_repeat=0,background=np.zeros_like(x),plans=[])]
    for d in range(DONOR_REPEATS):
        rng = np.random.default_rng(keyed_seed(C['intervention_seed'],rec.record_id,'donor',d))
        bg, plans = donor_background(x,donors,PATCH,rng,C['donor_block_patches'])
        if any(p['donor_subject'] in test_subjects for p in plans):
            raise AlignmentError('test-subject donor block')
        libs.append(dict(masker='train_donor',donor_repeat=d,background=bg,plans=plans))
    for lib in libs:
        lib['background_sha256'] = sha256(np.ascontiguousarray(lib['background']).tobytes()).hexdigest()
    return libs
'''


TRAIN_CELL = r'''
model_rows = []
with stage_lock(MODEL_STORE,'training_and_predictions'):
    manifest['completion_status'] = 'training_and_predictions'
    atomic_json(OUT/'analysis_manifest.json',manifest)
    for arm in ARMS:
        for seed in SEEDS:
            for fold,(train,test) in enumerate(splits):
                m,norm,info,spec = model_for(arm,seed,fold)
                model_rows.append(dict(architecture=arm,seed=seed,fold=fold,**info))
                atomic_csv(OUT/'model_inventory.csv',pd.DataFrame(model_rows))
                for idx in test:
                    path = prediction_path(info['model_id'],idx)
                    if path.exists(): load_prediction(info['model_id'],idx); continue
                    rec = records.iloc[int(idx)]
                    x = norm.transform(recordings[int(idx)].kanal)
                    predictor = FrozenPredictor(m,DEVICE,len(x))
                    started=time.perf_counter(); z,a=predictor.original(x)
                    timed_event('original_prediction',time.perf_counter()-started,
                                architecture=arm,seed=seed,fold=fold,idx=int(idx))
                    if len(a)!=rec.n_patches: raise AlignmentError('temporal patch grid mismatch')
                    p=dict(design_id=DESIGN_ID,runtime_id=RUNTIME_ID,model_id=info['model_id'],
                           checkpoint_sha256=info['checkpoint_sha256'],idx=int(idx),
                           record_id=rec.record_id,subject=rec.subject,label=int(rec.label),
                           task=int(rec.task),fold=int(fold),seed=int(seed),architecture=arm,
                           logit=float(z),probability=float(expit(z)),alpha=a)
                    atomic_pickle(path,p); load_prediction(info['model_id'],idx)
                del m,norm
                torch.cuda.empty_cache()
                print(COHORT,arm,'seed',seed,'fold',fold,'complete; checkpoint retained')
models=pd.DataFrame(model_rows)
if len(models)!=200 or models.duplicated(['architecture','seed','fold']).any():
    raise AlignmentError('20-seed bank incomplete; no reduced-seed success label')
atomic_csv(OUT/'model_inventory.csv',models)
model_lookup=models.set_index(['architecture','seed','fold']).model_id.to_dict()
prediction_rows=[]
for m in models.itertuples(index=False):
    for idx in splits[int(m.fold)][1]:
        p=load_prediction(m.model_id,int(idx));a=p['alpha'];positive=a[a>0]
        entropy=float(-np.sum(positive*np.log2(positive)))
        prediction_rows.append({k:v for k,v in p.items() if k!='alpha'} | dict(
            n_patches=len(a),entropy_bits=entropy,normalized_entropy=entropy/np.log2(len(a)),
            effective_support=2**entropy,max_mass=float(a.max())))
predictions=pd.DataFrame(prediction_rows)
atomic_csv(OUT/'oof_predictions.csv',predictions)
subject_predictions=(predictions.groupby(['architecture','seed','subject','label'],as_index=False)
                     .agg(probability=('probability','mean'),n_records=('idx','size')))
atomic_csv(OUT/'subject_probability_per_seed.csv',subject_predictions)
performance=[]
for (arm,seed),g in subject_predictions.groupby(['architecture','seed']):
    performance.append(dict(architecture=arm,seed=seed,n_subjects=len(g),
        auroc=roc_auc_score(g.label,g.probability)))
atomic_csv(OUT/'subject_auroc_per_seed.csv',pd.DataFrame(performance))
manifest.update(completion_status='predictions_complete',n_checkpoints=len(models),
                n_records=len(records),n_subjects=records.subject.nunique())
atomic_json(OUT/'analysis_manifest.json',manifest)
print('All 200 checkpoint identities and OOF maps are available. Next: interventions.')
'''


def cells_add(cells, kind, text):
    cells.append((nbf.v4.new_code_cell if kind == "code" else nbf.v4.new_markdown_cell)(text.strip()))


RANK_METHODS = r'''
digest = spec_digest


def paired_dose(own, other, masker):
    left={(c['requested_fraction'],c['donor_repeat']):c for c in own if c['masker']==masker}
    right={(c['requested_fraction'],c['donor_repeat']):c for c in other if c['masker']==masker}
    if set(left)!=set(right): raise AlignmentError('dose comparison lost a budget/background')
    detail=[]
    for key in sorted(left):
        a,b=left[key],right[key];sa=set(map(int,a['selected']));sb=set(map(int,b['selected']))
        gap=abs(a['target_rms']-b['target_rms'])/max(a['target_rms'],b['target_rms'],1e-8)
        detail.append(dict(requested_fraction=key[0],donor_repeat=key[1],
            own_rms=a['target_rms'],transferred_rms=b['target_rms'],relative_dose_gap=gap,
            own_run_lengths=a['target_run_lengths'],transferred_run_lengths=b['target_run_lengths'],
            identical_geometry=a['target_run_lengths']==b['target_run_lengths'],
            mask_jaccard=len(sa&sb)/len(sa|sb),n_masked=len(sa)))
    return detail


def evaluate_checkpoint_record(predictor,x,p,libraries,identity,sources):
    path=OUT/'record_evaluations'/identity['model_id']/(str(identity['idx'])+'.pkl')
    source_ids={str(seed):q['model_id'] for seed,q in sources.items()}
    context=dict(design_id=DESIGN_ID,runtime_id=RUNTIME_ID,**identity,
        source_ids=source_ids,input_sha256=sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
        backgrounds=[lib['background_sha256'] for lib in libraries])
    if path.exists():
        saved=read_local_pickle(path)
        if saved['context']!=context: raise AlignmentError('completed record cache identity mismatch')
        return saved
    progress=path.with_name(path.stem+'.progress.pkl')
    if progress.exists():
        partial=read_local_pickle(progress)
        if partial['context']!=context: raise AlignmentError('partial record identity mismatch')
        cache_components=partial['components']
    else: cache_components={}
    original,fresh=predictor.original(x)
    assert_logit(original,float(p['logit']),'original '+identity['record_id'])
    for fraction in FRACTIONS:
        k=max(1,int(np.ceil(len(fresh)*fraction)))
        old=np.sort(ranking_order(p['alpha'],identity['record_id'])[:k])
        now=np.sort(ranking_order(fresh,identity['record_id'])[:k])
        if not np.array_equal(old,now): raise AlignmentError('runtime changed an evaluated attention mask')
    alpha_delta=float(np.max(np.abs(fresh-np.asarray(p['alpha']))))

    def evaluate_rank(alpha):
        components=new_rank_components(predictor,x,p,ranking_order(alpha,identity['record_id']),
                                       libraries,identity,cache_components)
        atomic_pickle(progress,dict(context=context,components=cache_components))
        summaries,points=summarize_rank(components,{},identity)
        return {r['masker']:r for r in summaries},components,points

    own,own_components,own_points=evaluate_rank(p['alpha'])
    self_checks=[]
    for masker,r in own.items():
        d=r['G_abs_delta_logit']-r['G_abs_delta_logit']
        if r['matched_curve_complete'] and d!=0: raise AlignmentError('self-swap D is not exactly zero')
        self_checks.append(dict(**identity,masker=masker,defined=r['matched_curve_complete'],D=d))
    pairs=[];placebos=[];dose_rows=[];reference_search=[]
    entropy=lambda a: float(-np.sum(np.asarray(a)[np.asarray(a)>0]*np.log2(np.asarray(a)[np.asarray(a)>0])))
    rec=records.iloc[identity['idx']]
    for source_seed,q in sorted(sources.items()):
        if (q['record_id']!=p['record_id'] or q['fold']!=p['fold'] or q['subject']!=p['subject'] or q['task']!=p['task']
                or q['architecture']!=p['architecture'] or len(q['alpha'])!=len(p['alpha'])):
            raise AlignmentError('source/target recording, fold, architecture or grid mismatch')
        other,other_components,other_points=evaluate_rank(q['alpha'])
        pair_base=dict(**identity,target_seed=int(p['seed']),source_seed=int(source_seed),
            source_model_id=q['model_id'],js_divergence=divergence(p['alpha'],q['alpha']),
            abs_delta_logit=abs(float(p['logit'])-float(q['logit'])),
            abs_delta_prob=abs(float(p['probability'])-float(q['probability'])),
            class_agreement=(float(p['logit'])>=0)==(float(q['logit'])>=0),
            duration_seconds=float(rec.n_samples/C['fs']),n_patches=int(rec.n_patches),
            target_probability=float(p['probability']),
            confidence=float(max(p['probability'],1-p['probability'])),
            normalized_entropy=entropy(p['alpha'])/np.log2(len(p['alpha'])),
            max_mass=float(np.max(p['alpha'])),
            primary_pair=bool(COHORT=='uci395' or (p['seed']>=3 and source_seed>=3)))
        for masker in MASKERS:
            a,b=own[masker],other[masker]
            eligible=bool(a['matched_curve_complete'] and b['matched_curve_complete'])
            dose=paired_dose(own_components,other_components,masker)
            for dr in dose: dose_rows.append(dict(**pair_base,masker=masker,**dr))
            r=dict(**pair_base,masker=masker,own_complete=a['matched_curve_complete'],
                swap_complete=b['matched_curve_complete'],pair_eligible=eligible,
                eligibility_reason='complete' if eligible else 'own_control_failure' if not a['matched_curve_complete'] else 'transferred_control_failure',
                T_own=a['T_abs_delta_logit'],T_swap=b['T_abs_delta_logit'],
                R_own=a['R_abs_delta_logit'],R_swap=b['R_abs_delta_logit'],
                G_own=a['G_abs_delta_logit'],G_swap=b['G_abs_delta_logit'],
                D=a['G_abs_delta_logit']-b['G_abs_delta_logit'] if eligible else np.nan,
                D_H=a['H']-b['H'],raw_T_own_minus_swap=a['T_abs_delta_logit']-b['T_abs_delta_logit'],
                max_relative_dose_gap=max(dr['relative_dose_gap'] for dr in dose),
                mean_mask_jaccard=float(np.mean([dr['mask_jaccard'] for dr in dose])),
                mean_requested_rms=float(np.mean([dr['transferred_rms'] for dr in dose])),
                dose_matched_pair=eligible and all(dr['relative_dose_gap']<=C['dose_sensitivity_caliper'] for dr in dose))
            r['abs_D']=abs(r['D']);r['abs_D_H']=abs(r['D_H'])
            r['normalization_eligible']=eligible and np.isfinite(r['D_H'])
            r['stable_gate']=r['class_agreement'] and r['abs_delta_prob']<=.02 and r['abs_delta_logit']<=.1
            pairs.append(r)
        for kind in ('circular_shift','permutation'):
            alpha,summary,candidates=choose_placebo(p['alpha'],q['alpha'],identity,source_seed,kind)
            compact=[{k:r[k] for k in ('candidate_id','candidate_jsd','absolute_jsd_error',
                     'selected','candidate_alpha_sha256')} for r in candidates]
            reference_search.append(dict(**summary,candidates=compact))
            if alpha is not None: placebo,_,_=evaluate_rank(alpha)
            for masker in MASKERS:
                real=next(r for r in pairs if r['source_seed']==source_seed and r['masker']==masker)
                reference_complete=alpha is not None and placebo[masker]['matched_curve_complete']
                joint=real['pair_eligible'] and reference_complete
                placebos.append(dict(**pair_base,masker=masker,reference=kind,
                    real_pair_eligible=real['pair_eligible'],jsd_match_complete=bool(alpha is not None),
                    reference_pair_eligible=bool(reference_complete),joint_eligible=bool(joint),
                    jsd_absolute_error=summary['absolute_jsd_error'],
                    failure_reason='complete' if joint else 'jsd_reference_failure' if alpha is None
                        else 'real_control_failure' if not real['pair_eligible'] else 'placebo_control_failure',
                    G_real_minus_placebo=real['G_swap']-placebo[masker]['G_abs_delta_logit'] if joint else np.nan))
    assert_logit(forward_one(predictor,x),float(p['logit']),'after interventions '+identity['record_id'])
    saved=dict(context=context,pairs=pairs,placebos=placebos,dose=dose_rows,
               reference_search=reference_search,components=cache_components,self_checks=self_checks,
               alpha_max_abs_difference=alpha_delta,donor_plans=[{k:v for k,v in lib.items() if k!='background'} for lib in libraries])
    atomic_pickle(path,saved)
    if progress.exists(): progress.unlink()
    return saved
'''


INTERVENTION_CELL = r'''
pair_rows=[];placebo_rows=[];self_rows=[]
dose_partial=OUT/('swap_dose_per_background.csv.tmp.'+SESSION+'_'+str(time.time_ns()))
with stage_lock(OUT,'interventions'):
    manifest['completion_status']='interventions_running';atomic_json(OUT/'analysis_manifest.json',manifest)
    for row in models.itertuples(index=False):
        m,norm,info,_=model_for(row.architecture,int(row.seed),int(row.fold))
        if info['model_id']!=row.model_id: raise AlignmentError('model identity changed after prediction stage')
        train,test=splits[int(row.fold)]
        donor_pools={int(task):[(int(i),str(records.iloc[i].subject),norm.transform(recordings[i].kanal))
                     for i in train if int(records.iloc[i].task)==int(task)] for task in records.task.unique()}
        source_seeds=schedule.loc[(schedule.architecture==row.architecture)&(schedule.target_seed==row.seed),'source_seed'].tolist()
        if len(source_seeds)!=5 or row.seed in source_seeds: raise AlignmentError('invalid five-source schedule')
        for idx in test:
            rec=records.iloc[int(idx)];p=load_prediction(row.model_id,int(idx))
            sources={int(s):load_prediction(model_lookup[(row.architecture,int(s),int(row.fold))],int(idx)) for s in source_seeds}
            x=norm.transform(recordings[int(idx)].kanal)
            predictor=FrozenPredictor(m,DEVICE,len(x))
            libraries=make_training_libraries(x,donor_pools[int(rec.task)],rec)
            identity=dict(architecture=row.architecture,seed=int(row.seed),fold=int(row.fold),
                          model_id=row.model_id,idx=int(idx),record_id=rec.record_id,
                          subject=rec.subject,label=int(rec.label))
            saved=evaluate_checkpoint_record(predictor,x,p,libraries,identity,sources)
            pair_rows.extend(saved['pairs']);placebo_rows.extend(saved['placebos'])
            if saved['dose']:
                new_file=not dose_partial.exists()
                with dose_partial.open('a',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(saved['dose'][0]))
                    if new_file:writer.writeheader()
                    writer.writerows(saved['dose'])
            self_rows.extend(saved['self_checks'])
            if timing_rows:
                for event in timing_rows:timed_event(**event)
                timing_rows.clear()
        print(COHORT,row.architecture,'seed',row.seed,'fold',row.fold,'interventions retained')
        del m,norm,donor_pools;torch.cuda.empty_cache()
pair_curves=pd.DataFrame(pair_rows);placebo_pairs=pd.DataFrame(placebo_rows)
self_checks=pd.DataFrame(self_rows)
expected=len(records)*len(ARMS)*20*5*2
if len(pair_curves)!=expected or len(placebo_pairs)!=expected*2:
    raise AlignmentError('not all planned transfer/reference rows were exported')
keys=['architecture','target_seed','source_seed','record_id','masker']
if pair_curves.duplicated(keys).any():raise AlignmentError('duplicate transfer pair')
if not (self_checks.loc[self_checks.defined,'D']==0).all():raise AlignmentError('self-swap failed')
atomic_csv(OUT/'swap_pair_curves.csv',pair_curves)
atomic_csv(OUT/'swap_placebo_pairs.csv',placebo_pairs)
os.replace(dose_partial,OUT/'swap_dose_per_background.csv')
atomic_csv(OUT/'self_swap_checks.csv',self_checks)
manifest.update(completion_status='interventions_complete',n_planned_pair_replacement_rows=expected,
                n_exported_pair_replacement_rows=len(pair_curves))
atomic_json(OUT/'analysis_manifest.json',manifest)
print('All planned rows retained, including explicit matching failures.')
'''


STAT_METHODS = r'''
N_BOOT,CONFIDENCE=C['n_boot'],C['confidence']
subject_table=records[['subject','label']].drop_duplicates().sort_values('subject').reset_index(drop=True)
subject_index={s:i for i,s in enumerate(subject_table.subject)}
rng=np.random.default_rng(C['bootstrap_seed'])
BOOT_COUNTS=np.zeros((N_BOOT,len(subject_table)),int)
for label,g in subject_table.groupby('label'):
    indices=g.index.to_numpy()
    samples=rng.choice(indices,size=(N_BOOT,len(indices)),replace=True)
    for i,draw in enumerate(samples):BOOT_COUNTS[i]+=np.bincount(draw,minlength=len(subject_table))


def balanced_weights(g):
    if g.empty:return np.array([],float)
    return 1/(g.groupby('subject').idx.transform('nunique').to_numpy(float)
        *g.groupby(['subject','idx']).target_seed.transform('nunique').to_numpy(float)
        *g.groupby(['subject','idx','target_seed']).source_seed.transform('nunique').to_numpy(float))


def mean_stat(column):
    return lambda g,w:float(np.average(g[column].to_numpy(float),weights=w)) if w.sum()>0 else np.nan


def rank_stat(column):
    groups=None
    def statistic(g,w):
        nonlocal groups
        if w.sum()<=0:return np.nan
        if groups is None:groups=(tie_groups(g.js_divergence.to_numpy(float)),tie_groups(g[column].to_numpy(float)))
        return weighted_corr(weighted_midrank(groups[0],w),weighted_midrank(groups[1],w),w)
    return statistic


def interval(g,statistic):
    if g.empty:return dict(estimate=np.nan,ci_low=np.nan,ci_high=np.nan,p_value=np.nan,
        n_pairs=0,n_records=0,n_subjects=0,n_target_seeds=0,n_boot_valid=0,ci_status='empty_support')
    w=balanced_weights(g);estimate=statistic(g,w)
    idx=np.array([subject_index[s] for s in g.subject],int)
    draws=np.array([statistic(g,w*count[idx]) for count in BOOT_COUNTS])
    finite=np.isfinite(draws)
    allowed=(g.subject.nunique()>=C['min_ci_subjects'] and np.isfinite(estimate)
             and finite.sum()>=np.ceil(.95*N_BOOT))
    lo,hi=np.quantile(draws[finite],[.025,.975]) if allowed else (np.nan,np.nan)
    p=(1+np.count_nonzero(np.abs(draws[finite]-estimate)>=abs(estimate)))/(1+finite.sum()) if allowed else np.nan
    return dict(estimate=estimate,ci_low=lo,ci_high=hi,p_value=p,
        n_pairs=len(g),n_records=g.idx.nunique(),n_subjects=g.subject.nunique(),
        n_target_seeds=g.target_seed.nunique(),n_boot_valid=int(finite.sum()),
        ci_status='reported_conditional_subject_interval' if allowed else 'insufficient_support_or_finite_draws')


def subject_effects(g,column):
    rows=[]
    for subject,h in g.groupby('subject'):
        rows.append(dict(subject=subject,label=int(h.label.iloc[0]),
                         value=float(np.average(h[column],weights=balanced_weights(h)))))
    return pd.DataFrame(rows)


def savefig(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.svg'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.png'),dpi=200,bbox_inches='tight')
    plt.show();plt.close(fig)
'''


ANALYSIS_CELL = r'''
primary_rows=[];secondary=[];associations=[];coverage=[];seed_rows=[];loso=[];placebo_ci=[];bias=[]
continuous=['duration_seconds','n_patches','confidence','normalized_entropy','max_mass',
            'mean_requested_rms','max_relative_dose_gap','mean_mask_jaccard','label']
for (arm,masker),all_group in pair_curves.groupby(['architecture','masker']):
    group=all_group.loc[all_group.primary_pair].copy()
    g=group.loc[group.pair_eligible].copy()
    ident=dict(cohort=COHORT,architecture=arm,masker=masker,analysis_set='new_seed_replication')
    primary_rows.append(dict(**ident,endpoint='D',**interval(g,mean_stat('D'))))
    coverage.append(dict(**ident,n_pairs_planned=len(group),n_pairs_eligible=len(g),
        n_records_planned=group.idx.nunique(),n_records_eligible=g.idx.nunique(),
        n_subjects_planned=group.subject.nunique(),n_subjects_eligible=g.subject.nunique(),
        n_targets_planned=group.target_seed.nunique(),n_targets_eligible=g.target_seed.nunique(),
        n_stable_planned=int(group.stable_gate.sum()),n_stable_eligible=int(g.stable_gate.sum()),
        n_dose_eligible=int(g.dose_matched_pair.sum()),
        own_failures=int((~group.own_complete).sum()),
        transferred_failures=int((~group.swap_complete).sum())))
    for scenario,subset in [('all_primary',g),('dose_gap_at_most_5percent',g.loc[g.dose_matched_pair]),
                            ('stable_output',g.loc[g.stable_gate]),
                            ('pooled_20_seed_sensitivity',all_group.loc[all_group.pair_eligible])]:
        for endpoint in ['G_own','G_swap','D','D_H','raw_T_own_minus_swap']:
            valid=subset.loc[np.isfinite(subset[endpoint])].copy()
            secondary.append(dict(**ident,scenario=scenario,endpoint=endpoint,
                                  **interval(valid,mean_stat(endpoint))))
    for endpoint in ('D','abs_D','D_H','abs_D_H'):
        valid=g.loc[np.isfinite(g[endpoint])].copy()
        associations.append(dict(**ident,endpoint=endpoint,estimator='weighted_spearman',
                                  **interval(valid,rank_stat(endpoint))))
    for seed,h in g.groupby('target_seed'):
        for endpoint in ('D','D_H','G_swap'):
            valid=h.loc[np.isfinite(h[endpoint])]
            seed_rows.append(dict(**ident,target_seed=int(seed),endpoint=endpoint,
                estimate=mean_stat(endpoint)(valid,balanced_weights(valid)),
                n_pairs=len(valid),n_subjects=valid.subject.nunique()))
    for seed in sorted(set(group.target_seed)|set(group.source_seed)):
        h=g.loc[(g.target_seed!=seed)&(g.source_seed!=seed)]
        loso.append(dict(**ident,dropped_seed=int(seed),endpoint='D',
            estimate=mean_stat('D')(h,balanced_weights(h)),n_pairs=len(h),
            n_subjects=h.subject.nunique(),interpretation='influence_not_seed_population_CI'))
    eligible=group.loc[group.pair_eligible];excluded=group.loc[~group.pair_eligible]
    for column in continuous:
        out=dict(**ident,variable=column,n_eligible_pairs=len(eligible),n_excluded_pairs=len(excluded))
        moments=[]
        for tag,h in [('eligible',eligible),('excluded',excluded)]:
            valid=h.loc[np.isfinite(h[column])];w=balanced_weights(valid)
            mean=float(np.average(valid[column],weights=w)) if w.sum()>0 else np.nan
            var=float(np.average((valid[column]-mean)**2,weights=w)) if w.sum()>0 else np.nan
            out[tag+'_mean']=mean;out[tag+'_subjects']=valid.subject.nunique();moments.append((mean,var))
        scale=np.sqrt((moments[0][1]+moments[1][1])/2)
        out['standardized_mean_difference']=(moments[0][0]-moments[1][0])/scale if scale>0 else np.nan
        bias.append(out)
    for kind in ('circular_shift','permutation'):
        ref=placebo_pairs.loc[(placebo_pairs.architecture==arm)&(placebo_pairs.masker==masker)
            &(placebo_pairs.reference==kind)&placebo_pairs.primary_pair]
        valid=ref.loc[ref.joint_eligible].copy()
        placebo_ci.append(dict(**ident,reference=kind,n_pairs_planned=len(ref),
            n_jsd_matched=int(ref.jsd_match_complete.sum()),
            n_real_complete=int(ref.real_pair_eligible.sum()),
            **interval(valid,mean_stat('G_real_minus_placebo'))))

tables={'primary_mean_D_unadjusted.csv':pd.DataFrame(primary_rows),
    'secondary_mean_contrasts.csv':pd.DataFrame(secondary),'jsd_associations.csv':pd.DataFrame(associations),
    'matching_coverage.csv':pd.DataFrame(coverage),'target_seed_effects.csv':pd.DataFrame(seed_rows),
    'leave_one_seed_out.csv':pd.DataFrame(loso),'placebo_contrasts.csv':pd.DataFrame(placebo_ci),
    'eligible_excluded_characteristics.csv':pd.DataFrame(bias)}

# Replacement contrasts are paired on the exact same directed pair identities.
pair_keys=['architecture','target_seed','source_seed','fold','model_id','source_model_id','idx','record_id','subject','label']
left=pair_curves.loc[pair_curves.primary_pair&(pair_curves.masker=='train_median')]
right=pair_curves.loc[pair_curves.primary_pair&(pair_curves.masker=='train_donor')]
common=left.merge(right,on=pair_keys,how='outer',suffixes=('_median','_donor'),validate='one_to_one',indicator=True)
if not (common['_merge']=='both').all():raise AlignmentError('replacement planned support mismatch')
common['D_median_minus_donor']=common.D_median-common.D_donor
common['D_H_median_minus_donor']=common.D_H_median-common.D_H_donor
replacement=[]
for arm,g in common.groupby('architecture'):
    for endpoint in ('D_median_minus_donor','D_H_median_minus_donor'):
        valid=g.loc[g.pair_eligible_median&g.pair_eligible_donor&np.isfinite(g[endpoint])]
        replacement.append(dict(cohort=COHORT,architecture=arm,endpoint=endpoint,
                                 **interval(valid,mean_stat(endpoint))))
tables['replacement_common_support_contrasts.csv']=pd.DataFrame(replacement)

# Comparison with discovery: paired SUBJECT effects on intersecting subjects.
discovery_comparisons=[]
if COHORT=='newhandpd':
    old=pd.read_csv(ROOT/C['discovery_swap']/'swap_pair_curves.csv')
    old['pair_eligible']=old.pair_eligible.astype(str).str.lower().map({'true':True,'false':False})
    for (arm,masker),g in pair_curves.loc[pair_curves.primary_pair&pair_curves.pair_eligible].groupby(['architecture','masker']):
        previous=old.loc[(old.architecture==arm)&(old.masker==masker)&old.pair_eligible].copy()
        paired=subject_effects(g,'D').merge(subject_effects(previous,'D'),on=['subject','label'],
                    suffixes=('_new','_discovery'),how='inner',validate='one_to_one')
        paired['difference']=paired.value_new-paired.value_discovery
        values=paired.difference.to_numpy(float)
        indices=np.array([subject_index[s] for s in paired.subject],int)
        draws=np.array([np.average(values,weights=count[indices]) if count[indices].sum()>0 else np.nan for count in BOOT_COUNTS])
        good=draws[np.isfinite(draws)]
        allowed=len(paired)>=C['min_ci_subjects'] and len(good)>=.95*N_BOOT
        lo,hi=np.quantile(good,[.025,.975]) if allowed else (np.nan,np.nan)
        discovery_comparisons.append(dict(cohort=COHORT,architecture=arm,masker=masker,
            estimate=float(np.mean(values)) if len(values) else np.nan,ci_low=lo,ci_high=hi,
            n_subjects_common=len(paired),n_subjects_new=g.subject.nunique(),
            n_subjects_discovery=previous.subject.nunique(),
            interpretation='conditional bank/schedule comparison; not an equivalence test'))
tables['new_minus_discovery_subject_contrast.csv']=pd.DataFrame(discovery_comparisons)
for name,table in tables.items():atomic_csv(OUT/name,table)
display(tables['primary_mean_D_unadjusted.csv']);display(tables['matching_coverage.csv'])
print('Primary p-values are unadjusted here. Notebook 24 applies the fixed eight-test Holm family.')
'''


PLOT_FINISH_CELL = r'''
labels={'gru':'BiGRU','mamba2':'BiMamba-2'}
colors={'train_median':'#785194','train_donor':'#28776b'}
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for j,endpoint in enumerate(('D','G_swap')):
    table=tables['primary_mean_D_unadjusted.csv'] if endpoint=='D' else tables['secondary_mean_contrasts.csv']
    if endpoint!='D':table=table.loc[(table.endpoint==endpoint)&(table.scenario=='all_primary')]
    for i,r in enumerate(table.itertuples(index=False)):
        axes[j].plot(r.estimate,i,'o',color=colors[r.masker])
        if np.isfinite(r.ci_low):axes[j].plot([r.ci_low,r.ci_high],[i,i],color=colors[r.masker])
    axes[j].set_yticks(range(len(table)),[labels[r.architecture]+' / '+r.masker.replace('train_','') for r in table.itertuples(index=False)])
    axes[j].axvline(0,color='.5',ls=':');axes[j].set_xlabel('Logit-AOPC advantage (nominal subject CI)')
    axes[j].set_title('Own minus transferred advantage' if endpoint=='D' else 'Transferred advantage over matched control')
savefig(fig,'ranking_transfer_replication')
fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
for i,arm in enumerate(ARMS):
    for j,masker in enumerate(MASKERS):
        g=pair_curves.loc[pair_curves.primary_pair&pair_curves.pair_eligible&(pair_curves.architecture==arm)&(pair_curves.masker==masker)]
        axes[i,j].scatter(g.js_divergence,g.abs_D,s=5,alpha=.12,rasterized=True,color=colors[masker])
        axes[i,j].set(xlim=(0,1),xlabel='JSD (base-2 divergence)',ylabel='Absolute own−transferred advantage',
                      title=labels[arm]+' / '+masker.replace('train_',''))
savefig(fig,'jsd_absolute_swap_replication')
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
cov=tables['matching_coverage.csv']
for i,r in enumerate(cov.itertuples(index=False)):
    axes[0].barh(i,100*r.n_pairs_eligible/r.n_pairs_planned,color=colors[r.masker])
    axes[1].barh(i,100*r.n_stable_eligible/r.n_pairs_planned,color=colors[r.masker])
for ax in axes:ax.set_yticks(range(len(cov)),[labels[r.architecture]+' / '+r.masker.replace('train_','') for r in cov.itertuples(index=False)])
axes[0].set(xlim=(0,100),xlabel='Percent of planned primary pairs',title='Complete matching')
max_stable=max((100*r.n_stable_eligible/r.n_pairs_planned for r in cov.itertuples(index=False)),default=0)
axes[1].set(xlim=(0,max(5,max_stable*1.3)),xlabel='Percent of planned primary pairs',title='Stable-output subset: separate scale')
savefig(fig,'matching_and_stable_coverage')

# Finish only after every fixed target was trained/loaded and every planned row exists.
for relative,expected in LOCK['inputs'].items():
    if file_sha256(ROOT/relative)!=expected:raise AlignmentError('protected input/source changed: '+relative)
if file_sha256(PLAN/'PROTOCOL.md')!=LOCK['protocol_sha256']:raise AlignmentError('protocol changed mid-run')
if notebook_code_digest(NOTEBOOK)!=SCIENTIFIC_SOURCE_DIGEST:
    raise AlignmentError('notebook scientific code changed during execution')
alignment=pd.DataFrame([
    dict(check='exact_200_checkpoint_identities',passed=len(models)==200 and not models.duplicated(['architecture','seed','fold']).any()),
    dict(check='one_OOF_map_per_architecture_seed_record',passed=len(predictions)==len(records)*40 and not predictions.duplicated(['architecture','seed','record_id']).any()),
    dict(check='all_planned_pair_replacement_rows_retained',passed=len(pair_curves)==expected and not pair_curves.duplicated(keys).any()),
    dict(check='all_defined_self_swaps_exact_zero',passed=bool((self_checks.loc[self_checks.defined,'D']==0).all())),
    dict(check='newhandpd_primary_excludes_discovery_seeds',passed=COHORT!='newhandpd' or bool(((pair_curves.loc[pair_curves.primary_pair,'target_seed']>=3)&(pair_curves.loc[pair_curves.primary_pair,'source_seed']>=3)).all())),
    dict(check='all_training_test_subject_sets_disjoint',passed=all(not(set(records.subject.iloc[a])&set(records.subject.iloc[b])) for a,b in splits)),
])
if not alignment.passed.all():raise AlignmentError('final alignment check failed')
atomic_csv(OUT/'alignment_checks.csv',alignment)
manifest.update(completion_status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
    n_checkpoints=len(models),n_records=len(records),n_subjects=records.subject.nunique(),
    primary_training_seeds=list(range(3,20)) if COHORT=='newhandpd' else list(range(20)),
    pooled_training_seeds=list(range(20)),
    inference='subject conditional on trained bank/split/schedule/matching; seed diagnostics separate',
    tables=sorted(p.name for p in OUT.glob('*.csv')),
    figures=sorted(p.name for p in FIG.glob('*')),
    scientific_source_sha256=SCIENTIFIC_SOURCE_DIGEST,
    result_sha256={p.name:file_sha256(p) for p in OUT.glob('*.csv')},
    bank_contains_new_training=True,
    new_checkpoint_target=170 if COHORT=='newhandpd' else 200,
    models_trained_this_session=int(models.trained_now.sum()),
    checkpoint_resume_unit='complete_fold; not mid_epoch')
atomic_json(OUT/'analysis_manifest.json',manifest)
atomic_json(OUT/'sessions'/('complete_'+SESSION+'.json'),manifest)
atomic_json(ROOT/'results/attention_replication20'/COHORT/'latest_complete.json',
            dict(path=str(OUT.relative_to(ROOT)),design_id=DESIGN_ID,runtime_id=RUNTIME_ID,
                 manifest_sha256=file_sha256(OUT/'analysis_manifest.json')))
print('COMPLETE:',OUT)
print('Source schedule, raw failures, nested record caches, and checkpoint identities retained.')
'''


def build_notebook(cohort):
    number=22 if cohort=='newhandpd' else 23
    filename=f'{number}_replication20_{"newhandpd" if cohort=="newhandpd" else "uci_sst"}.ipynb'
    cells=[]
    cells_add(cells,'md',f'''# {number} — Replikasi audit 20 seed: {cohort}

**Run All oleh Anda melakukan training, inferensi, intervensi dan bootstrap baru.**
Agen hanya menyiapkan notebook. Baca dan jalankan notebook 21 terlebih dahulu.
NewHandPD: 170 model baru + 30 checkpoint lama; UCI-SST: 200 model baru.
Semua fold, prediksi, komponen mask, dan rekaman yang selesai dapat dilanjutkan.
Training yang terputus di tengah epoch memulai ulang fold tersebut.

Target 20 tetap. Jangan mengganti seed/recipe/schedule setelah melihat hasil.
Pasangan terjadwal lima sumber per target, bukan semua pasangan. NewHandPD
memisahkan pasangan 17 seed baru dari sensitivitas gabungan 20 seed.
Hasil lama, manuscript dan notebook 01–20 tidak ditimpa. Registrasi publik belum ada.
''')
    cells_add(cells,'code',COMMON)
    cells_add(cells,'code',f"COHORT={cohort!r}\nNOTEBOOK=ROOT/'notebooks/{filename}'")
    cells_add(cells,'md','''## Input, runtime dan checkpoint

Cache dan sumber training dipatok. Fail-fast pada identitas yang berbeda; tidak
ada training fallback untuk checkpoint discovery. Runtime berbeda mendapat namespace
evaluasi berbeda. Dua Run All pada stage yang sama ditolak oleh process lock.
''')
    cells_add(cells,'code',BANK_METHODS)
    cells_add(cells,'md','''## Training dan prediksi OOF — tahap mahal pertama

Resep final-epoch lama dipertahankan. Statistik normalizer hanya dari training fold.
Semua 20 seed harus lengkap sebelum tahap intervensi. Prediksi dan attention asli
disimpan per rekaman/checkpoint; tabel probabilitas subjek dirata-ratakan per seed.
''')
    cells_add(cells,'code',TRAIN_CELL)
    cells_add(cells,'md','''## Kontrol dan transfer ranking pada target tetap

Implementasi pencarian kontrol dan AOPC disalin secara tekstual dari builder 17/20
yang dipatok. Matching baru dilakukan untuk setiap mask. Shared-component cache
membuat mask identik berbagi kontrol, sehingga self-swap menghasilkan D=0 persis.
Tidak ada penyesuaian caliper berdasarkan respons model.
''')
    cells_add(cells,'code',old_function_sources(17,('mask_hash','mask_runs','patch_energy',
        'geometry_candidates','matched_candidates','forward_one','assert_logit')))
    cells_add(cells,'code',old_function_sources(20,('timed_forward','ranking_order','divergence',
        'choose_placebo','output_metrics','trapezoid_score','component_context','make_component',
        'new_rank_components','summarize_rank')))
    cells_add(cells,'code',RANK_METHODS)
    cells_add(cells,'code',INTERVENTION_CELL)
    cells_add(cells,'md','''## Inferensi subjek dan diagnostik

CI bersyarat pada bank/split/jadwal. Pembobotan: subjek → rekaman → target → sumber.
Seed tambahan bukan peserta tambahan. Leave-one-seed-out menghapus seed dalam
kedua peran dan merupakan diagnostik pengaruh. Keluarga Holm delapan kontras primer
baru diselesaikan di notebook 24 setelah kedua dataset lengkap.
''')
    cells_add(cells,'code',old_function_sources(18,('weighted_corr','tie_groups','weighted_midrank')))
    cells_add(cells,'code',STAT_METHODS)
    cells_add(cells,'code',ANALYSIS_CELL)
    cells_add(cells,'code',PLOT_FINISH_CELL)
    return nbf.v4.new_notebook(cells=cells),filename


PREFLIGHT_CELL = r'''
DESIGN_REPORT=PLAN/'preflight'
DESIGN_REPORT.mkdir(exist_ok=True)
budgets=[]
for cohort in ('newhandpd','uci395'):
    records,splits=prepare_cohort(cohort)
    atomic_csv(DESIGN_REPORT/(cohort+'_record_manifest.csv'),records)
    rows=[]
    for fold,(train,test) in enumerate(splits):
        for role,indices in [('train',train),('test',test)]:
            for idx in indices:rows.append(dict(fold=fold,role=role,idx=int(idx),
                record_id=records.iloc[idx].record_id,subject=records.iloc[idx].subject))
    atomic_csv(DESIGN_REPORT/(cohort+'_fold_assignments.csv'),pd.DataFrame(rows))
    for arm in C['architectures']:
        graph=schedule_all.loc[(schedule_all.cohort==cohort)&(schedule_all.architecture==arm)]
        edges=set(zip(graph.target_seed,graph.source_seed))
        if len(graph)!=100 or not all((b,a) in edges and a!=b for a,b in edges):
            raise AlignmentError('schedule uniqueness/reciprocity failure')
        if not (graph.groupby('target_seed').size()==5).all() or not (graph.groupby('source_seed').size()==5).all():
            raise AlignmentError('schedule degree imbalance')
        primary=graph.loc[graph.primary_pair]
        # Bounds: no cache reuse, all search quotas succeed, both references pass.
        # Per rank: 4 budgets * (median 1+20 forwards + donor 5+20 forwards).
        rank_bound=1+5+2*5
        forward_bound=len(records)*20*rank_bound*4*(21+25)
        budgets.append(dict(cohort=cohort,architecture=arm,n_records=len(records),
            n_subjects=records.subject.nunique(),n_HC=records.loc[records.label==0].subject.nunique(),
            n_PD=records.loc[records.label==1].subject.nunique(),n_checkpoints=100,
            new_checkpoints=85 if cohort=='newhandpd' else 100,
            scheduled_directions=len(graph),primary_directions=len(primary),
            scheduled_pair_replacement_rows=len(records)*len(graph)*2,
            primary_pair_replacement_rows=len(records)*len(primary)*2,
            no_reuse_successful_quota_forward_upper_bound=forward_bound))
budget=pd.DataFrame(budgets)
atomic_csv(DESIGN_REPORT/'planned_budget.csv',budget)
atomic_json(DESIGN_REPORT/'completion.json',dict(completion_status='complete',
    design_id=LOCK['design_id'],protocol_sha256=LOCK['protocol_sha256'],
    training_performed=False,model_inference_performed=False,new_interventions_performed=False,
    note='Input/split/design checks only; no empirical effect, runtime or precision estimate.'))
display(budget)
print('PREFLIGHT COMPLETE. No model has run. Read the budget before notebooks 22/23.')
print('Forward counts are conservative design bounds, not actual work or elapsed time.')
'''


COMBINED_CELL = r'''
tables=[];inputs=[]
for cohort in ('newhandpd','uci395'):
    pointer=ROOT/'results/attention_replication20'/cohort/'latest_complete.json'
    if not pointer.is_file():raise FileNotFoundError('dataset has no complete run: '+str(pointer))
    ref=json.loads(pointer.read_text());directory=ROOT/ref['path']
    manifest_path=directory/'analysis_manifest.json'
    if file_sha256(manifest_path)!=ref['manifest_sha256']:raise AlignmentError('completed manifest changed')
    info=json.loads(manifest_path.read_text())
    if (info['completion_status']!='complete' or info['design_id']!=LOCK['design_id']
            or info['protocol_sha256']!=LOCK['protocol_sha256'] or info['n_checkpoints']!=200):
        raise AlignmentError('incomplete/different replication bank: '+cohort)
    models=pd.read_csv(directory/'model_inventory.csv')
    if len(models)!=200 or models.duplicated(['architecture','seed','fold']).any():
        raise AlignmentError('checkpoint inventory is not unique and complete')
    for arm in C['architectures']:
        if set(models.loc[models.architecture==arm,'seed'])!=set(range(20)):
            raise AlignmentError('missing planned seed')
    source=directory/'primary_mean_D_unadjusted.csv'
    for name in ('primary_mean_D_unadjusted.csv','model_inventory.csv','alignment_checks.csv','swap_pair_curves.csv'):
        if file_sha256(directory/name)!=info['result_sha256'][name]:raise AlignmentError('result file changed: '+name)
    if not pd.read_csv(directory/'alignment_checks.csv').passed.all():raise AlignmentError('alignment check failed')
    result=pd.read_csv(source)
    if len(result)!=4 or set(result.cohort)!={cohort}:raise AlignmentError('primary stratum denominator')
    tables.append(result);inputs.append(dict(cohort=cohort,path=str(directory.relative_to(ROOT)),
        primary_sha256=file_sha256(source),runtime_id=info['runtime_id'],
        n_subjects=info['n_subjects'],n_records=info['n_records'],
        primary_training_seeds=info['primary_training_seeds']))
combined=pd.concat(tables,ignore_index=True)
if len(combined)!=8:raise AlignmentError('eight-test family incomplete')
values=combined.p_value.to_numpy(float)
# An undefined test occupies its place conservatively as p=1 in the fixed family;
# its own reported adjusted value remains undefined.
order=np.argsort(np.where(np.isfinite(values),values,1.))
adjusted=np.maximum.accumulate((8-np.arange(8))*np.where(np.isfinite(values[order]),values[order],1.))
mapped=np.empty(8,float);mapped[order]=np.minimum(adjusted,1.)
combined['p_holm_eight_primary_tests']=np.where(np.isfinite(values),mapped,np.nan)
combined['ci_multiplicity']='nominal percentile; not simultaneous intervals'
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUT=ROOT/'results/attention_replication20/combined'/LOCK['design_id']/stamp
OUT.mkdir(parents=True,exist_ok=False)
atomic_csv(OUT/'primary_eight_strata_holm.csv',combined)
atomic_json(OUT/'analysis_manifest.json',dict(completion_status='complete',
    design_id=LOCK['design_id'],inputs=inputs,family_size=8,
    endpoint='subject-balanced own minus transferred RMS-control advantage',
    uncertainty='conditional subject intervals; no seed-population interval',
    public_preregistration=False,old_preregistration_changed=False,
    training_performed=False,model_inference_performed=False,new_interventions_performed=False))
display(combined)
print('COMPLETE:',OUT)
print('No cross-dataset pooling of raw logit-AOPC; each row remains a separate stratum.')
'''


def special_notebook(number, title, instructions, body):
    cells=[]
    cells_add(cells,'md',f'# {number} — {title}\n\n'+instructions)
    cells_add(cells,'code',COMMON)
    cells_add(cells,'code',body)
    return nbf.v4.new_notebook(cells=cells)


def validate_notebook(notebook, name):
    notebook.metadata={
        'kernelspec':{'display_name':'Skripsi (.venv)','language':'python','name':'skripsi'},
        'language_info':{'name':'python','version':'3.11.8'},
        'analysis_status':CONFIG['analysis_status'], 'executed_by_builder':False,
    }
    for index,cell in enumerate(notebook.cells):
        if cell.cell_type=='code':
            compile(cell.source,f'{name}:cell{index}','exec')
            if cell.execution_count is not None or cell.outputs:
                raise ValueError('builder must not include executed output')
    nbf.validate(notebook)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-unexecuted',action='store_true',
                        help='refresh only unexecuted implementation drafts, before final seal or research run')
    parser.add_argument('--finalize',action='store_true',help='seal the validated local design; not public registration')
    args=parser.parse_args()
    generated=[]
    preflight=special_notebook(21,'Desain dan preflight replikasi 20 seed',
        'Run All memeriksa input, fold, jadwal dan jumlah evaluasi. Tidak melakukan training, '
        'inference, intervensi atau bootstrap. Target tetap 20; registrasi publik belum dilakukan. '
        'Baca PROTOCOL.md dan README.md sebelum notebook 22/23.',PREFLIGHT_CELL)
    generated.append((preflight,'21_replication20_design_preflight.ipynb'))
    generated.extend(build_notebook(c) for c in ('newhandpd','uci395'))
    combined=special_notebook(24,'Ringkasan dua dataset dan keluarga Holm',
        'Jalankan sesudah 22 dan 23 berstatus complete. Menggunakan tabel hasil nyata; '
        'tanpa model/training/intervensi. Kedua dataset tetap dilaporkan terpisah. '
        'Delapan uji primer mean-D dikoreksi bersama; CI tetap nominal.',COMBINED_CELL)
    generated.append((combined,'24_replication20_joint_report.ipynb'))
    for notebook,name in generated:validate_notebook(notebook,name)
    paths=[Path(__file__).relative_to(ROOT),
        *[Path('scripts')/f'build_nb_{n}.py' for n in (16,17,18,19,20)],
        *[Path('src')/f for f in ('audit_e0_e1.py','audit_e2.py','e2_checkpoints.py',
            'model.py','training.py','preprocessing.py','newhandpd.py')],
        Path(CONFIG['newhandpd_cache']),Path(CONFIG['uci395_cache']),
        Path(CONFIG['discovery_run'])/'analysis_manifest.json',
        Path(CONFIG['discovery_run'])/'record_manifest.csv',
        Path(CONFIG['discovery_run'])/'model_inventory.csv',
        Path(CONFIG['discovery_swap'])/'swap_pair_curves.csv']
    pins={str(p):file_hash(ROOT/p) for p in paths}
    rows=schedule_rows()
    from io import StringIO
    stream=StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows);schedule_text=stream.getvalue()
    lock=dict(configuration=CONFIG,protocol_sha256=file_hash(PLAN/'PROTOCOL.md'),inputs=pins,
              schedule_sha256=hashlib.sha256(schedule_text.encode()).hexdigest())
    lock['design_id']=hashlib.sha256(json.dumps(lock,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    destination=PLAN/'design_lock.json'
    refreshing=False
    if destination.exists() and json.loads(destination.read_text())!=lock:
        previous=json.loads(destination.read_text())
        outputs=ROOT/'results/attention_replication20'
        if ((PLAN/'FINALIZED.json').exists() or not args.refresh_unexecuted
                or (outputs.exists() and any(outputs.rglob(previous['design_id'])))):
            raise RuntimeError('Existing design differs: use a new plan version after execution/finalization.')
        for _,name in generated:
            path=ROOT/'notebooks'/name
            if path.exists():
                old=nbf.read(path,as_version=4)
                if any(c.cell_type=='code' and (c.execution_count is not None or c.outputs) for c in old.cells):
                    raise RuntimeError('Executed notebook cannot be refreshed: '+name)
        archive=PLAN/'implementation_drafts'/previous['design_id']
        archive.mkdir(parents=True,exist_ok=True)
        (archive/'design_lock.json').write_bytes(destination.read_bytes())
        for _,name in generated:
            path=ROOT/'notebooks'/name
            if path.exists():(archive/name).write_bytes(path.read_bytes())
        refreshing=True
    (PLAN/'directed_seed_schedule.csv').write_bytes(schedule_text.encode())
    destination.write_text(json.dumps(lock,indent=2),encoding='utf-8')
    written=[]
    for notebook,name in generated:
        path=ROOT/'notebooks'/name
        if path.exists():
            old=nbf.read(path,as_version=4)
            old_code=[c.source for c in old.cells if c.cell_type=='code']
            new_code=[c.source for c in notebook.cells if c.cell_type=='code']
            if old_code!=new_code and not refreshing:
                raise RuntimeError('Existing notebook scientific code differs; preserve and version it: '+name)
            if any(c.cell_type=='code' and (c.execution_count is not None or c.outputs) for c in old.cells):
                print('Executed notebook preserved without rewrite:',path);continue
        nbf.write(notebook,path)
        written.append(dict(path=str(path.relative_to(ROOT)),cells=len(notebook.cells),
                            code_cells=sum(c.cell_type=='code' for c in notebook.cells),
                            code_compiles=True,notebook_valid=True,cells_executed=0))
        print('Generated without executing:',path)
    report=dict(design_id=lock['design_id'],generated=written,
        source_functions_extracted_without_execution=True,
        schedule_checks='400 unique nonself directed edges; balanced degree five and reciprocal directions',
        research_computation_performed=False,public_preregistration_performed=False,
        original_builders_or_notebooks_modified=False)
    (PLAN/'BUILD_CHECKS.json').write_text(json.dumps(report,indent=2))
    if args.finalize:
        (PLAN/'FINALIZED.json').write_text(json.dumps(dict(design_id=lock['design_id'],
            protocol_sha256=lock['protocol_sha256'],locally_specified_before_new_outcomes=True,
            discovery_results_already_known=True,public_preregistration=False,
            finalized_utc=datetime.now(timezone.utc).isoformat()),indent=2))
    print('Design ID:',lock['design_id'])
    print('No notebook/scientific cell, training, inference or analysis executed.')


if __name__=='__main__':main()
