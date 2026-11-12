"""Generate a separate strict-resume execution; never run scientific cells.

The sealed design, FAST builder/notebooks, checkpoints and result namespaces are
read-only parents. Recovery selects registered Triton configurations by numerical
alignment only. It never relaxes the mask gate or replaces a stored attention map.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "plans/attention_replication20_resume_20261023"
PARENT_PLAN = ROOT / "plans/attention_audit_replication20_fast_20261020"
FAST_SOURCE = ROOT / "scripts/build_nb_22_24_fast.py"
spec = importlib.util.spec_from_file_location("replication20_fast_parent", FAST_SOURCE)
fast = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fast)


SETUP = r'''
RESUME_PLAN=ROOT/'plans/attention_replication20_resume_20261023'
EXECUTION_LOCK=json.loads((RESUME_PLAN/'execution_lock.json').read_text())
EXECUTION_ID=EXECUTION_LOCK['execution_id']
if EXECUTION_ID!=spec_digest({k:v for k,v in EXECUTION_LOCK.items() if k!='execution_id'}):
    raise AlignmentError('resume execution digest mismatch')
if EXECUTION_LOCK['parent_design_id']!=LOCK['design_id']:
    raise AlignmentError('resume execution belongs to another design')
for relative,expected_hash in EXECUTION_LOCK['source_sha256'].items():
    if file_sha256(ROOT/relative)!=expected_hash:
        raise AlignmentError('resume execution source changed: '+relative)
FAST_ROOT=ROOT/'results/attention_replication20_resume'
PARENT_FAST_ROOT=ROOT/'results/attention_replication20_fast'
PARENT_FAST_EXECUTION_ID=EXECUTION_LOCK['parent_fast_execution_id']
MAX_ALIGNMENT_PROFILES=128
ALIGNMENT_REPEATS=2
'''


CACHE_METHODS = r'''
PARENT_FAST_OUT=PARENT_FAST_ROOT/COHORT/DESIGN_ID/RUNTIME_ID/PARENT_FAST_EXECUTION_ID
CACHE_PARENT_DIRS=(PARENT_FAST_OUT,LEGACY_OUT)
manifest.update(execution_patch='strict_alignment_resume_v1',
    cache_parent_directories=[str(p.relative_to(ROOT)) for p in CACHE_PARENT_DIRS],
    alignment_recovery=dict(maximum_profiles=MAX_ALIGNMENT_PROFILES,repeats=ALIGNMENT_REPEATS,
        exact_operational_masks=True,logit_atol=1e-4,logit_rtol=1e-4,
        selection='first registered configuration passing numerical alignment; no effect selection'))
atomic_json(OUT/'analysis_manifest.json',manifest)

_parent_idle_check=assert_reference_runs_idle


def assert_reference_runs_idle():
    _parent_idle_check()
    candidates=list(PARENT_FAST_ROOT.glob('*.active.lock'))
    for cohort in ('newhandpd','uci395'):
        directory=PARENT_FAST_ROOT/cohort/DESIGN_ID/RUNTIME_ID/PARENT_FAST_EXECUTION_ID
        candidates.extend(directory.glob('*.active.lock'))
    for path in candidates:
        owner=json.loads(path.read_text())
        if owner['host']!=socket.gethostname():
            raise RuntimeError('parent FAST lock belongs to another host: '+str(path))
        try:os.kill(int(owner['pid']),0)
        except ProcessLookupError:continue
        if int(owner['pid'])!=os.getpid():
            raise RuntimeError('parent FAST run is active; stop it before resume: '+str(path))


def prediction_path(model_id,idx):
    relative=Path('predictions')/model_id/(str(int(idx))+'.pkl')
    for directory in (OUT,*CACHE_PARENT_DIRS):
        candidate=directory/relative
        if candidate.is_file():return candidate
    return OUT/relative


def compatible_record_cache(path,context,partial=False):
    relative=path.relative_to(OUT)
    for directory in (OUT,*CACHE_PARENT_DIRS):
        candidate=directory/relative
        if candidate.is_file():
            saved=validate_cached_components(read_local_pickle(candidate),context,partial)
            if directory!=OUT:
                register_reuse(candidate,'partial_record' if partial else 'complete_record')
            elif saved.get('execution_id')!=EXECUTION_ID:
                raise AlignmentError('resume cache belongs to another execution version')
            return saved
    return None
'''


ALIGNMENT_METHODS = r'''
def alignment_report(observed,fresh,p,identity):
    saved=np.asarray(p['alpha'],float);fresh=np.asarray(fresh,float)
    if (saved.shape!=fresh.shape or saved.ndim!=1 or len(saved)<1
            or not np.isfinite(saved).all() or not np.isfinite(fresh).all()
            or (saved<0).any() or (fresh<0).any()
            or not np.isclose(saved.sum(),1,atol=1e-4,rtol=0)
            or not np.isclose(fresh.sum(),1,atol=1e-4,rtol=0)):
        raise AlignmentError('attention grid, finite values, sign or mass invalid')
    mismatches=[]
    for fraction in FRACTIONS:
        k=max(1,int(np.ceil(len(saved)*fraction)))
        expected=np.sort(ranking_order(saved,identity['record_id'])[:k])
        actual=np.sort(ranking_order(fresh,identity['record_id'])[:k])
        if not np.array_equal(expected,actual):
            mismatches.append(dict(fraction=float(fraction),k=k,
                removed=np.setdiff1d(expected,actual).tolist(),added=np.setdiff1d(actual,expected).tolist()))
    anchor=bool(np.isfinite(observed) and np.isclose(observed,float(p['logit']),atol=1e-4,rtol=1e-4))
    return dict(passed=anchor and not mismatches,anchor_passed=anchor,
        mask_mismatches=mismatches,saved_logit=float(p['logit']),observed_logit=float(observed),
        logit_abs_difference=float(abs(observed-float(p['logit']))),
        alpha_max_abs_difference=float(np.max(np.abs(saved-fresh))))


def alignment_event(identity,**details):
    row=dict(**identity,session=SESSION,execution_id=EXECUTION_ID,**details)
    with (OUT/'runtime_alignment_trials.jsonl').open('a') as stream:
        stream.write(json.dumps(row,default=str)+'\n')


def replay_partial_components(predictor,x,p,libraries,components):
    by_background={b['background_sha256']:b for b in libraries}
    n=0;maximum=0.
    for group in components.values():
        for component in group:
            library=by_background[component['background_sha256']]
            assert_logit(float(component['original_logit']),float(p['logit']),'cached original anchor')
            points=[(component['selected'],float(component['target_logit']))]
            points.extend((c['selected'],float(c['perturbed_logit'])) for c in component['controls'])
            for selected,expected in points:
                actual=forward_one(predictor,apply_mask(x,library['background'],np.asarray(selected,int),PATCH))
                maximum=max(maximum,abs(actual-expected));n+=1
                if not np.isclose(actual,expected,atol=1e-4,rtol=1e-4):
                    return dict(passed=False,n_points=n,max_abs_difference=maximum,
                                expected=expected,observed=float(actual))
    return dict(passed=True,n_points=n,max_abs_difference=maximum)


def runtime_snapshot(objects):
    return {name:dict(configs=list(obj.configs),cache=dict(obj.cache),
                      best_config=getattr(obj,'best_config',None)) for name,obj in objects.items()}


def restore_runtime(objects,snapshot):
    for name,obj in objects.items():
        obj.configs=list(snapshot[name]['configs'])
        obj.cache.clear();obj.cache.update(snapshot[name]['cache'])
        if snapshot[name]['best_config'] is not None:obj.best_config=snapshot[name]['best_config']
        elif hasattr(obj,'best_config'):delattr(obj,'best_config')


def alignment_candidates(registered,native,known):
    from itertools import product
    profiles=[];seen=set()
    def add(profile):
        if set(profile)!=set(registered):return
        if not all(any(config_payload(c)==profile[name] for c in registered[name]) for name in registered):return
        fingerprint=spec_digest(profile)
        if fingerprint not in seen and len(profiles)<MAX_ALIGNMENT_PROFILES:
            seen.add(fingerprint);profiles.append(profile)
    add(native)
    for profile in known:add(profile)
    # One-kernel changes first; all choices come from the installed registrations.
    for name in ('bmm','chunk_state','chunk_scan','cumsum','state_passing'):
        for config in registered[name]:add(dict(native,**{name:config_payload(config)}))
    representatives={}
    for name in ('bmm','chunk_state','chunk_scan'):
        choices=[]
        for block in (32,64):
            options=[c for c in registered[name] if c.kwargs.get('BLOCK_SIZE_K')==block]
            if options:
                choices.append(min(options,key=lambda c:(
                    (c.kwargs['BLOCK_SIZE_M']+c.kwargs['BLOCK_SIZE_N'])*c.kwargs['BLOCK_SIZE_K']*c.num_stages,
                    c.kwargs['BLOCK_SIZE_M']*c.kwargs['BLOCK_SIZE_N'],c.num_warps)))
        representatives[name]=choices
    # Fixed small-tile combinations, bounded before any study response is inspected.
    for cumsum,bmm,state,scan,passing in product(registered['cumsum'],representatives['bmm'],
            representatives['chunk_state'],representatives['chunk_scan'],registered['state_passing']):
        add(dict(cumsum=config_payload(cumsum),bmm=config_payload(bmm),chunk_state=config_payload(state),
                 chunk_scan=config_payload(scan),state_passing=config_payload(passing)))
        if len(profiles)>=MAX_ALIGNMENT_PROFILES:break
    return profiles


def known_alignment_profiles():
    result=[]
    for item in EXECUTION_LOCK['known_runtime_profiles']:
        path=ROOT/item['path']
        if file_sha256(path)!=item['sha256']:raise AlignmentError('historical runtime profile changed')
        payload=json.loads(path.read_text())
        if spec_digest(payload['spec'])!=payload['profile_id']:
            raise AlignmentError('historical runtime profile digest mismatch')
        result.append(payload['spec']['kernels'])
    return result


def profile_binding(p,x,identity,libraries,sources):
    return dict(design_id=DESIGN_ID,runtime_id=RUNTIME_ID,execution_id=EXECUTION_ID,
        model_id=identity['model_id'],record_id=identity['record_id'],idx=int(identity['idx']),
        checkpoint_sha256=CHECKPOINT_BINDINGS[identity['model_id']]['checkpoint_sha256'],
        input_sha256=sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
        alpha_sha256=sha256(np.asarray(p['alpha'],dtype='<f8').tobytes()).hexdigest(),
        saved_logit=float(p['logit']),backgrounds=[b['background_sha256'] for b in libraries],
        kernel_sources=sources,atol=1e-4,rtol=1e-4,exact_masks=True)


def check_alignment_profile(predictor,x,p,identity,libraries,components,profile_id,trial,phase):
    last=None
    for repeat in range(ALIGNMENT_REPEATS):
        observed,fresh=predictor.original(x)
        report=alignment_report(observed,fresh,p,identity)
        alignment_event(identity,phase=phase,trial=trial,repeat=repeat,profile_id=profile_id,**report)
        if not report['passed']:return None
        last=(observed,fresh)
    replay=replay_partial_components(predictor,x,p,libraries,components)
    alignment_event(identity,phase=phase+'_partial_replay',trial=trial,profile_id=profile_id,**replay)
    if not replay['passed']:return None
    observed,fresh=predictor.original(x)
    after=alignment_report(observed,fresh,p,identity)
    alignment_event(identity,phase=phase+'_after_replay',trial=trial,profile_id=profile_id,**after)
    return (observed,fresh) if after['passed'] else None


def verified_original(predictor,x,p,identity,libraries,components):
    predictor.alignment_profile_id='native_autotuned'
    profile_path=OUT/'runtime_alignment_profiles'/identity['model_id']/(str(identity['idx'])+'.json')
    objects=registered=sources=None
    if identity['architecture']=='mamba2' and profile_path.is_file():
        objects,registered,sources=runtime_kernels()
        saved=json.loads(profile_path.read_text());profile_spec=saved['spec']
        binding=profile_binding(p,x,identity,libraries,sources)
        if (saved['profile_id']!=spec_digest(profile_spec) or profile_spec['binding']!=binding):
            raise AlignmentError('saved numerical alignment profile has a different binding')
        predictor.runtime_restore=(objects,runtime_snapshot(objects))
        pin_runtime(objects,registered,profile_spec['kernels'])
        result=check_alignment_profile(predictor,x,p,identity,libraries,components,saved['profile_id'],
                                       'saved','saved_profile')
        if result is None:
            raise AlignmentError('saved alignment profile failed strict replay; inspect runtime_alignment_trials.jsonl')
        predictor.alignment_profile_id=saved['profile_id']
        return result
    observed,fresh=predictor.original(x)
    report=alignment_report(observed,fresh,p,identity)
    alignment_event(identity,phase='native',trial=0,profile_id='native_autotuned',**report)
    if report['passed']:
        replay=replay_partial_components(predictor,x,p,libraries,components)
        if replay['n_points']:
            alignment_event(identity,phase='native_partial_replay',trial=0,profile_id='native_autotuned',**replay)
        if replay['passed']:return observed,fresh
    if identity['architecture']!='mamba2':
        raise AlignmentError('reference GRU alignment failed; no Mamba kernel recovery applies')
    objects,registered,sources=runtime_kernels()
    snapshot=runtime_snapshot(objects)
    predictor.runtime_restore=(objects,snapshot)
    native={name:config_payload(obj.best_config) for name,obj in objects.items()}
    binding=profile_binding(p,x,identity,libraries,sources)
    from triton.runtime.errors import OutOfResources, PTXASError
    from triton.compiler.errors import CompileTimeAssertionFailure
    candidates=alignment_candidates(registered,native,known_alignment_profiles())
    print('Strict runtime recovery:',identity['architecture'],'seed',identity['seed'],
          'fold',identity['fold'],identity['record_id'],';',len(candidates),'bounded profiles')
    for trial,profile in enumerate(candidates,1):
        pin_runtime(objects,registered,profile)
        profile_spec=dict(format_version=1,binding=binding,kernels=profile,
            selection='first registered profile passing all exact masks, anchors and partial-point replay',
            max_profiles=MAX_ALIGNMENT_PROFILES,repeats=ALIGNMENT_REPEATS)
        profile_id=spec_digest(profile_spec)
        try:
            result=check_alignment_profile(predictor,x,p,identity,libraries,components,profile_id,trial,'recovery')
        except (OutOfResources,PTXASError,CompileTimeAssertionFailure) as error:
            alignment_event(identity,phase='resource_rejected',trial=trial,profile_id=profile_id,
                            error_type=type(error).__name__,error=str(error))
            continue
        if result is not None:
            # This numerical profile is immutable once selected; later failure stops.
            if profile_path.exists():raise AlignmentError('refusing to replace an accepted runtime profile')
            atomic_json(profile_path,dict(profile_id=profile_id,spec=profile_spec))
            predictor.alignment_profile_id=profile_id
            print('Strict alignment recovered at profile',trial,'; all stored operational masks retained.')
            return result
    raise AlignmentError('no compatible runtime among '+str(len(candidates))+
        ' bounded profiles; tolerance and stored maps retained; inspect runtime_alignment_trials.jsonl')


def release_record_runtime(predictor):
    state=getattr(predictor,'runtime_restore',None)
    if state is not None:
        restore_runtime(*state)
        predictor.runtime_restore=None


def prepare_resume_backend():
    # The parent run already used reference batch-one for both architectures.
    # No accelerated implementation or new performance claim is introduced here.
    profile=dict(completion_status='complete',execution_id=EXECUTION_ID,runtime_id=RUNTIME_ID,
        cohort=COHORT,choices={arm:dict(backend='reference_batch_one',
            reason='unchanged_reference_execution_with_strict_alignment_recovery') for arm in ARMS},
        model_batch_size=1,cuda_graph_enabled=False,model_recipe_changed=False,
        benchmark_performed=False,accelerated_backend_selected=False,
        alignment_validation='each unfinished record; early known failure probe for NewHandPD')
    atomic_json(OUT/'fast_backend_profile.json',profile)
    return profile


def early_alignment_probe():
    if COHORT!='newhandpd':return
    case=EXECUTION_LOCK['known_failure_probe']
    rows=models.loc[(models.architecture==case['architecture'])&(models.seed==case['seed'])&(models.fold==case['fold'])]
    if len(rows)!=1:raise AlignmentError('early alignment probe model identity not unique')
    row=rows.iloc[0];idx=int(case['idx']);rec=records.iloc[idx]
    if row.model_id!=case['model_id'] or rec.record_id!=case['record_id']:
        raise AlignmentError('early alignment probe identity changed')
    relative=Path('record_evaluations')/row.model_id/(str(idx)+'.pkl')
    if any((d/relative).is_file() for d in (OUT,*CACHE_PARENT_DIRS)):
        print('Known failure record already cached; full loop will authenticate and reuse it.');return
    model,norm,info,_=model_for(row.architecture,int(row.seed),int(row.fold))
    if info['trained_now']:raise AlignmentError('probe must use an available checkpoint')
    train,test=splits[int(row.fold)]
    if idx not in test:raise AlignmentError('probe recording is outside its held-out fold')
    donors=[(int(i),str(records.iloc[i].subject),norm.transform(recordings[int(i)].kanal))
            for i in train if int(records.iloc[i].task)==int(rec.task)]
    x=norm.transform(recordings[idx].kanal)
    libraries=make_training_libraries(x,donors,rec)
    p=load_prediction(row.model_id,idx)
    identity=dict(architecture=row.architecture,seed=int(row.seed),fold=int(row.fold),model_id=row.model_id,
                  idx=idx,record_id=rec.record_id,subject=rec.subject,label=int(rec.label))
    sources={str(seed):load_prediction(model_lookup[(row.architecture,int(seed),int(row.fold))],idx)['model_id']
             for seed in schedule.loc[(schedule.architecture==row.architecture)&(schedule.target_seed==row.seed),'source_seed']}
    context=dict(design_id=DESIGN_ID,runtime_id=RUNTIME_ID,**identity,source_ids=sources,
        input_sha256=sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
        backgrounds=[lib['background_sha256'] for lib in libraries])
    progress=OUT/relative.with_name(str(idx)+'.progress.pkl')
    partial=compatible_record_cache(progress,context,partial=True)
    predictor=ValidatedCasePredictor(model,DEVICE,len(x))
    try:
        observed,fresh=verified_original(predictor,x,p,identity,libraries,
                                        partial['components'] if partial is not None else {})
        report=alignment_report(observed,fresh,p,identity)
        atomic_json(OUT/'early_alignment_probe.json',dict(**identity,**report,
            alignment_profile_id=predictor.alignment_profile_id,execution_id=EXECUTION_ID))
        print('Early failing-record probe passed. Next: reuse completed records and continue remaining work.')
    finally:
        release_record_runtime(predictor)
        del predictor,model,norm
        torch.cuda.empty_cache()
'''


def changed_once(source, old, new):
    return fast.replace_once(source, old, new)


def build_notebook(cohort):
    nb, old_name = fast.build_notebook(cohort)
    name = old_name.replace("_fast.ipynb", "_resume.ipynb")
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.source = cell.source.replace(old_name, name)
    nb.cells[1].source = changed_once(nb.cells[1].source, fast.EXECUTION_SETUP.strip(), SETUP.strip())
    nb.cells[4].source += "\n\n" + CACHE_METHODS.strip()
    kernels = fast.legacy_functions(17, ("runtime_kernels", "config_payload", "pin_runtime"))
    nb.cells[10].source += "\n\n" + kernels + "\n\n" + ALIGNMENT_METHODS.strip()
    nb.cells[12].source = """with fast_gpu_stage('reference_backend_setup'):
    FAST_PROFILE=prepare_resume_backend()
manifest.update(backend_validation_pending=False,fast_backend_choices=FAST_PROFILE['choices'],
                fast_backend_profile_sha256=file_sha256(OUT/'fast_backend_profile.json'))
atomic_json(OUT/'analysis_manifest.json',manifest)
print('Reference batch-one selected; no acceleration benchmark or tolerance change.')"""
    old_gate = """    original,fresh=predictor.original(x)
    assert_logit(original,float(p['logit']),'original '+identity['record_id'])
    for fraction in FRACTIONS:
        k=max(1,int(np.ceil(len(fresh)*fraction)))
        old=np.sort(ranking_order(p['alpha'],identity['record_id'])[:k])
        now=np.sort(ranking_order(fresh,identity['record_id'])[:k])
        if not np.array_equal(old,now): raise AlignmentError('runtime changed an evaluated attention mask')"""
    nb.cells[13].source = changed_once(nb.cells[13].source, old_gate,
        "    original,fresh=verified_original(predictor,x,p,identity,libraries,cache_components)")
    nb.cells[13].source = changed_once(nb.cells[13].source,
        "execution_id=EXECUTION_ID,evaluation_backend=predictor.backend)",
        "execution_id=EXECUTION_ID,evaluation_backend=predictor.backend,\n"
        "               alignment_profile_id=getattr(predictor,'alignment_profile_id','cached'))")
    nb.cells[13].source += """

_strict_evaluate_checkpoint_record=evaluate_checkpoint_record


def evaluate_checkpoint_record(predictor,x,p,libraries,identity,sources):
    try:return _strict_evaluate_checkpoint_record(predictor,x,p,libraries,identity,sources)
    finally:release_record_runtime(predictor)
"""
    nb.cells.insert(14, nbf.v4.new_code_cell("""with fast_gpu_stage('early_alignment_probe'):
    early_alignment_probe()
"""))
    nb.cells[0].source = """# %s — Strict resume of the existing 20-seed design

Open this new notebook, select **Skripsi (.venv)**, then **Restart Kernel and Run All**.
Stop old/reference/FAST notebooks first. No original file or result is overwritten.
Completed checkpoint, prediction and record caches are read through from FAST and
reference runs. An early probe tests the known failed NewHandPD recording before
the long loop. Numerical recovery uses at most 128 registered Triton profiles and
retains exact masks at all four budgets plus atol=rtol=1e-4 for logits. Partial
cached intervention points are replayed before reuse. No map substitution, quota
reduction, record exclusion or effect-based selection is allowed. Failed recovery
stops with diagnostics. All research execution is performed by you, not the builder.
""" % (22 if cohort == "newhandpd" else 23)
    nb.cells[3].source = """## Identity and read-through caches

The scientific design and runtime identity are unchanged. A separate execution
identity and output directory record this repair. Both parent cache namespaces
remain read-only. The model store is shared by authenticated checkpoint identity.
"""
    nb.cells[7].source = """## Unchanged interventions and strict numerical recovery

The reference model receives batch one, the original precision, complete temporal
grid and original patch replacements. Patch-energy memoization, control sampling,
matching, quotas and summaries are inherited unchanged. Recovery changes only
registered forward-kernel scheduling after an alignment failure.
"""
    nb.cells[11].source = """## Runtime compatibility

No resident acceleration is selected in this version. The reference backend remains
in use. A recovered profile must reproduce all stored operational masks and logit
anchors repeatedly. Every cached partial target/control point must also replay
within the original tolerance. An accepted per-record profile is immutable; failure
when it is reused stops the run. Completed record caches need no new forward.
"""
    return nb, name


def build_combined():
    nb, _ = fast.build_combined()
    nb.cells[1].source = changed_once(nb.cells[1].source, fast.EXECUTION_SETUP.strip(), SETUP.strip())
    nb.cells[0].source = """# 24 — Joint report for the strict-resume execution

Run after both 22_resume and 23_resume have completed. The same eight primary
strata and Holm family are retained. This notebook does not run any model.
"""
    return nb, "24_replication20_joint_report_resume.ipynb"


def known_profiles():
    result=[]
    for path in sorted((ROOT / "results/e2_amplitude_backend_profiles").glob("*.json")):
        payload=json.loads(path.read_text())
        if "spec" in payload and "kernels" in payload["spec"]:
            result.append(dict(path=str(path.relative_to(ROOT)),sha256=fast.file_hash(path)))
    return result


def main():
    # Verify parents before generating anything. Never refresh or amend their locks.
    parent=json.loads((PARENT_PLAN / "execution_lock.json").read_text())
    for relative,expected in parent["source_sha256"].items():
        if fast.file_hash(ROOT / relative)!=expected:raise ValueError("Parent changed: "+relative)
    design=json.loads((fast.PARENT_PLAN / "design_lock.json").read_text())
    for relative,expected in design["inputs"].items():
        if fast.file_hash(ROOT / relative)!=expected:raise ValueError("Sealed design input changed: "+relative)
    PLAN.mkdir(parents=True,exist_ok=True)
    if not (PLAN / "EXECUTION_AMENDMENT.md").is_file():raise ValueError("Missing execution amendment")
    sources=[Path(__file__),FAST_SOURCE,ROOT / "scripts/build_nb_21_24.py",
             PARENT_PLAN / "execution_lock.json",PLAN / "EXECUTION_AMENDMENT.md"]
    lock=dict(format_version=1,parent_design_id=parent["parent_design_id"],
        parent_fast_execution_id=parent["execution_id"],execution_patch="strict_alignment_resume_v1",
        scientific_design_unchanged=True,precision_unchanged=True,model_batch_size=1,
        max_alignment_profiles=128,alignment_repeats=2,logit_atol=1e-4,logit_rtol=1e-4,
        exact_masks=True,known_runtime_profiles=known_profiles(),
        known_failure_probe=dict(architecture="mamba2",seed=7,fold=4,idx=191,
            record_id="newhandpd:P28:task1:seg0",
            model_id="5b23c95f0f6509c32f5912359abbf60eb548596d09dd3c1c28228ed7c6417c4e"),
        source_sha256={str(p.relative_to(ROOT)):fast.file_hash(p) for p in sources})
    lock["execution_id"]=hashlib.sha256(json.dumps(lock,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    path=PLAN / "execution_lock.json"
    if path.exists() and json.loads(path.read_text())!=lock:
        raise RuntimeError("Resume version already sealed; create a new version instead")
    generated=[build_notebook(c) for c in ("newhandpd","uci395")]+[build_combined()]
    for nb,name in generated:
        fast.validate_notebook(nb,name)
        nb.metadata["execution_patch"]="strict_alignment_resume_v1"
        destination=ROOT / "notebooks" / name
        if destination.exists():
            old=nbf.read(destination,as_version=4)
            if any(c.cell_type=="code" and (c.execution_count is not None or c.outputs) for c in old.cells):
                if [c.source for c in old.cells if c.cell_type=="code"]!=[c.source for c in nb.cells if c.cell_type=="code"]:
                    raise RuntimeError("Never overwrite an executed notebook: "+name)
                print("Executed resume notebook preserved:",destination);continue
        nbf.write(nb,destination)
        print("Generated without execution:",destination)
    path.write_text(json.dumps(lock,indent=2)+"\n")
    (PLAN / "BUILD_CHECKS.json").write_text(json.dumps(dict(execution_id=lock["execution_id"],
        generated=[dict(path="notebooks/"+name,cells=len(nb.cells),
                        static_compile=True,notebook_schema=True,cells_executed=0) for nb,name in generated],
        research_computation_performed=False,gpu_numerical_recovery="Pending user execution",
        original_files_modified=False),indent=2)+"\n")
    print("Execution ID:",lock["execution_id"])
    print("No training, model forward, intervention or statistical analysis executed.")


if __name__=="__main__":main()
