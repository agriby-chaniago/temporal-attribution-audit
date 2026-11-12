"""Generate versioned batch-one resident-GPU notebooks; execute no research.

The sealed scientific design and builders 21--24 are read-only parents.  This
builder changes execution and cache plumbing, never the intervention recipe.
The user, not this builder, runs all model validation, timing and study cells.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "scripts/build_nb_21_24.py"
PARENT_PLAN = ROOT / "plans/attention_audit_replication20_20261019"
PLAN = ROOT / "plans/attention_audit_replication20_fast_20261020"


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parent_constants():
    wanted = {"COMMON", "BANK_METHODS", "TRAIN_CELL", "RANK_METHODS",
              "INTERVENTION_CELL", "STAT_METHODS", "ANALYSIS_CELL",
              "PLOT_FINISH_CELL", "COMBINED_CELL"}
    result = {}
    for node in ast.parse(PARENT.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    result[target.id] = ast.literal_eval(node.value)
    if set(result) != wanted:
        raise ValueError("Parent source blocks are incomplete")
    return result


def legacy_functions(number, names):
    """Read literal notebook source in a parent builder without importing it."""
    tree = ast.parse((ROOT / f"scripts/build_nb_{number}.py").read_text())
    snippets = []
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        if fn.name == "swap_methods_source":
            snippets.append(ast.literal_eval(fn.body[0].value))
        if fn.name in {"build_notebook", "_finish_notebook"}:
            for node in fn.body:
                if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name)
                        and node.value.func.id == "code"):
                    snippets.extend(a.value for a in node.value.args
                                    if isinstance(a, ast.Constant) and isinstance(a.value, str))
    found = {}
    for source in snippets:
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                if node.name in found:
                    raise ValueError("Duplicate legacy function " + node.name)
                found[node.name] = ast.get_source_segment(source, node)
    if set(found) != set(names):
        raise ValueError(f"Missing legacy functions: {set(names) - set(found)}")
    return "\n\n".join(found[n] for n in names)


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Parent source no longer matches the reviewed execution patch: " + old[:100])
    return source.replace(old, new, 1)


EXECUTION_SETUP = r'''
FAST_PLAN=ROOT/'plans/attention_audit_replication20_fast_20261020'
EXECUTION_LOCK=json.loads((FAST_PLAN/'execution_lock.json').read_text())
EXECUTION_ID=EXECUTION_LOCK['execution_id']
if EXECUTION_ID!=spec_digest({k:v for k,v in EXECUTION_LOCK.items() if k!='execution_id'}):
    raise AlignmentError('execution lock digest mismatch')
if EXECUTION_LOCK['parent_design_id']!=LOCK['design_id']:
    raise AlignmentError('execution patch belongs to a different scientific design')
for relative,expected_hash in EXECUTION_LOCK['source_sha256'].items():
    if file_sha256(ROOT/relative)!=expected_hash:
        raise AlignmentError('versioned execution source changed: '+relative)
FAST_ROOT=ROOT/'results/attention_replication20_fast'
'''


EXECUTION_METHODS = r'''
REUSE_REGISTRY={}
CHECKPOINT_BINDINGS={}
FAST_PROFILE=None


def assert_reference_runs_idle():
    # Read-only checks: never delete, recover or write a lock in the old run.
    candidates=[]
    for cohort in ('newhandpd','uci395'):
        directory=ROOT/'results/attention_replication20'/cohort/DESIGN_ID/RUNTIME_ID
        candidates.extend(directory.glob('*.active.lock'))
        training=ROOT/'results/attention_replication20_models'/cohort/'training_and_predictions.active.lock'
        if training.is_file():candidates.append(training)
    for path in candidates:
        owner=json.loads(path.read_text())
        if owner['host']!=socket.gethostname():
            raise RuntimeError('reference lock from another host; inspect before continuing: '+str(path))
        try:os.kill(int(owner['pid']),0)
        except ProcessLookupError:continue
        if int(owner['pid'])==os.getpid():continue
        raise RuntimeError('reference notebook is still active; Interrupt it and wait before Run All fast: '+str(path))


@contextmanager
def fast_gpu_stage(stage):
    assert_reference_runs_idle()
    with stage_lock(FAST_ROOT,'single_gpu'):
        manifest['execution_stage']=stage
        yield


def register_reuse(path,kind):
    relative=str(Path(path).relative_to(ROOT))
    if relative not in REUSE_REGISTRY:
        REUSE_REGISTRY[relative]=dict(source=relative,kind=kind,sha256=file_sha256(path))


def flush_reuse_registry():
    previous={}
    path=OUT/'cache_reuse.json'
    if path.is_file():previous=json.loads(path.read_text())['artifacts']
    previous.update(REUSE_REGISTRY)
    atomic_json(path,dict(execution_id=EXECUTION_ID,parent_design_id=DESIGN_ID,
                         reference_directory=str(LEGACY_OUT.relative_to(ROOT)),artifacts=previous))


_reference_model_for=model_for


def model_for(arm,seed,fold):
    model,norm,info,spec=_reference_model_for(arm,seed,fold)
    CHECKPOINT_BINDINGS[info['model_id']]=dict(architecture=arm,seed=int(seed),fold=int(fold),
        checkpoint_sha256=info['checkpoint_sha256'])
    return model,norm,info,spec


def prediction_path(model_id,idx):
    relative=Path('predictions')/model_id/(str(int(idx))+'.pkl')
    destination=OUT/relative
    if destination.exists():return destination
    reference=LEGACY_OUT/relative
    return reference if reference.is_file() else destination


_reference_load_prediction=load_prediction


def load_prediction(model_id,idx):
    p=_reference_load_prediction(model_id,idx)
    rec=records.iloc[int(idx)]
    if int(p['idx'])!=int(idx) or int(p['task'])!=int(rec.task):
        raise AlignmentError('prediction recording/task index mismatch')
    binding=CHECKPOINT_BINDINGS.get(model_id)
    if binding is None:raise AlignmentError('prediction checkpoint has not been authenticated')
    if any(p[key]!=value for key,value in binding.items()):
        raise AlignmentError('prediction checkpoint/seed/configuration identity mismatch')
    path=prediction_path(model_id,idx)
    if path.is_relative_to(LEGACY_OUT):register_reuse(path,'OOF_prediction')
    return p


_reference_make_libraries=make_training_libraries


def make_training_libraries(x,donors,rec):
    libraries=_reference_make_libraries(x,donors,rec)
    # Identical CPU float64 expression, evaluated once per record/background.
    # Matching candidates, random keys, direct RMS verification and quotas stay unchanged.
    for library in libraries:
        value=patch_energy(x,library['background'],PATCH)
        value.setflags(write=False)
        library['_patch_energy']=value
    return libraries


def validate_cached_components(saved,context,partial):
    if saved.get('context')!=context:
        raise AlignmentError('reference/fast record cache identity mismatch')
    components=saved.get('components')
    if not isinstance(components,dict):raise AlignmentError('invalid component cache schema')
    for key,group in components.items():
        if len(group)!=1+DONOR_REPEATS:raise AlignmentError('cached budget lost a background')
        for component in group:
            if (component['model_id']!=context['model_id'] or component['idx']!=context['idx']
                    or component['record_id']!=context['record_id']
                    or component['normalized_input_sha256']!=context['input_sha256']
                    or component['background_sha256'] not in context['backgrounds']
                    or component['requested_fraction'] not in FRACTIONS
                    or key!=(component['requested_fraction'],component['target_mask_sha256'])
                    or mask_hash(component['selected'])!=component['target_mask_sha256']):
                raise AlignmentError('cached component differs from model/record/grid/input/mask identity')
            required=RANDOM_REPEATS if component['masker']=='train_median' else RANDOM_REPEATS//DONOR_REPEATS
            if len(component['controls'])!=(required if component['complete'] else 0):
                raise AlignmentError('cached control quota differs from the sealed recipe')
    if not partial:
        expected={(int(seed),masker) for seed in context['source_ids'] for masker in MASKERS}
        observed={(int(row['source_seed']),row['masker']) for row in saved['pairs']}
        if observed!=expected or len(saved['pairs'])!=len(expected):
            raise AlignmentError('cached completed record has incomplete scheduled pairs')
    return saved


def compatible_record_cache(path,context,partial=False):
    relative=path.relative_to(OUT)
    for candidate in (path,LEGACY_OUT/relative):
        if candidate.is_file():
            saved=validate_cached_components(read_local_pickle(candidate),context,partial)
            if candidate.is_relative_to(LEGACY_OUT):
                register_reuse(candidate,'partial_record' if partial else 'complete_record')
            elif saved.get('execution_id')!=EXECUTION_ID:
                raise AlignmentError('fast record cache belongs to another execution patch')
            return saved
    return None
'''


RESIDENT_METHODS = r'''
def resident_mask_input(buffer,base,background,selected,patch):
    """Replacement only: no arithmetic, padding, dtype or model-batch change."""
    if buffer.shape!=base.shape or background.shape!=base.shape or base.shape[0]!=1:
        raise AlignmentError('resident input must retain batch one and exact T/C')
    selected=np.asarray(selected,dtype=np.int64)
    w=base.shape[1]//patch
    if (selected.ndim!=1 or len(np.unique(selected))!=len(selected)
            or (len(selected) and (selected.min()<0 or selected.max()>=w))):
        raise AlignmentError('resident patch indices outside the complete temporal grid')
    buffer.copy_(base)
    if len(selected):
        indices=torch.as_tensor(selected,device=base.device,dtype=torch.long)
        shape=(w,patch,base.shape[2])
        destination=buffer[0,:w*patch].view(shape)
        source=background[0,:w*patch].view(shape)
        destination.index_copy_(0,indices,source.index_select(0,indices))
    return buffer


class ResidentBatchOne:
    def __init__(self,model,device,x,libraries):
        self.model=model
        self.length=len(x)
        self.patch=int(model.patch_embed.patch_size)
        self.base=torch.from_numpy(np.ascontiguousarray(x[None],dtype=np.float32)).to(device)
        self.backgrounds={lib['background_sha256']:torch.from_numpy(
            np.ascontiguousarray(lib['background'][None],dtype=np.float32)).to(device) for lib in libraries}
        self.buffer=torch.empty_like(self.base)
        self.lengths=torch.full((1,),self.length,device=device,dtype=torch.long)

    def mask_input(self,library,selected):
        return resident_mask_input(self.buffer,self.base,self.backgrounds[library['background_sha256']],
                                   selected,self.patch)

    def logits(self,library,masks):
        # Several sequential batch-one calls, not a batched model call.
        # Every result is retained on-device until ONE transfer at this boundary.
        torch.cuda.synchronize()
        started=time.perf_counter()
        values=[]
        with torch.inference_mode():
            for selected in masks:
                current=self.mask_input(library,selected)
                output,_=self.model(current,self.lengths)
                if output.numel()!=1:raise AlignmentError('resident forward changed the model batch')
                values.append(output.reshape(()))
            result=torch.stack(values).cpu().numpy().astype(float)
        return result,time.perf_counter()-started


def reference_mask_logits(predictor,x,library,masks):
    # Reproduce the old timed-forward boundaries for an honest wall-clock comparison.
    torch.cuda.synchronize()
    started=time.perf_counter();values=[]
    for selected in masks:
        torch.cuda.synchronize()
        value=forward_one(predictor,apply_mask(x,library['background'],selected,PATCH))
        torch.cuda.synchronize()
        values.append(value)
    return np.asarray(values,float),time.perf_counter()-started


def operational_alpha_check(fresh,p,record_id):
    if len(fresh)!=len(p['alpha']):raise AlignmentError('validation attention grid differs')
    for fraction in FRACTIONS:
        k=max(1,int(np.ceil(len(fresh)*fraction)))
        expected=np.sort(ranking_order(p['alpha'],record_id)[:k])
        observed=np.sort(ranking_order(fresh,record_id)[:k])
        if not np.array_equal(expected,observed):
            raise AlignmentError('validation changed an operational attention mask')


def own_validation_plan(x,p,libraries,identity):
    order=ranking_order(p['alpha'],identity['record_id']);plan=[]
    for fraction in FRACTIONS:
        selected=np.sort(order[:max(1,int(np.ceil(len(order)*fraction)))])
        for masker in MASKERS:
            group=[lib for lib in libraries if lib['masker']==masker]
            energies=np.stack([lib['_patch_energy'] for lib in group])
            required=[RANDOM_REPEATS if masker=='train_median' else RANDOM_REPEATS//DONOR_REPEATS]*len(group)
            rng=np.random.default_rng(keyed_seed(MATCH_SEED,identity['model_id'],identity['record_id'],
                                                masker,float(fraction),mask_hash(selected)))
            matches,diags=matched_candidates(energies,selected,x.size,required,rng,RMS_REL_TOL,RMS_ABS_TOL,
                                            MAX_CANDIDATES_PER_BUDGET,CANDIDATE_BATCH)
            for library,controls,diag in zip(group,matches,diags):
                masks=[selected]+([np.asarray(c['selected'],int) for c in controls] if diag['complete'] else [])
                plan.append((library,masks))
    return plan


def validation_signature():
    inventory=models[['architecture','seed','fold','model_id','checkpoint_sha256']].to_dict('records')
    return spec_digest(dict(execution_id=EXECUTION_ID,runtime_id=RUNTIME_ID,cohort=COHORT,
                            model_inventory=inventory,notebook_source=SCIENTIFIC_SOURCE_DIGEST))


def benchmark_resident_backend():
    """User-run numerical/performance gate; does not select any scientific result."""
    assert_reference_runs_idle()
    signature=validation_signature();path=OUT/'fast_backend_profile.json'
    if path.is_file():
        profile=json.loads(path.read_text())
        if profile['signature']!=signature or profile['completion_status']!='complete':
            raise AlignmentError('fast-backend validation profile has another identity or is incomplete')
        for name,expected_hash in profile['validation_sha256'].items():
            if file_sha256(OUT/name)!=expected_hash:raise AlignmentError('backend validation table changed')
        return profile
    validations=[];benchmarks=[];choices={}
    for arm in ARMS:
        row=models.loc[models.architecture==arm].sort_values(['seed','fold']).iloc[0]
        model,norm,info,_=model_for(arm,int(row.seed),int(row.fold))
        train,test=splits[int(row.fold)]
        pool={int(task):[(int(i),str(records.iloc[i].subject),norm.transform(recordings[i].kanal))
              for i in train if int(records.iloc[i].task)==int(task)] for task in records.task.unique()}
        held=records.iloc[test].sort_values(['n_samples','record_id'])
        probe_indices=list(dict.fromkeys([int(held.iloc[0].idx),int(held.iloc[-1].idx)]))
        for idx in probe_indices:
            rec=records.iloc[idx];p=load_prediction(row.model_id,idx)
            x=norm.transform(recordings[idx].kanal)
            libraries=make_training_libraries(x,pool[int(rec.task)],rec)
            reference=FrozenPredictor(model,DEVICE,len(x))
            original,fresh=reference.original(x)
            assert_logit(original,float(p['logit']),'validation anchor '+rec.record_id)
            operational_alpha_check(fresh,p,rec.record_id)
            resident=ResidentBatchOne(model,DEVICE,x,libraries)
            identity=dict(architecture=arm,seed=int(row.seed),fold=int(row.fold),model_id=row.model_id,
                          idx=idx,record_id=rec.record_id,subject=rec.subject,label=int(rec.label))
            plan=own_validation_plan(x,p,libraries,identity)
            # Check every distinct validation input byte-for-byte before its model call.
            exact=True;n_masks=0
            with torch.inference_mode():
                for library,masks in plan:
                    for selected in masks:
                        cpu=np.ascontiguousarray(apply_mask(x,library['background'],selected,PATCH),dtype=np.float32)
                        gpu=resident.mask_input(library,selected)[0].cpu().numpy()
                        exact=exact and np.array_equal(cpu.view(np.uint32),gpu.view(np.uint32))
                        n_masks+=1
            if not exact:raise AlignmentError('resident replacement changed input bits; no fast execution allowed')
            # Small, symmetric warm-up. Both paths keep batch one and the same dtype.
            library,masks=plan[0]
            reference_mask_logits(reference,x,library,masks[:2]);resident.logits(library,masks[:2])
            max_delta=0.;all_close=True
            for repeat in range(3):
                seconds={'reference':0.,'resident':0.};outputs={}
                for mode in (('reference','resident') if repeat%2==0 else ('resident','reference')):
                    pieces=[]
                    for library,masks in plan:
                        values,elapsed=(reference_mask_logits(reference,x,library,masks) if mode=='reference'
                                        else resident.logits(library,masks))
                        seconds[mode]+=elapsed;pieces.extend(values)
                    outputs[mode]=np.asarray(pieces,float)
                delta=np.abs(outputs['reference']-outputs['resident'])
                max_delta=max(max_delta,float(delta.max()))
                all_close=all_close and bool(np.allclose(outputs['resident'],outputs['reference'],atol=1e-4,rtol=1e-4))
                benchmarks.append(dict(**identity,repeat=repeat,n_forwards=n_masks,
                    reference_seconds=seconds['reference'],resident_seconds=seconds['resident'],
                    speedup=seconds['reference']/seconds['resident'],batch_size=1))
            after=forward_one(reference,x)
            assert_logit(after,float(p['logit']),'validation final anchor '+rec.record_id)
            validations.append(dict(**identity,n_samples=len(x),n_masks=n_masks,input_bits_identical=exact,
                logits_within_original_tolerance=all_close,max_abs_logit_difference=max_delta,
                checkpoint_sha256=info['checkpoint_sha256'],batch_size=1))
            atomic_csv(OUT/'fast_backend_validation.csv',pd.DataFrame(validations))
            atomic_csv(OUT/'fast_backend_benchmark.csv',pd.DataFrame(benchmarks))
            del resident,reference
        checks=[r for r in validations if r['architecture']==arm]
        timings=[r for r in benchmarks if r['architecture']==arm]
        speedup=sum(r['reference_seconds'] for r in timings)/sum(r['resident_seconds'] for r in timings)
        numeric=all(r['input_bits_identical'] and r['logits_within_original_tolerance'] for r in checks)
        enabled=numeric and speedup>=1.05
        choices[arm]=dict(backend='resident_batch_one' if enabled else 'reference_batch_one',
            numerical_checks_passed=numeric,measured_speedup=speedup,
            reason='validated_and_at_least_5percent_faster' if enabled else
                   'reference_fallback_numerical_check' if not numeric else 'reference_fallback_no_measured_gain')
        del model,norm,pool;torch.cuda.empty_cache()
    profile=dict(completion_status='complete',signature=signature,execution_id=EXECUTION_ID,
        runtime_id=RUNTIME_ID,cohort=COHORT,choices=choices,
        model_batch_size=1,benchmark_repeats=3,minimum_speedup=1.05,atol=1e-4,rtol=1e-4,
        cuda_graph_enabled=False,
        cuda_graph_reason='Skipped: original GRU uses lengths.detach().cpu(); Mamba-2 uses lengths.tolist(). No encoder rewrite.',
        model_recipe_changed=False,scientific_results_selected=False,
        validation_sha256={n:file_sha256(OUT/n) for n in ('fast_backend_validation.csv','fast_backend_benchmark.csv')})
    atomic_json(path,profile)
    return profile


class ValidatedCasePredictor:
    """Lazy: a complete reference cache triggers no extra forward or GPU buffers."""
    def __init__(self,model,device,length):
        self.reference=FrozenPredictor(model,device,length)
        self.model=model;self.device=device;self.length=int(length)
        self.resident=None;self.backend='reference_batch_one'

    def __call__(self,array):return self.reference(array)

    def original(self,array):return self.reference.original(array)

    def configure_record(self,x,libraries,identity,p):
        assert_reference_runs_idle()
        if FAST_PROFILE is None:raise AlignmentError('run the backend validation cell before interventions')
        choice=FAST_PROFILE['choices'][identity['architecture']]
        if choice['backend']!='resident_batch_one':return
        candidate=ResidentBatchOne(self.model,self.device,x,libraries)
        order=ranking_order(p['alpha'],identity['record_id'])
        selected=np.sort(order[:max(1,int(np.ceil(len(order)*FRACTIONS[0])))])
        comparisons=[]
        # Check unmasked output and a masked input under every background on every NEW record.
        for library in libraries:
            masks=[np.array([],dtype=int),selected]
            expected,_=reference_mask_logits(self.reference,x,library,masks)
            observed,_=candidate.logits(library,masks)
            ok=bool(np.allclose(observed,expected,atol=1e-4,rtol=1e-4))
            comparisons.append(ok)
            with (OUT/'record_backend_checks.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(**identity,masker=library['masker'],donor_repeat=library['donor_repeat'],
                    within_original_tolerance=ok,max_abs_difference=float(np.max(np.abs(observed-expected))),
                    execution_id=EXECUTION_ID))+'\n')
        if all(comparisons):
            self.resident=candidate;self.backend='resident_batch_one'
        else:
            del candidate
            # No tolerance widening, cache deletion or new matching rule.
            self.backend='reference_batch_one'

    def mask_logits(self,x,library,masks,identity,fraction):
        if self.resident is None:
            values,seconds=reference_mask_logits(self.reference,x,library,masks)
        else:values,seconds=self.resident.logits(library,masks)
        timing_rows.append(dict(**identity,phase='resident_forward_group' if self.resident is not None else 'reference_forward_group',
            masker=library['masker'],donor_repeat=library['donor_repeat'],requested_fraction=float(fraction),
            seconds=seconds,n_forwards=len(masks),batch_size=1,execution_id=EXECUTION_ID))
        return values


def new_rank_components(predictor,x,p,order,libraries,identity,component_cache):
    output=[];original=float(p['logit']);w=len(order)
    for fraction in FRACTIONS:
        selected=np.sort(order[:max(1,int(np.ceil(w*fraction)))])
        key=(float(fraction),mask_hash(selected))
        if key not in component_cache:
            components=[]
            for masker in MASKERS:
                group=[b for b in libraries if b['masker']==masker]
                energies=np.stack([b['_patch_energy'] for b in group])
                required=[RANDOM_REPEATS if masker=='train_median' else RANDOM_REPEATS//DONOR_REPEATS]*len(group)
                rng=np.random.default_rng(keyed_seed(MATCH_SEED,identity['model_id'],identity['record_id'],masker,
                                                    float(fraction),mask_hash(selected)))
                started=time.perf_counter()
                matches,diags=matched_candidates(energies,selected,x.size,required,rng,RMS_REL_TOL,RMS_ABS_TOL,
                                                MAX_CANDIDATES_PER_BUDGET,CANDIDATE_BATCH)
                timing_rows.append(dict(**identity,phase='candidate_search',masker=masker,
                    requested_fraction=float(fraction),target_mask_sha256=mask_hash(selected),
                    seconds=time.perf_counter()-started))
                for library,accepted,diagnostic in zip(group,matches,diags):
                    chosen=[np.asarray(match['selected'],int) for match in accepted] if diagnostic['complete'] else []
                    logits=predictor.mask_logits(x,library,[selected]+chosen,identity,fraction)
                    controls=[dict(repeat=repeat,selected=mask,perturbed_logit=float(logits[repeat+1]))
                              for repeat,mask in enumerate(chosen)]
                    component=make_component(identity,x,selected,fraction,library,original,float(logits[0]),
                                             controls,diagnostic,'new_matched_mask')
                    component.update(evaluation_backend=predictor.backend,execution_id=EXECUTION_ID)
                    components.append(component)
            component_cache[key]=components
        output.extend(component_cache[key])
    return output
'''


def notebook_sources(cohort):
    blocks = parent_constants()
    filename = f'{22 if cohort == "newhandpd" else 23}_replication20_{"newhandpd" if cohort == "newhandpd" else "uci_sst"}_fast.ipynb'
    common = blocks["COMMON"] + "\n" + EXECUTION_SETUP
    bank = replace_once(blocks["BANK_METHODS"],
        "OUT = ROOT/'results/attention_replication20'/COHORT/DESIGN_ID/RUNTIME_ID\nFIG = ROOT/'figures/attention_replication20'/COHORT/DESIGN_ID/RUNTIME_ID",
        "LEGACY_OUT = ROOT/'results/attention_replication20'/COHORT/DESIGN_ID/RUNTIME_ID\nOUT = FAST_ROOT/COHORT/DESIGN_ID/RUNTIME_ID/EXECUTION_ID\nFIG = ROOT/'figures/attention_replication20_fast'/COHORT/DESIGN_ID/RUNTIME_ID/EXECUTION_ID")
    bank = replace_once(bank, "atomic_json(OUT/'analysis_manifest.json', manifest)",
        "manifest.update(execution_id=EXECUTION_ID,execution_patch='resident_batch_one_v1',\n"
        "    reference_directory=str(LEGACY_OUT.relative_to(ROOT)),scientific_design_unchanged=True,\n"
        "    model_batch_size=1,cuda_graph_enabled=False,backend_validation_pending=True)\n"
        "atomic_json(OUT/'analysis_manifest.json', manifest)")
    train = replace_once(blocks["TRAIN_CELL"],
        "with stage_lock(MODEL_STORE,'training_and_predictions'):",
        "with fast_gpu_stage('training_and_predictions'), stage_lock(MODEL_STORE,'training_and_predictions'):")
    train += "\nflush_reuse_registry()\n"
    make_component = legacy_functions(20, ("make_component",))
    expression = "patch_energy(x,library['background'],PATCH)"
    if make_component.count(expression) != 2:
        raise ValueError("Reviewed energy memoization locations changed")
    make_component = make_component.replace(expression, "library['_patch_energy']")
    ranks = replace_once(blocks["RANK_METHODS"],
        "    if path.exists():\n        saved=read_local_pickle(path)\n        if saved['context']!=context: raise AlignmentError('completed record cache identity mismatch')\n        return saved\n    progress=path.with_name(path.stem+'.progress.pkl')\n    if progress.exists():\n        partial=read_local_pickle(progress)\n        if partial['context']!=context: raise AlignmentError('partial record identity mismatch')\n        cache_components=partial['components']\n    else: cache_components={}",
        "    saved=compatible_record_cache(path,context)\n    if saved is not None:return saved\n    progress=path.with_name(path.stem+'.progress.pkl')\n    partial=compatible_record_cache(progress,context,partial=True)\n    cache_components=partial['components'] if partial is not None else {}")
    ranks = replace_once(ranks,
        "    alpha_delta=float(np.max(np.abs(fresh-np.asarray(p['alpha']))))",
        "    alpha_delta=float(np.max(np.abs(fresh-np.asarray(p['alpha']))))\n    predictor.configure_record(x,libraries,identity,p)")
    ranks = replace_once(ranks, "dict(context=context,components=cache_components)",
                         "dict(context=context,components=cache_components,execution_id=EXECUTION_ID)")
    ranks = replace_once(ranks,
        "donor_plans=[{k:v for k,v in lib.items() if k!='background'} for lib in libraries])",
        "donor_plans=[{k:v for k,v in lib.items() if k not in ('background','_patch_energy')} for lib in libraries],\n"
        "               execution_id=EXECUTION_ID,evaluation_backend=predictor.backend)")
    intervention = replace_once(blocks["INTERVENTION_CELL"], "with stage_lock(OUT,'interventions'):",
                                "with fast_gpu_stage('interventions'), stage_lock(OUT,'interventions'):")
    intervention = replace_once(intervention, "predictor=FrozenPredictor(m,DEVICE,len(x))",
                                 "predictor=ValidatedCasePredictor(m,DEVICE,len(x))")
    intervention = replace_once(intervention,
        "            rec=records.iloc[int(idx)];p=load_prediction(row.model_id,int(idx))",
        "            assert_reference_runs_idle()\n            rec=records.iloc[int(idx)];p=load_prediction(row.model_id,int(idx))")
    intervention = replace_once(intervention,
        "        print(COHORT,row.architecture,'seed',row.seed,'fold',row.fold,'interventions retained')",
        "        flush_reuse_registry()\n        print(COHORT,row.architecture,'seed',row.seed,'fold',row.fold,'interventions retained; reference caches reused')")
    finish = replace_once(blocks["PLOT_FINISH_CELL"],
        "for relative,expected in LOCK['inputs'].items():\n    if file_sha256(ROOT/relative)!=expected:",
        "for relative,pinned_sha256 in LOCK['inputs'].items():\n    if file_sha256(ROOT/relative)!=pinned_sha256:")
    finish = replace_once(finish,
        "# Finish only after every fixed target was trained/loaded and every planned row exists.",
        "# Finish only after every fixed target was trained/loaded and every planned row exists.\n"
        "for relative,pinned_sha256 in EXECUTION_LOCK['source_sha256'].items():\n"
        "    if file_sha256(ROOT/relative)!=pinned_sha256:\n"
        "        raise AlignmentError('execution source changed mid-run: '+relative)")
    finish = replace_once(finish,
        "atomic_json(ROOT/'results/attention_replication20'/COHORT/'latest_complete.json',",
        "flush_reuse_registry()\natomic_json(FAST_ROOT/COHORT/'latest_complete.json',")
    finish = replace_once(finish,
        "    checkpoint_resume_unit='complete_fold; not mid_epoch')",
        "    checkpoint_resume_unit='complete_fold; not mid_epoch',execution_id=EXECUTION_ID,\n"
        "    fast_backend_profile_sha256=file_sha256(OUT/'fast_backend_profile.json'),\n"
        "    fast_backend_choices=FAST_PROFILE['choices'],backend_validation_pending=False)")
    return dict(filename=filename,common=common,bank=bank,train=train,ranks=ranks,
                intervention=intervention,finish=finish,make_component=make_component,
                stat=blocks["STAT_METHODS"],analysis=blocks["ANALYSIS_CELL"])


def add(cells, kind, source):
    factory = nbf.v4.new_code_cell if kind == "code" else nbf.v4.new_markdown_cell
    cells.append(factory(source.strip()))


def build_notebook(cohort):
    s = notebook_sources(cohort)
    number = 22 if cohort == "newhandpd" else 23
    cells = []
    add(cells, "md", f"""# {number} FAST — Replikasi 20 seed, batch satu, {cohort}

Versi eksekusi baru, bukan perubahan desain ilmiah. Checkpoint dan cache notebook
22/23 asli dibaca-saja; keluaran FAST masuk direktori terpisah. Notebook 21 tetap
prasyarat. **Interrupt notebook lama, tunggu berhenti, lalu Restart Kernel and Run
All notebook FAST ini. Jangan jalankan kedua versi bersamaan.**

20 seed, lima source per target, fold, donor, masker, budget, kuota kontrol,
matching, estimator dan bootstrap tetap. Forward model tetap batch **satu** dan
presisi semula. CUDA Graphs tidak dipaksakan: encoder asli memuat operasi CPU.

Run All oleh Anda melakukan seluruh komputasi penelitian. Builder tidak
menjalankan model, benchmark, intervensi atau statistik. Cache checkpoint lengkap,
OOF, rekaman lengkap dan komponen dari progress lama dapat dipakai kembali setelah
pemeriksaan identitas. Fold training yang belum selesai tetap mengulang fold itu.
""")
    add(cells, "code", s["common"])
    add(cells, "code", f"COHORT={cohort!r}\nNOTEBOOK=ROOT/'notebooks/{s['filename']}'")
    add(cells, "md", """## Identitas dan cache

Runtime dan design ID lama tetap diperiksa. Execution ID baru memisahkan jalur
resident-GPU dari referensi. Cache lama tidak ditulis, dipindah atau dihapus.
Hasil yang sudah tersimpan dipertahankan; tidak ada training ulang otomatis bagi
checkpoint lengkap. Lock menolak versi referensi yang masih aktif.
""")
    add(cells, "code", s["bank"] + "\n" + EXECUTION_METHODS)
    add(cells, "md", """## Checkpoint dan prediksi

Tahap ini memuat checkpoint serta OOF yang cocok. Untuk NewHandPD yang sudah
menyelesaikan training notebook 22, 200 checkpoint dipakai kembali. UCI-SST tetap
membutuhkan banknya sendiri sesuai rencana lama. Resep training tidak diubah.
""")
    add(cells, "code", s["train"])
    add(cells, "md", """## Metode identik dan eksekusi resident-GPU

Input dan enam background disimpan di GPU untuk satu rekaman. Mask hanya menyalin
nilai pengganti; tail dan grid tetap. Setiap forward tetap (1, T, 6). Hasil beberapa
forward batch-satu dikumpulkan sebelum satu transfer hasil ke CPU. Sinkronisasi
internal encoder asli tetap ada. Energi patch CPU dihitung sekali per background;
sampler, urutan RNG, verifikasi RMS langsung, geometri dan aturan gagal tidak diubah.
""")
    add(cells, "code", legacy_functions(17, ("mask_hash", "mask_runs", "patch_energy",
        "geometry_candidates", "matched_candidates", "forward_one", "assert_logit")))
    add(cells, "code", legacy_functions(20, ("ranking_order", "divergence", "choose_placebo",
        "output_metrics", "trapezoid_score", "component_context", "summarize_rank"))
        + "\n\n" + s["make_component"])
    add(cells, "code", RESIDENT_METHODS)
    add(cells, "md", """## Pemeriksaan numerik dan benchmark — dijalankan oleh Anda

Pilih rekaman terpendek dan terpanjang pada seed/fold pertama tiap arsitektur
secara deterministik. Seluruh budget, background dan kontrol own-mask yang
berhasil dicocokkan diperiksa. Input harus identik bit; logit memakai toleransi
anchor semula (atol=rtol=1e-4). Tiga pengulangan dengan urutan jalur bergantian
mengukur waktu wall-clock, termasuk input, masking dan pengambilan hasil.

Resident dipakai per arsitektur hanya jika pemeriksaan numerik lolos dan waktu
terukur setidaknya 5% lebih cepat. Jika tidak, gunakan jalur referensi batch satu.
Setiap rekaman baru mendapat pemeriksaan unmasked dan masked tambahan. Penentuan
backend ini berdasarkan kesetaraan numerik/kecepatan, bukan arah efek penelitian.
""")
    add(cells, "code", """with fast_gpu_stage('backend_validation'):
    FAST_PROFILE=benchmark_resident_backend()
manifest.update(backend_validation_pending=False,fast_backend_choices=FAST_PROFILE['choices'],
                fast_backend_profile_sha256=file_sha256(OUT/'fast_backend_profile.json'))
atomic_json(OUT/'analysis_manifest.json',manifest)
display(pd.DataFrame([dict(architecture=arm,**choice) for arm,choice in FAST_PROFILE['choices'].items()]))
print('CUDA Graphs skipped; every model call retains batch one. Next: resumable interventions.')
""")
    add(cells, "code", s["ranks"])
    add(cells, "code", s["intervention"])
    add(cells, "md", """## Inferensi dan penyajian: estimator lama dipertahankan

CI subjek, bobot hierarkis, sensitivitas, pemisahan seed baru, dan keluarga Holm
tetap. Tidak ada pengurangan kuota, source, placebo atau seed. Notebook 24 FAST
membaca kedua bank FAST lengkap. Ketidakpastian tetap bersyarat pada bank tersedia.
""")
    add(cells, "code", legacy_functions(18, ("weighted_corr", "tie_groups", "weighted_midrank")))
    add(cells, "code", s["stat"])
    add(cells, "code", s["analysis"])
    add(cells, "code", s["finish"])
    return nbf.v4.new_notebook(cells=cells), s["filename"]


def build_combined():
    blocks = parent_constants()
    body = replace_once(blocks["COMBINED_CELL"],
        "pointer=ROOT/'results/attention_replication20'/cohort/'latest_complete.json'",
        "pointer=FAST_ROOT/cohort/'latest_complete.json'")
    body = replace_once(body, "    models=pd.read_csv(directory/'model_inventory.csv')",
        "    if info.get('execution_id')!=EXECUTION_ID or info.get('backend_validation_pending',True):\n"
        "        raise AlignmentError('fast execution identity/validation incomplete: '+cohort)\n"
        "    if file_sha256(directory/'fast_backend_profile.json')!=info['fast_backend_profile_sha256']:\n"
        "        raise AlignmentError('fast backend profile changed: '+cohort)\n"
        "    models=pd.read_csv(directory/'model_inventory.csv')")
    body = replace_once(body,
        "OUT=ROOT/'results/attention_replication20/combined'/LOCK['design_id']/stamp",
        "OUT=FAST_ROOT/'combined'/LOCK['design_id']/EXECUTION_ID/stamp")
    body = replace_once(body,
        "    design_id=LOCK['design_id'],inputs=inputs,family_size=8,",
        "    design_id=LOCK['design_id'],execution_id=EXECUTION_ID,inputs=inputs,family_size=8,")
    cells = []
    add(cells, "md", """# 24 FAST — Dua dataset dan keluarga Holm

Jalankan sesudah notebook 22 FAST dan 23 FAST lengkap. Tabel hasil nyata tetap
memiliki delapan strata. Tidak menjalankan model, training atau intervensi.
Notebook dan seluruh hasil lama dipertahankan.
""")
    add(cells, "code", blocks["COMMON"] + "\n" + EXECUTION_SETUP + "\nmanifest={}\n")
    add(cells, "code", body)
    return nbf.v4.new_notebook(cells=cells), "24_replication20_joint_report_fast.ipynb"


def validate_notebook(notebook, name):
    notebook.metadata = dict(kernelspec=dict(display_name="Skripsi (.venv)", language="python", name="skripsi"),
        language_info=dict(name="python", version="3.11.8"), executed_by_builder=False,
        execution_patch="resident_batch_one_v1", scientific_design_unchanged=True)
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"{name}:cell{i}", "exec")
            if cell.execution_count is not None or cell.outputs:
                raise ValueError("Builder must not contain executed results")
    nbf.validate(notebook)


def main():
    plan = json.loads((PARENT_PLAN / "design_lock.json").read_text())
    for relative, expected in plan["inputs"].items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError("Parent sealed input changed: " + relative)
    if file_hash(PARENT_PLAN / "PROTOCOL.md") != plan["protocol_sha256"]:
        raise ValueError("Parent protocol changed")
    generated = [build_notebook(c) for c in ("newhandpd", "uci395")]
    generated.append(build_combined())
    for notebook, name in generated:
        validate_notebook(notebook, name)
    spec = dict(format_version=1, parent_design_id=plan["design_id"],
        scientific_design_unchanged=True, model_batch_size=1, precision_unchanged=True,
        cuda_graph_enabled=False, backend_policy="numerical gate then measured gain >= 1.05; otherwise reference",
        source_sha256={str(Path(__file__).relative_to(ROOT)): file_hash(__file__),
                       str(PARENT.relative_to(ROOT)): file_hash(PARENT),
                       str((PLAN / "EXECUTION_AMENDMENT.md").relative_to(ROOT)): file_hash(PLAN / "EXECUTION_AMENDMENT.md")})
    spec["execution_id"] = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    lock_path = PLAN / "execution_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != spec:
        old_id = json.loads(lock_path.read_text())["execution_id"]
        directory = ROOT / "results/attention_replication20_fast"
        if directory.exists() and any(directory.glob(f"*/*/*/{old_id}")):
            raise RuntimeError("Execution already used: create a new builder/plan version")
    for notebook, name in generated:
        path = ROOT / "notebooks" / name
        if path.exists():
            old = nbf.read(path, as_version=4)
            if any(c.cell_type == "code" and (c.execution_count is not None or c.outputs) for c in old.cells):
                if [c.source for c in old.cells if c.cell_type == "code"] != [c.source for c in notebook.cells if c.cell_type == "code"]:
                    raise RuntimeError("Never overwrite an executed notebook: " + name)
                print("Executed fast notebook preserved:", path)
                continue
        nbf.write(notebook, path)
        print("Generated without execution:", path)
    lock_path.write_text(json.dumps(spec, indent=2))
    checks = dict(execution_id=spec["execution_id"], parent_design_id=plan["design_id"],
        generated=[dict(path="notebooks/" + name, cells=len(nb.cells),
                        code_cells=sum(c.cell_type == "code" for c in nb.cells),
                        static_compile=True, notebook_schema=True, cells_executed=0) for nb, name in generated],
        research_computation_performed=False, original_builder_modified=False,
        original_notebooks_modified=False, original_results_modified=False,
        cuda_graph_static_review="Not enabled: original model host operations prevent direct capture",
        gpu_numerical_validation="Pending user execution", speedup="Not measured")
    (PLAN / "BUILD_CHECKS.json").write_text(json.dumps(checks, indent=2))
    print("Execution ID:", spec["execution_id"])
    print("No model, benchmark, intervention, training, bootstrap or research analysis executed.")


if __name__ == "__main__":
    main()
