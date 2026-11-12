"""Build the fixed-model ranking-swap notebook; never execute scientific cells."""
from build_nb_19 import (
    add_finish, add_preflight, code, inference_source, legacy_functions,
    markdown, nbf, write_notebook,
)


def swap_methods_source():
    return r'''
def timed_forward(predictor,array,phase,identity):
    torch.cuda.synchronize()
    started=time.perf_counter()
    value=forward_one(predictor,array)
    torch.cuda.synchronize()
    timing_rows.append(dict(**identity,phase=phase,seconds=time.perf_counter()-started))
    return value


def ranking_order(alpha,record_id):
    values=np.asarray(alpha,float)
    tie=np.random.default_rng(keyed_seed(recipe['intervention_seed'],record_id,'ties')).random(len(values))
    return np.lexsort((tie,-values))


def divergence(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape or a.ndim!=1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AlignmentError('map pair has incompatible grid or values')
    if (a<0).any() or (b<0).any() or a.sum()<=0 or b.sum()<=0:
        raise AlignmentError('map pair has invalid mass')
    return float(np.clip(jensenshannon(a/a.sum(),b/b.sum(),base=2)**2,0,1))


def choose_placebo(own,source,identity,source_seed,kind):
    # Candidate acceptance depends only on maps/JSD, never intervention responses.
    target=divergence(own,source);own=np.asarray(own,float);w=len(own)
    seed=keyed_seed(PLACEBO_SEED,identity['model_id'],identity['record_id'],source_seed,kind)
    rng=np.random.default_rng(seed)
    identifiers=(np.sort(rng.choice(np.arange(1,w),size=min(PLACEBO_CANDIDATES,w-1),replace=False))
                 if kind=='circular_shift' else np.arange(PLACEBO_CANDIDATES))
    best=None;rows=[]
    for candidate_id in identifiers:
        candidate=np.roll(own,int(candidate_id)) if kind=='circular_shift' else own[rng.permutation(w)]
        jsd=divergence(own,candidate);error=abs(jsd-target)
        row=dict(**identity,source_seed=int(source_seed),reference=kind,candidate_id=int(candidate_id),
                 generator_seed=int(seed),target_jsd=target,candidate_jsd=jsd,absolute_jsd_error=error,
                 candidate_alpha_sha256=sha256(np.asarray(candidate,dtype='<f8').tobytes()).hexdigest(),
                 unchanged_map=bool(np.array_equal(candidate,own)))
        rows.append(row)
        if best is None or (error,int(candidate_id))<(best[0],best[1]):
            best=(error,int(candidate_id),candidate.copy(),row)
    passed=best is not None and best[0]<=PLACEBO_JSD_CALIPER
    summary=dict(**identity,source_seed=int(source_seed),reference=kind,target_jsd=target,
        generator_seed=int(seed),n_candidates=len(rows),matching_complete=bool(passed),
        absolute_jsd_error=float(best[0]) if best is not None else np.nan,
        candidate_id=int(best[1]) if best is not None else -1,
        candidate_jsd=float(best[3]['candidate_jsd']) if best is not None else np.nan,
        failure_reason='none' if passed else 'no_candidate_within_prespecified_jsd_caliper')
    for row in rows:
        row['selected']=bool(passed and row['candidate_id']==best[1])
    return (best[2] if passed else None),summary,rows


def output_metrics(original,perturbed):
    sign=1 if original>=0 else -1
    return dict(abs_delta_logit=float(abs(original-perturbed)),
        signed_margin_drop=float(sign*(original-perturbed)),
        predicted_class_prob_drop=float(expit(sign*original)-expit(sign*perturbed)))


def trapezoid_score(values,fractions):
    values=np.asarray(values,float);fractions=np.asarray(fractions,float)
    if not np.isfinite(values).all():return np.nan
    if len(values)!=len(FRACTIONS) or np.any(np.diff(fractions)<0) or fractions[-1]<=0:
        raise AlignmentError('AOPC grid is incomplete or nonmonotone')
    return float(np.trapezoid(np.r_[0.,values],np.r_[0.,fractions])/fractions[-1])


def component_context(identity,fraction,selected,library,x):
    return dict(model_id=identity['model_id'],idx=int(identity['idx']),
        record_id=identity['record_id'],masker=library['masker'],donor_repeat=int(library['donor_repeat']),
        requested_fraction=float(fraction),target_mask_sha256=mask_hash(selected),
        normalized_input_sha256=sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
        background_sha256=library['background_sha256'],patch=PATCH)


def make_component(identity,x,selected,fraction,library,original,target_logit,controls,diagnostic,origin):
    context=component_context(identity,fraction,selected,library,x)
    lengths=mask_runs(selected)[2]
    target_rms=float(np.sqrt(patch_energy(x,library['background'],PATCH)[selected].sum()/x.size))
    complete=bool(diagnostic['complete'])
    required=RANDOM_REPEATS if library['masker']=='train_median' else RANDOM_REPEATS//DONOR_REPEATS
    if complete and len(controls)!=required:
        raise AlignmentError('complete control component does not have required quota')
    if not complete and controls:
        raise AlignmentError('incomplete component must not be reweighted from partial controls')
    for row in controls:
        chosen=np.asarray(row['selected'],int)
        rms=float(np.sqrt(patch_energy(x,library['background'],PATCH)[chosen].sum()/x.size))
        if (len(chosen)!=len(selected) or mask_runs(chosen)[2]!=lengths
                or np.array_equal(chosen,selected)
                or abs(rms-target_rms)>RMS_ABS_TOL+RMS_REL_TOL*target_rms):
            raise AlignmentError('accepted control violates budget, geometry, identity exclusion or RMS')
        row.update(mask_sha256=mask_hash(chosen),input_rms_change=rms,
                   selected_indices_json=json.dumps(chosen.tolist()),
                   **output_metrics(original,float(row['perturbed_logit'])))
    target=output_metrics(original,target_logit)
    means={m:float(np.mean([c[m] for c in controls])) if complete else np.nan for m in METRICS}
    values=np.asarray([c['abs_delta_logit'] for c in controls],float)
    component_id=digest(context)
    # Sharing this key preserves accepted-candidate reuse across donor backgrounds.
    search_id=digest({k:v for k,v in context.items() if k not in ('donor_repeat','background_sha256')})
    return dict(**identity,**{k:v for k,v in context.items() if k not in identity},component_id=component_id,
        matching_search_id=search_id,origin=origin,selected=selected.copy(),selected_indices_json=json.dumps(selected.tolist()),
        target_run_lengths=json.dumps(lengths),target_rms=target_rms,n_masked=len(selected),n_patches=len(x)//PATCH,
        actual_fraction=len(selected)/(len(x)//PATCH),original_logit=float(original),
        target_logit=float(target_logit),target_metrics=target,control_metrics=means,controls=controls,
        complete=complete,n_required=required,n_controls=len(controls),diagnostic=diagnostic,
        control_abs_delta_logit_sd=float(values.std(ddof=1)) if len(values)>1 else np.nan,
        control_abs_delta_logit_min=float(values.min()) if len(values) else np.nan,
        control_abs_delta_logit_max=float(values.max()) if len(values) else np.nan)


def replay_target(predictor,x,p,libraries,identity):
    observed,fresh=predictor.original(x)
    assert_logit(observed,float(p['logit']),'target original '+identity['record_id'])
    report=alpha_report(p['alpha'],fresh,identity['record_id'],recipe['intervention_seed'],FRACTIONS)
    if not report['alpha_masks_identical']:
        raise AlignmentError('target attention operational masks changed: '+identity['record_id'])
    own=ranking_order(p['alpha'],identity['record_id']);maximum=0.
    for fraction in FRACTIONS:
        selected=np.sort(own[:max(1,int(np.ceil(len(own)*fraction)))])
        for library in libraries:
            ref=top_refs.loc[(identity['model_id'],identity['idx'],library['masker'],
                              library['donor_repeat'],float(fraction))]
            if ref.mask_sha256!=mask_hash(selected):raise AlignmentError('baseline top mask mismatch')
            value=timed_forward(predictor,apply_mask(x,library['background'],selected,PATCH),
                                'baseline_attention_replay',identity)
            maximum=max(maximum,abs(value-float(ref.perturbed_logit)))
            assert_logit(value,float(ref.perturbed_logit),'baseline attention '+identity['record_id'])
    after=timed_forward(predictor,x,'original_anchor_replay',identity)
    assert_logit(after,float(p['logit']),'after target replay '+identity['record_id'])
    return dict(**identity,verification='strict_values' if report['alpha_values_allclose']
                else 'exact_masks_all_point_replay',baseline_replay_max_abs_diff=maximum,
                unmasked_before=observed,unmasked_after=after,
                **{k:v for k,v in report.items() if k!='masks'})


def legacy_own_components(predictor,x,p,libraries,identity):
    output={};order=ranking_order(p['alpha'],identity['record_id']);original=float(p['logit'])
    for fraction in FRACTIONS:
        selected=np.sort(order[:max(1,int(np.ceil(len(order)*fraction)))])
        components=[]
        for library in libraries:
            key=(identity['model_id'],identity['idx'],library['masker'],library['donor_repeat'],float(fraction))
            ref=top_refs.loc[key];diagnostic=old_diagnostics.loc[key]
            complete=bool(diagnostic.complete);controls=[]
            if complete:
                data=old_controls.get_group(key).sort_values('repeat')
                for old in data.itertuples(index=False):
                    chosen=np.asarray(json.loads(old.selected_indices_json),int)
                    value=timed_forward(predictor,apply_mask(x,library['background'],chosen,PATCH),
                                        'legacy_control_replay',identity)
                    assert_logit(value,float(old.perturbed_logit),'legacy RMS control '+identity['record_id'])
                    controls.append(dict(repeat=int(old.repeat),selected=chosen,
                        perturbed_logit=float(old.perturbed_logit),replay_observed_logit=value))
            elif key in old_controls.groups:
                raise AlignmentError('legacy incomplete background unexpectedly has evaluated controls')
            diag={k:diagnostic[k] for k in ('n_matches_found','candidates_drawn','search_status')}
            diag.update(complete=complete,legacy_control_id=EXPECTED_CONTROL)
            components.append(make_component(identity,x,selected,fraction,library,original,
                float(ref.perturbed_logit),controls,diag,'legacy_own_verified'))
        output[(float(fraction),mask_hash(selected))]=components
    return output


def new_rank_components(predictor,x,p,order,libraries,identity,component_cache):
    output=[];original=float(p['logit']);w=len(order)
    for fraction in FRACTIONS:
        selected=np.sort(order[:max(1,int(np.ceil(w*fraction)))])
        key=(float(fraction),mask_hash(selected))
        if key not in component_cache:
            components=[]
            for masker in MASKERS:
                background_group=[b for b in libraries if b['masker']==masker]
                energies=np.stack([patch_energy(x,b['background'],PATCH) for b in background_group])
                required=[RANDOM_REPEATS if masker=='train_median' else RANDOM_REPEATS//DONOR_REPEATS]*len(background_group)
                rng=np.random.default_rng(keyed_seed(MATCH_SEED,identity['model_id'],identity['record_id'],masker,
                                                     float(fraction),mask_hash(selected)))
                started=time.perf_counter()
                matches,diags=matched_candidates(energies,selected,x.size,required,rng,RMS_REL_TOL,RMS_ABS_TOL,
                                                MAX_CANDIDATES_PER_BUDGET,CANDIDATE_BATCH)
                timing_rows.append(dict(**identity,phase='candidate_search',masker=masker,
                    requested_fraction=float(fraction),target_mask_sha256=mask_hash(selected),
                    seconds=time.perf_counter()-started))
                for library,accepted,diagnostic in zip(background_group,matches,diags):
                    target=timed_forward(predictor,apply_mask(x,library['background'],selected,PATCH),
                                         'new_ranking_forward',identity)
                    controls=[]
                    if diagnostic['complete']:
                        for repeat,match in enumerate(accepted):
                            chosen=np.asarray(match['selected'],int)
                            value=timed_forward(predictor,apply_mask(x,library['background'],chosen,PATCH),
                                                'new_control_forward',identity)
                            controls.append(dict(repeat=repeat,selected=chosen,perturbed_logit=value))
                    components.append(make_component(identity,x,selected,fraction,library,original,target,
                                                       controls,diagnostic,'new_matched_mask'))
            component_cache[key]=components
        output.extend(component_cache[key])
    return output


def summarize_rank(components,rank_meta,identity):
    rows=[];points=[]
    for masker in MASKERS:
        bg_expected=1 if masker=='train_median' else DONOR_REPEATS
        group=[c for c in components if c['masker']==masker]
        for fraction in FRACTIONS:
            bg=[c for c in group if c['requested_fraction']==fraction]
            if len(bg)!=bg_expected or len({c['donor_repeat'] for c in bg})!=bg_expected:
                raise AlignmentError('ranking lost a required background')
            complete=all(c['complete'] for c in bg)
            point=dict(**identity,**rank_meta,masker=masker,requested_fraction=float(fraction),
                actual_fraction=bg[0]['actual_fraction'],matched_budget_complete=complete,
                n_backgrounds=len(bg),n_masked=bg[0]['n_masked'],n_patches=bg[0]['n_patches'],
                target_mask_sha256=bg[0]['target_mask_sha256'],target_run_lengths=bg[0]['target_run_lengths'],
                target_rms_mean=float(np.mean([c['target_rms'] for c in bg])),
                component_ids=json.dumps([c['component_id'] for c in bg]))
            for metric in METRICS:
                point['T_'+metric]=float(np.mean([c['target_metrics'][metric] for c in bg]))
                point['R_'+metric]=float(np.mean([c['control_metrics'][metric] for c in bg])) if complete else np.nan
            points.append(point)
        local=sorted([p for p in points if p['masker']==masker],key=lambda p:p['requested_fraction'])
        row=dict(**identity,**rank_meta,masker=masker,
                 matched_curve_complete=all(p['matched_budget_complete'] for p in local))
        for metric in METRICS:
            for prefix in ('T','R'):
                row[prefix+'_'+metric]=trapezoid_score([p[prefix+'_'+metric] for p in local],
                                                       [p['actual_fraction'] for p in local])
            row['G_'+metric]=row['T_'+metric]-row['R_'+metric]
        row['effect_sum']=row['T_abs_delta_logit']+row['R_abs_delta_logit']
        row['H']=(row['G_abs_delta_logit']/row['effect_sum']
                  if row['matched_curve_complete'] and row['effect_sum']>PRIMARY_FLOOR else np.nan)
        rows.append(row)
    return rows,points


def evaluate_target_record(predictor,x,p,libraries,identity):
    validation=replay_target(predictor,x,p,libraries,identity)
    own=np.asarray(p['alpha'],float)
    cache=legacy_own_components(predictor,x,p,libraries,identity)
    ranks=[dict(rank_id='own',kind='own',source_seed=identity['seed'],source_model_id=identity['model_id'],
                js_divergence=0.,reference='none',reference_match=True,reference_jsd_error=0.,alpha=own)]
    reference_support=[];candidate_rows=[]
    for source_seed in SEEDS:
        if source_seed==identity['seed']:continue
        source_id=model_lookup[(identity['architecture'],int(source_seed),identity['fold'])]
        source=maps[(source_id,identity['idx'])]
        alpha=np.asarray(source['alpha'],float)
        real=dict(rank_id=f'seed_{source_seed}',kind='real_swap',source_seed=int(source_seed),
            source_model_id=source_id,js_divergence=divergence(own,alpha),reference='none',
            reference_match=True,reference_jsd_error=0.,alpha=alpha)
        ranks.append(real)
        for kind in ('circular_shift','permutation'):
            shifted,support,candidates=choose_placebo(own,alpha,identity,source_seed,kind)
            reference_support.append(support);candidate_rows.extend(candidates)
            if shifted is not None:
                ranks.append(dict(rank_id=f'{kind}_for_seed_{source_seed}',kind='placebo',
                    source_seed=int(source_seed),source_model_id=source_id,
                    js_divergence=divergence(own,shifted),reference=kind,reference_match=True,
                    reference_jsd_error=support['absolute_jsd_error'],alpha=shifted))
    rank_scores=[];rank_points=[];rank_links=[]
    for rank in ranks:
        order=ranking_order(rank['alpha'],identity['record_id'])
        components=new_rank_components(predictor,x,p,order,libraries,identity,cache)
        meta={k:v for k,v in rank.items() if k!='alpha'}
        scores,points=summarize_rank(components,meta,identity)
        rank_scores.extend(scores);rank_points.extend(points)
        rank_links.extend(dict(**identity,rank_id=rank['rank_id'],component_id=c['component_id']) for c in components)
    after=timed_forward(predictor,x,'after_record_anchor',identity)
    assert_logit(after,float(p['logit']),'after record '+identity['record_id'])
    # Self-swap uses the same ranking function, masks and component plans.
    self_order=ranking_order(own.copy(),identity['record_id'])
    self_components=new_rank_components(predictor,x,p,self_order,libraries,identity,cache)
    self_scores,_=summarize_rank(self_components,{k:v for k,v in ranks[0].items() if k!='alpha'},identity)
    self_checks=[]
    for masker in MASKERS:
        a=next(s for s in rank_scores if s['kind']=='own' and s['masker']==masker)
        b=next(s for s in self_scores if s['masker']==masker)
        defined=a['matched_curve_complete']
        delta=a['G_abs_delta_logit']-b['G_abs_delta_logit'] if defined else np.nan
        if defined and delta!=0.:raise AlignmentError('self-swap must have exact zero D with reused components')
        self_checks.append(dict(**identity,masker=masker,D=delta,defined=defined,
            exact_zero_if_defined=bool(delta==0.) if defined else None,
            controls_reused=True,baseline_replay_max_abs_diff=validation['baseline_replay_max_abs_diff']))
    components={c['component_id']:c for group in cache.values() for c in group}
    return dict(identity=identity,validation=validation,rank_scores=rank_scores,rank_points=rank_points,
        rank_links=rank_links,components=components,reference_support=reference_support,
        candidate_rows=candidate_rows,self_checks=self_checks)


def cache_record(path,payload):
    from artefak import simpan_atomik
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError('record cache is immutable; refusing overwrite: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    payload['analysis_id']=ANALYSIS_ID
    simpan_atomik(path,payload)
    meta=dict(analysis_id=ANALYSIS_ID,identity=payload['identity'],payload_sha256=file_hash(path))
    path.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')


def load_record_cache(path,predictor,x,p,libraries,identity):
    meta_path=path.with_suffix('.json')
    if meta_path.is_file():
        meta=json.loads(meta_path.read_text())
        if meta['analysis_id']!=ANALYSIS_ID or meta['identity']!=identity or meta['payload_sha256']!=file_hash(path):
            raise AlignmentError('record cache identity/hash mismatch')
    payload=pickle.loads(path.read_bytes())
    if payload.get('analysis_id')!=ANALYSIS_ID or payload['identity']!=identity:
        raise AlignmentError('record cache payload identity mismatch')
    validation=replay_target(predictor,x,p,libraries,identity)
    lookup={(b['masker'],b['donor_repeat']):b for b in libraries}
    # Full replay includes new controls; cache reuse saves matching, not this check.
    for component in payload['components'].values():
        library=lookup[(component['masker'],component['donor_repeat'])]
        context=component_context(identity,component['requested_fraction'],component['selected'],library,x)
        if digest(context)!=component['component_id']:raise AlignmentError('cache component context changed')
        value=timed_forward(predictor,apply_mask(x,library['background'],component['selected'],PATCH),
                            'cache_ranking_replay',identity)
        assert_logit(value,component['target_logit'],'cached ranking '+identity['record_id'])
        for control in component['controls']:
            value=timed_forward(predictor,apply_mask(x,library['background'],control['selected'],PATCH),
                                'cache_control_replay',identity)
            assert_logit(value,control['perturbed_logit'],'cached control '+identity['record_id'])
    payload['validation']=validation
    if not meta_path.exists():
        # Recover a crash between the atomic payload write and sidecar creation.
        # Only this new namespace is touched, and only after complete replay.
        meta=dict(analysis_id=ANALYSIS_ID,identity=identity,payload_sha256=file_hash(path),
                  sidecar_recovered_after_full_replay=True)
        with meta_path.open('x') as stream:json.dump(meta,stream,indent=2)
    return payload


def stream_table(name,rows):
    if not rows:return
    path=OUT/(name+'.csv')
    frame=pd.DataFrame(rows)
    if path.exists():
        header=pd.read_csv(path,nrows=0).columns.tolist()
        if set(header)!=set(frame.columns):
            raise AlignmentError('streamed table schema changed: '+name)
        frame=frame[header]
    frame.to_csv(path,mode='a',header=not path.exists(),index=False)


def flush_timings():
    if not timing_rows:return
    columns=['architecture','seed','fold','model_id','idx','record_id','subject','label',
             'phase','masker','requested_fraction','target_mask_sha256','seconds']
    frame=pd.DataFrame(timing_rows).reindex(columns=columns)
    path=OUT/'computation_timings.csv'
    frame.to_csv(path,mode='a',header=not path.exists(),index=False)
    timing_rows.clear()


def export_record(payload):
    stream_table('target_validation',[payload['validation']])
    stream_table('rank_scores',payload['rank_scores'])
    stream_table('rank_points',payload['rank_points'])
    stream_table('rank_component_links',payload['rank_links'])
    stream_table('placebo_matching',payload['reference_support'])
    stream_table('placebo_candidates',payload['candidate_rows'])
    stream_table('self_swap_checks',payload['self_checks'])
    component_rows=[];control_rows=[]
    for c in payload['components'].values():
        base={k:v for k,v in c.items() if k not in ('selected','controls','target_metrics','control_metrics','diagnostic')}
        base.update({'target_'+k:v for k,v in c['target_metrics'].items()})
        base.update({'control_'+k:v for k,v in c['control_metrics'].items()})
        base.update(search_candidates_drawn=c['diagnostic'].get('candidates_drawn',np.nan),
                    search_matches_found=c['diagnostic'].get('n_matches_found',np.nan),
                    search_status=c['diagnostic'].get('search_status','unknown'),
                    diagnostic_json=json.dumps(c['diagnostic'],default=str,sort_keys=True))
        component_rows.append(base)
        for control in c['controls']:
            row={k:v for k,v in control.items() if k!='selected'}
            row.setdefault('replay_observed_logit',np.nan)
            control_rows.append(dict(component_id=c['component_id'],matching_search_id=c['matching_search_id'],
                model_id=c['model_id'],idx=c['idx'],masker=c['masker'],donor_repeat=c['donor_repeat'],
                requested_fraction=c['requested_fraction'],target_mask_sha256=c['target_mask_sha256'],**row))
    stream_table('control_components',component_rows)
    stream_table('control_responses',control_rows)
    # Background-specific dose differences are not hidden by background averaging.
    links={(r['rank_id'],c['masker'],c['donor_repeat'],c['requested_fraction']):c
           for r in payload['rank_links'] for c in [payload['components'][r['component_id']]]}
    dose=[]
    for rank in payload['rank_scores']:
        if rank['kind']!='real_swap':continue
        for fraction in FRACTIONS:
            repeats=(0,) if rank['masker']=='train_median' else range(DONOR_REPEATS)
            for repeat in repeats:
                own=links[('own',rank['masker'],repeat,float(fraction))]
                swap=links[(rank['rank_id'],rank['masker'],repeat,float(fraction))]
                dose.append(dict(**payload['identity'],source_seed=rank['source_seed'],
                    source_model_id=rank['source_model_id'],masker=rank['masker'],donor_repeat=int(repeat),
                    requested_fraction=float(fraction),actual_fraction=own['actual_fraction'],
                    rms_own=own['target_rms'],rms_swap=swap['target_rms'],
                    rms_swap_minus_own=swap['target_rms']-own['target_rms'],
                    own_run_lengths=own['target_run_lengths'],swap_run_lengths=swap['target_run_lengths'],
                    own_component_id=own['component_id'],swap_component_id=swap['component_id']))
    stream_table('paired_background_dose',dose)


def component_reweighting(component):
    if not component['complete']:return np.full(MC_DRAWS,np.nan)
    numer=np.zeros(MC_DRAWS);denom=np.zeros(MC_DRAWS)
    for row in component['controls']:
        rng=np.random.default_rng(keyed_seed(MC_SEED,component['matching_search_id'],row['mask_sha256']))
        weights=rng.exponential(size=MC_DRAWS)
        numer+=weights*row['abs_delta_logit'];denom+=weights
    return numer/denom


def rank_control_reweighting(payload,rank_id,masker):
    point_map={(r['rank_id'],r['masker'],r['requested_fraction']):r for r in payload['rank_points']}
    means=[];fractions=[]
    memo=payload.setdefault('_component_mc',{})
    for fraction in FRACTIONS:
        row=point_map[(rank_id,masker,float(fraction))]
        if not row['matched_budget_complete']:return np.full(MC_DRAWS,np.nan)
        components=[payload['components'][key] for key in json.loads(row['component_ids'])]
        values=[]
        for c in components:
            if c['component_id'] not in memo:memo[c['component_id']]=component_reweighting(c)
            values.append(memo[c['component_id']])
        means.append(np.mean(values,axis=0));fractions.append(row['actual_fraction'])
    values=np.vstack([np.zeros(MC_DRAWS),*means])
    return np.trapezoid(values,np.r_[0.,fractions],axis=0)/fractions[-1]
'''


def build_notebook():
    cells=[]
    markdown(cells, r'''
# Fixed-model ranking swap with amplitude-aware controls

Additional exploratory analysis, designed 2026-10-16. This notebook uses the
existing three-seed NewHandPD checkpoints only. **No training or calibration.**
It performs new forward interventions only when you execute its cells. Historical
E1/E2/RMS/bridge outputs and manuscript V5 remain unchanged.

Read `plans/attention_audit_extension_20261016/PROTOCOL.md`. Open this notebook with
kernel **Skripsi (.venv)** and **Restart Kernel → Run All**, after notebook 19.
Notebook 19 results are not used to choose this notebook's endpoints or settings.
CUDA and the original package/runtime profile are required. Each run writes a new
directory under `results/attention_ranking_swap/` and corresponding `figures/`.
Content-addressed record caches are separate from historical caches. After a crash,
Run All can reuse matching plans only after checksum and full numerical replay.

The primary endpoint is mean own-minus-swapped RMS-adjusted advantage, D. AOPC
uses absolute logit change. Report transferred-ranking advantage and both signed
and absolute D. A positive D favors the own ranking under this intervention; it
is not automatically evidence of clinical explanation quality.

Builder: `.venv/bin/python scripts/build_nb_20.py` (generation only).
The agent generates and statically checks source, without executing these cells.
''')
    add_preflight(cells,"20_fixed_model_ranking_swap.ipynb","build_nb_20.py",
                  "attention_ranking_swap","FIXED_MODEL_RANKING_SWAP_RMS_V1")
    code(cells, r'''
import sys, time, torch
from e2_checkpoints import get_fold_model, FrozenPredictor, spec_digest
from newhandpd import muat_cache_bisp
from audit_e2 import apply_mask

# ROOT/src must precede imports of project modules (set explicitly below).
'''.replace('import sys, time, torch\n','import sys, time, torch\nsys.path.insert(0,str(ROOT/"src"))\n'))
    code(cells, r'''
BOOT_SEED=2002
MATCH_SEED,PLACEBO_SEED,MC_SEED=2000,2001,2003
PLACEBO_CANDIDATES,PLACEBO_JSD_CALIPER=199,.025
MC_DRAWS=500
RMS_REL_TOL,RMS_ABS_TOL=.05,1e-8
MAX_CANDIDATES_PER_BUDGET,CANDIDATE_BATCH=4096,128
DONOR_REPEATS,RANDOM_REPEATS=recipe['donor_repeats'],recipe['random_repeats']
DEVICE='cuda'
if not torch.cuda.is_available():raise RuntimeError('Original project CUDA runtime required; no fallback training')
VERSIONS={p:version(p) for p in ('numpy','pandas','scipy','scikit-learn','torch',
                               'mamba-ssm','causal-conv1d','triton','einops')}
VERSIONS['cuda_runtime']=str(torch.version.cuda)
check(VERSIONS==baseline_manifest['versions'],'original inference package versions')
timing_rows=[]
file_sha256=file_hash  # Name used by the embedded legacy runtime-source checker.
plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':9})
''')
    markdown(cells, r'''
## Locked input, matching and numerical functions

The following legacy functions are embedded verbatim from builder 17. The old
builder is checked against its saved hash at generation and execution. The new
notebook never executes an old notebook. Rejection sampling retains its existing
caliper, geometry, quotas and candidate cap; failed matching is not imputed.
''')
    code(cells, legacy_functions(17,("validate_recording","mask_runs","patch_energy","geometry_candidates",
                                   "matched_candidates","replay_donor","forward_one","assert_logit",
                                   "alpha_report","make_libraries","runtime_kernels","config_payload",
                                   "pin_runtime","check_runtime_profile","predictor_for_case")))
    markdown(cells, r'''
## Load the existing runtime profile and verify original sentinels

No bounded search is repeated and no installed package is edited. Restore the
recorded numerical flags, verify the complete environment and profile, then test
the first and longest recording of the original first BiMamba-2 checkpoint.
Every subsequent recording also receives baseline attention replay. Outputs from
an incompatible runtime are not substituted for the historical bank.
''')
    code(cells, r'''
cache_path=ROOT/'data/cache/newhandpd_spiral_fs100.npz'
track(cache_path,recipe['cache_sha256'])
recordings=muat_cache_bisp(cache_path)
check(len(recordings)==len(records),'record cache count')
for idx,recording in enumerate(recordings):validate_recording(recording,records.iloc[idx],FS)
models=models.sort_values(['architecture','seed','fold']).reset_index(drop=True)
model_lookup={(r.architecture,int(r.seed),int(r.fold)):r.model_id for r in models.itertuples(index=False)}
top_columns=['ranking','model_id','idx','masker','donor_repeat','requested_fraction',
             'mask_sha256','n_masked','perturbed_logit',*METRICS]
top_chunks=[]
track(BASELINE/'e2_curves_raw.csv')
for chunk in pd.read_csv(BASELINE/'e2_curves_raw.csv',usecols=top_columns,chunksize=25000,float_precision='round_trip'):
    top_chunks.append(chunk.loc[chunk.ranking=='alpha_top'])
REF_KEYS=['model_id','idx','masker','donor_repeat','requested_fraction']
top_refs=pd.concat(top_chunks,ignore_index=True).set_index(REF_KEYS)
check(top_refs.index.is_unique and len(top_refs)==len(maps)*6*len(FRACTIONS),'complete original attention references')
del top_chunks
old_diagnostics=diagnostics.set_index(REF_KEYS)
old_control_frame=pd.read_csv(AMPLITUDE/'controls_raw.csv',float_precision='round_trip')
old_control_frame=old_control_frame.loc[old_control_frame.control=='rms_matched_random'].copy()
old_controls=old_control_frame.groupby(REF_KEYS,sort=False)

profile_path=ROOT/amplitude_manifest['backend_profile']['path']
track(profile_path,pins['input_sha256'][str(profile_path.relative_to(ROOT))])
saved_profile=json.loads(profile_path.read_text())
check(spec_digest(saved_profile['spec'])==saved_profile['profile_id']
      ==amplitude_manifest['backend_profile']['profile_id'],'saved numerical profile ID')
runtime_spec=saved_profile['spec'];ctx=runtime_spec['environment']['torch_context']
check(str(torch.get_default_dtype())==ctx['default_dtype'],'recorded tensor dtype')
torch.set_float32_matmul_precision(ctx['float32_matmul_precision'])
torch.backends.cuda.matmul.allow_tf32=ctx['cuda_matmul_allow_tf32']
torch.backends.cudnn.allow_tf32=ctx['cudnn_allow_tf32']
torch.backends.cudnn.benchmark=ctx['cudnn_benchmark']
torch.use_deterministic_algorithms(ctx['deterministic_algorithms'])
actual_context=dict(default_dtype=str(torch.get_default_dtype()),float32_matmul_precision=torch.get_float32_matmul_precision(),
    cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
    cudnn_benchmark=bool(torch.backends.cudnn.benchmark),deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
objects,registered,runtime_sources=runtime_kernels()
environment=dict(versions=VERSIONS,device=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
                 torch_context=actual_context,sources=runtime_sources)
check(environment==runtime_spec['environment'],'saved backend environment and source identity')
for rel,checksum in runtime_sources.items():track(ROOT/rel,checksum)
pin_runtime(objects,registered,runtime_spec['kernels'])
probe_row=next(r for r in models.itertuples(index=False) if r.architecture=='mamba2')
probe_spec=specs[probe_row.model_id]
check(probe_row.model_id==runtime_spec['model_id'],'original first BiMamba-2 checkpoint')
probe_model,probe_norm,probe_info=get_fold_model(ROOT,probe_spec,recordings,DEVICE,allow_training=False)
check(not probe_info['trained_now'],'no probe training')
probe_payload=pickle.loads((ROOT/probe_row.evaluation_cache).read_bytes())
probe_training={int(i):probe_norm.transform(recordings[i].kanal) for i in probe_spec['split']['train_indices']}
probe_first=int(probe_spec['split']['test_indices'][0])
probe_longest=int(max(probe_spec['split']['test_indices'],key=lambda i:(int(records.iloc[i].n_patches),-int(i))))
probe_cases=[]
for idx in dict.fromkeys([probe_first,probe_longest]):
    rec=records.iloc[idx];x=probe_norm.transform(recordings[idx].kanal)
    donor_plan=next(p for p in probe_payload['donor_plans'] if int(p['idx'])==idx)
    identity=dict(architecture=probe_row.architecture,seed=int(probe_row.seed),fold=int(probe_row.fold),
        model_id=probe_row.model_id,idx=idx,record_id=rec.record_id,subject=rec.subject,label=int(rec.label))
    libraries=make_libraries(x,donor_plan['donor_library'],probe_training,probe_spec['split']['train_indices'],rec)
    probe_cases.append(dict(identity=identity,x=x,p=maps[(probe_row.model_id,idx)],libraries=libraries))
check([c['identity']['record_id'] for c in probe_cases]==runtime_spec['sentinel_record_ids'],'original sentinel recordings')
sentinel_points=[]
run_manifest.update(model_inference_performed=True,completion_status='runtime_probe')
(OUT/'analysis_manifest.json').write_text(json.dumps(run_manifest,indent=2),encoding='utf-8')
try:
    sentinel=check_runtime_profile(probe_model,probe_cases,runtime_spec['kernels'],0,sentinel_points)
    pd.DataFrame([sentinel]).to_csv(OUT/'runtime_profile_check.csv',index=False)
    pd.DataFrame(sentinel_points).to_csv(OUT/'runtime_sentinel_points.csv',index=False)
    check(sentinel['passed'],'saved runtime sentinel replay',sentinel)
finally:
    del probe_model,probe_norm,probe_payload,probe_training,probe_cases
    torch.cuda.empty_cache()
run_manifest.update(model_inference_performed=True,device=environment['device'],backend_profile_id=saved_profile['profile_id'],
                    backend_environment=environment)
''')
    code(cells, swap_methods_source())
    markdown(cells, r'''
## Complete directed schedule and immutable record caches

For three seeds use all six directed target-source pairs per recording. No ranking
is selected by its result. Own controls retain their original completeness flags.
Swapped/reference masks receive their own controls; identical masks share plans
and outputs. One selected circular/permuted reference per real pair is a bounded
diagnostic with a prespecified JSD caliper, not an exhaustive reference average.

The loop logs loading, replay, matching and forward timings. Cache writes are
atomic and checksum-bound; original files are read-only. A mismatch stops the run.
On a cache hit, all ranking/control logits are replayed; search is not repeated.
''')
    code(cells, r'''
schedule=[]
for arm in ARMS:
    for target in SEEDS:
        for source in SEEDS:
            if target!=source:schedule.append(dict(architecture=arm,target_seed=target,source_seed=source))
schedule=pd.DataFrame(schedule)
schedule.to_csv(OUT/'directed_seed_schedule.csv',index=False)
SWAP_SPEC=dict(format_version=1,analysis_kind=ANALYSIS_KIND,source_sha256=SOURCE_DIGEST,
    protocol_sha256=file_hash(PLAN/'PROTOCOL.md'),source_pins_sha256=file_hash(PLAN/'source_pins.json'),
    shared_builder_sha256=file_hash(ROOT/'scripts/build_nb_19.py'),
    baseline_evaluation_id=EXPECTED_EVALUATION,amplitude_control_id=EXPECTED_CONTROL,
    runtime_profile_id=saved_profile['profile_id'],environment=environment,
    recipe=recipe,matching=dict(seed=MATCH_SEED,relative_tol=RMS_REL_TOL,absolute_tol=RMS_ABS_TOL,
       max_candidates=MAX_CANDIDATES_PER_BUDGET,batch=CANDIDATE_BATCH,legacy_own_flags_preserved=True),
    placebo=dict(seed=PLACEBO_SEED,max_candidates=PLACEBO_CANDIDATES,jsd_caliper=PLACEBO_JSD_CALIPER,
                 selected_per_type=1,types=['circular_shift','permutation']),
    primary='subject-balanced mean D=G_own-G_swap',schedule=schedule.to_dict('records'),
    bootstrap=dict(n_boot=N_BOOT,seed=BOOT_SEED),control_sensitivity=dict(draws=MC_DRAWS,seed=MC_SEED))
ANALYSIS_ID=digest(SWAP_SPEC)
CACHE_ROOT=ROOT/'results/attention_ranking_swap_cache'/ANALYSIS_ID
CACHE_ROOT.mkdir(parents=True,exist_ok=True)
run_manifest.update(analysis_id=ANALYSIS_ID,analysis_spec=SWAP_SPEC,completion_status='interventions_running')
(OUT/'analysis_manifest.json').write_text(json.dumps(run_manifest,indent=2),encoding='utf-8')
all_scores=[];all_points=[];all_reference=[];mc_payloads={};record_cache_rows=[]
import fcntl
cache_lock=(CACHE_ROOT/'analysis.lock').open('a+')
try:
    fcntl.flock(cache_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
    cache_lock.close()
    raise RuntimeError('Another kernel is running this swap analysis; keep only one active Run All')
try:
    for model_row in models.itertuples(index=False):
        started=time.perf_counter();spec=specs[model_row.model_id]
        model,norm,info=get_fold_model(ROOT,spec,recordings,DEVICE,allow_training=False)
        check(not info['trained_now'] and info['checkpoint_sha256']==model_row.checkpoint_sha256,
              'checkpoint-only target',model_row.model_id)
        timing_rows.append(dict(model_id=model_row.model_id,architecture=model_row.architecture,
                               phase='checkpoint_load',seconds=time.perf_counter()-started))
        train=spec['split']['train_indices'];test=spec['split']['test_indices']
        track(ROOT/model_row.evaluation_cache)
        fold_payload=pickle.loads((ROOT/model_row.evaluation_cache).read_bytes())
        plans={int(p['idx']):p for p in fold_payload['donor_plans']}
        check(set(plans)==set(test),'saved donor plans cover target test fold')
        training_inputs={int(i):norm.transform(recordings[i].kanal) for i in train}
        for idx in test:
            idx=int(idx);rec=records.iloc[idx]
            identity=dict(architecture=model_row.architecture,seed=int(model_row.seed),fold=int(model_row.fold),
                model_id=model_row.model_id,idx=idx,record_id=rec.record_id,subject=rec.subject,label=int(rec.label))
            x=norm.transform(recordings[idx].kanal)
            predictor=FrozenPredictor(model,DEVICE,len(x));p=maps[(model_row.model_id,idx)]
            libraries=make_libraries(x,plans[idx]['donor_library'],training_inputs,train,rec)
            for b in libraries:b['background_sha256']=sha256(np.ascontiguousarray(b['background']).tobytes()).hexdigest()
            path=CACHE_ROOT/model_row.model_id/f'{idx}.pkl'
            reused=path.is_file()
            started=time.perf_counter()
            if reused:
                payload=load_record_cache(path,predictor,x,p,libraries,identity)
            else:
                run_manifest.update(new_interventions_performed=True)
                (OUT/'analysis_manifest.json').write_text(json.dumps(run_manifest,indent=2),encoding='utf-8')
                payload=evaluate_target_record(predictor,x,p,libraries,identity)
                serialization_start=time.perf_counter()
                cache_record(path,payload)
                timing_rows.append(dict(**identity,phase='record_cache_serialize',
                                        seconds=time.perf_counter()-serialization_start))
            record_cache_rows.append(dict(**identity,reused=reused,path=str(path.relative_to(ROOT)),
                sha256=file_hash(path),seconds_total=time.perf_counter()-started))
            export_start=time.perf_counter()
            export_record(payload)
            timing_rows.append(dict(**identity,phase='csv_export',seconds=time.perf_counter()-export_start))
            all_scores.extend(payload['rank_scores']);all_points.extend(payload['rank_points'])
            all_reference.extend(payload['reference_support'])
            # Cache only finite-pool reweighting outputs for downstream paired D.
            for masker in MASKERS:
                for score in payload['rank_scores']:
                    if score['masker']==masker and score['kind'] in {'own','real_swap'} and score['matched_curve_complete']:
                        mc_payloads[(identity['model_id'],idx,score['rank_id'],masker)]=rank_control_reweighting(
                            payload,score['rank_id'],masker)
            pd.DataFrame(record_cache_rows).to_csv(OUT/'record_cache_inventory.csv',index=False)
            flush_timings()
        print(model_row.architecture,'seed',model_row.seed,'fold',model_row.fold,'verified records:',len(test))
        del model,norm,predictor,training_inputs,fold_payload
        torch.cuda.empty_cache()
except Exception as exc:
    run_manifest.update(completion_status='failed',failure_type=type(exc).__name__,failure_message=str(exc))
    (OUT/'analysis_manifest.json').write_text(json.dumps(run_manifest,indent=2),encoding='utf-8')
    flush_timings()
    raise
finally:
    fcntl.flock(cache_lock.fileno(),fcntl.LOCK_UN)
    cache_lock.close()
rank_scores=pd.DataFrame(all_scores);rank_points=pd.DataFrame(all_points)
reference_support=pd.DataFrame(all_reference)
run_manifest.update(intervention_replays_performed=True,completion_status='aggregation_running')
(OUT/'analysis_manifest.json').write_text(json.dumps(run_manifest,indent=2),encoding='utf-8')
''')
    # Aggregation and plots are appended separately below; no scientific execution.
    return finish_swap_notebook(cells)


def finish_swap_notebook(cells):
    markdown(cells, "## Paired endpoints, coverage, conditional inference and figures")
    code(cells, inference_source(directed=True))
    code(cells, aggregation_source())
    code(cells, plotting_source())
    markdown(cells, r'''
## Completion and interpretation

Inspect `analysis_manifest.json` and `alignment_checks.csv` first, then self-swap,
replay, matching/coverage and dose tables. Primary `D` contrasts and G_swap use
the same eligible support. A positive mean G_swap is an above-control response
under the tested replacements; mean D near zero is not an equivalence result.
Reference contrasts use an additional common-support restriction. Inspect its
losses before interpreting a reference comparison.

`control_reweighting_sensitivity.csv` reports conditional finite-pool quantiles,
not subject CIs or uncertainty of drawing a fresh matching search. Three training
seeds remain three seeds; subject CIs condition on this bank. Secondary intervals
are exploratory and nominal. Save this run's whole results directory when sharing
results. Training-seed/UCI replication is a later, separately locked protocol.
''')
    add_finish(cells)
    return nbf.v4.new_notebook(cells=cells)


def aggregation_source():
    return r'''
def balanced_weights(data):
    # Directed schedule: equal subjects, recordings, target seeds, then source seeds.
    sources=data.groupby(['subject','idx','target_seed']).source_seed.transform('size').to_numpy(float)
    targets=data.groupby(['subject','idx']).target_seed.transform('nunique').to_numpy(float)
    recordings_per_subject=data.groupby('subject').idx.transform('nunique').to_numpy(float)
    return 1/(recordings_per_subject*targets*sources)


BOOT_COUNTS=cluster_counts(subjects,N_BOOT,BOOT_SEED)
np.savez_compressed(OUT/'subject_bootstrap_counts.npz',counts=BOOT_COUNTS,
                    subjects=SUBJECT_POOL.astype(str),labels=subjects.label.to_numpy())
own_lookup={(r.model_id,int(r.idx),r.masker):r for r in rank_scores[rank_scores.kind=='own'].itertuples(index=False)}
point_lookup={(r.model_id,int(r.idx),r.rank_id,r.masker,float(r.requested_fraction)):r
              for r in rank_points.itertuples(index=False)}
score_lookup={(r.model_id,int(r.idx),r.rank_id,r.masker):r for r in rank_scores.itertuples(index=False)}
reference_lookup={(r.model_id,int(r.idx),int(r.source_seed),r.reference):r
                  for r in reference_support.itertuples(index=False)}

pair_rows=[];budget_rows=[];reference_pairs=[];mc_D={}
for swap in rank_scores[rank_scores.kind=='real_swap'].itertuples(index=False):
    own=own_lookup[(swap.model_id,int(swap.idx),swap.masker)]
    source=maps[(swap.source_model_id,int(swap.idx))]
    target=maps[(swap.model_id,int(swap.idx))]
    eligible=bool(own.matched_curve_complete and swap.matched_curve_complete)
    qtarget,qsource=float(target['logit']),float(source['logit'])
    common=dict(architecture=swap.architecture,target_seed=int(swap.seed),source_seed=int(swap.source_seed),
        fold=int(swap.fold),target_model_id=swap.model_id,source_model_id=swap.source_model_id,
        idx=int(swap.idx),record_id=swap.record_id,subject=swap.subject,label=int(swap.label),masker=swap.masker,
        js_divergence=float(swap.js_divergence),abs_delta_logit=abs(qtarget-qsource),
        abs_delta_prob=abs(float(expit(qtarget))-float(expit(qsource))),class_agreement=(qtarget>=0)==(qsource>=0))
    row=dict(**common,own_complete=bool(own.matched_curve_complete),swap_complete=bool(swap.matched_curve_complete),
        pair_eligible=eligible,eligibility_reason='both_complete' if eligible else
        'both_incomplete' if not own.matched_curve_complete and not swap.matched_curve_complete else
        'own_incomplete' if not own.matched_curve_complete else 'swap_incomplete',
        T_own=own.T_abs_delta_logit,T_swap=swap.T_abs_delta_logit,
        R_own=own.R_abs_delta_logit,R_swap=swap.R_abs_delta_logit,
        G_own=own.G_abs_delta_logit if eligible else np.nan,
        G_swap=swap.G_abs_delta_logit if eligible else np.nan,
        effect_sum_own=own.effect_sum,effect_sum_swap=swap.effect_sum,
        raw_T_own_minus_swap=own.T_abs_delta_logit-swap.T_abs_delta_logit)
    row['D']=row['G_own']-row['G_swap'];row['abs_D']=abs(row['D'])
    row['D_signed_margin']=(own.G_signed_margin_drop-swap.G_signed_margin_drop) if eligible else np.nan
    row['D_predicted_class_probability']=(own.G_predicted_class_prob_drop-swap.G_predicted_class_prob_drop) if eligible else np.nan
    row['normalization_eligible']=eligible and own.effect_sum>PRIMARY_FLOOR and swap.effect_sum>PRIMARY_FLOOR
    row['D_H']=own.H-swap.H if row['normalization_eligible'] else np.nan
    row['abs_D_H']=abs(row['D_H'])
    row['stable_gate']=bool(common['class_agreement'] and common['abs_delta_prob']<=.02 and common['abs_delta_logit']<=.1)
    pair_rows.append(row)
    if eligible:
        ro=mc_payloads[(swap.model_id,int(swap.idx),'own',swap.masker)]
        rs=mc_payloads[(swap.model_id,int(swap.idx),swap.rank_id,swap.masker)]
        mc_D[(swap.model_id,int(swap.idx),int(swap.source_seed),swap.masker)]=row['raw_T_own_minus_swap']-ro+rs
    own_order=ranking_order(target['alpha'],swap.record_id)
    swap_order=ranking_order(source['alpha'],swap.record_id)
    for fraction in FRACTIONS:
        a=point_lookup[(swap.model_id,int(swap.idx),'own',swap.masker,float(fraction))]
        b=point_lookup[(swap.model_id,int(swap.idx),swap.rank_id,swap.masker,float(fraction))]
        if a.actual_fraction!=b.actual_fraction or a.n_masked!=b.n_masked or a.n_patches!=b.n_patches:
            raise AlignmentError('own/swap budget or grid mismatch')
        k=int(a.n_masked);ma=np.sort(own_order[:k]);mb=np.sort(swap_order[:k])
        if mask_hash(ma)!=a.target_mask_sha256 or mask_hash(mb)!=b.target_mask_sha256:
            raise AlignmentError('own/swap point does not use saved ranking mask')
        ok=bool(a.matched_budget_complete and b.matched_budget_complete)
        ta,tb=a.T_abs_delta_logit,b.T_abs_delta_logit
        ra,rb=a.R_abs_delta_logit,b.R_abs_delta_logit
        budget_rows.append(dict(**common,requested_fraction=float(fraction),actual_fraction=a.actual_fraction,
            n_masked=k,n_patches=a.n_patches,pair_eligible=ok,
            own_complete=a.matched_budget_complete,swap_complete=b.matched_budget_complete,
            T_own=ta,T_swap=tb,R_own=ra,R_swap=rb,
            G_own=ta-ra if ok else np.nan,G_swap=tb-rb if ok else np.nan,
            D=(ta-ra)-(tb-rb) if ok else np.nan,raw_T_own_minus_swap=ta-tb,
            rms_own_mean=a.target_rms_mean,rms_swap_mean=b.target_rms_mean,
            rms_swap_minus_own_mean=b.target_rms_mean-a.target_rms_mean,
            own_run_lengths=a.target_run_lengths,swap_run_lengths=b.target_run_lengths,
            top_mask_jaccard=len(np.intersect1d(ma,mb))/len(np.union1d(ma,mb)),
            identical_mask=bool(np.array_equal(ma,mb))))
    for reference in ('circular_shift','permutation'):
        support=reference_lookup[(swap.model_id,int(swap.idx),int(swap.source_seed),reference)]
        placebo=score_lookup.get((swap.model_id,int(swap.idx),f'{reference}_for_seed_{swap.source_seed}',swap.masker))
        valid=eligible and placebo is not None and placebo.matched_curve_complete
        ref=dict(**common,reference=reference,real_pair_eligible=eligible,
            jsd_match_complete=bool(support.matching_complete),jsd_absolute_error=support.absolute_jsd_error,
            placebo_jsd=support.candidate_jsd,reference_pair_eligible=bool(valid),
            failure_reason='none' if valid else support.failure_reason if not support.matching_complete
                           else 'real_pair_incomplete' if not eligible else 'placebo_rms_incomplete')
        ref['G_real']=row['G_swap'] if valid else np.nan
        ref['G_placebo']=placebo.G_abs_delta_logit if valid else np.nan
        ref['G_real_minus_placebo']=ref['G_real']-ref['G_placebo']
        ref['D_real']=row['D'] if valid else np.nan
        ref['D_placebo']=row['G_own']-ref['G_placebo'] if valid else np.nan
        ref['abs_D_real_minus_placebo']=abs(ref['D_real'])-abs(ref['D_placebo']) if valid else np.nan
        reference_pairs.append(ref)
pair_curves=pd.DataFrame(pair_rows);pair_budgets=pd.DataFrame(budget_rows)
reference_pairs=pd.DataFrame(reference_pairs)
PAIR_KEYS=['architecture','target_seed','source_seed','fold','target_model_id','source_model_id',
           'idx','record_id','subject','label','masker']
unique(pair_curves,PAIR_KEYS,'directed fixed-model pairs')
check(len(pair_curves)==len(records)*len(ARMS)*len(SEEDS)*(len(SEEDS)-1)*len(MASKERS),
      'every planned directed pair retained')
pair_curves.to_csv(OUT/'swap_pair_curves.csv',index=False)
pair_budgets.to_csv(OUT/'swap_pair_budgets.csv',index=False)
reference_pairs.to_csv(OUT/'swap_placebo_pairs.csv',index=False)

# All counts use planned pairs as the denominator, including failed matching.
coverage_rows=[];subject_coverage=[];seed_coverage=[]
for (arm,masker),group in pair_curves.groupby(['architecture','masker'],sort=True):
    row=dict(architecture=arm,masker=masker,n_pairs_total=len(group),n_records_total=group.idx.nunique(),
             n_subjects_total=len(subjects))
    for name,gate in [('matched',group.pair_eligible),('stable_before_matching',group.stable_gate),
                      ('stable_after_matching',group.stable_gate&group.pair_eligible),
                      ('normalized',group.normalization_eligible)]:
        row['n_pairs_'+name]=int(gate.sum());row['n_records_'+name]=group.loc[gate,'idx'].nunique()
        row['n_subjects_'+name]=group.loc[gate,'subject'].nunique()
    coverage_rows.append(row)
    for subject,label in subjects.itertuples(index=False):
        sub=group.loc[group.subject==subject]
        subject_coverage.append(dict(architecture=arm,masker=masker,subject=subject,label=int(label),
            n_pairs_total=len(sub),n_pairs_matched=int(sub.pair_eligible.sum()),
            n_pairs_stable_before=int(sub.stable_gate.sum()),n_pairs_stable_matched=int((sub.stable_gate&sub.pair_eligible).sum())))
    for (target,source),sub in group.groupby(['target_seed','source_seed']):
        seed_coverage.append(dict(architecture=arm,masker=masker,target_seed=target,source_seed=source,
            n_pairs_total=len(sub),n_pairs_matched=int(sub.pair_eligible.sum()),
            n_subjects_matched=sub.loc[sub.pair_eligible,'subject'].nunique()))
coverage=pd.DataFrame(coverage_rows)
coverage.to_csv(OUT/'swap_coverage.csv',index=False)
pd.DataFrame(subject_coverage).to_csv(OUT/'swap_subject_coverage.csv',index=False)
pd.DataFrame(seed_coverage).to_csv(OUT/'swap_seed_pair_coverage.csv',index=False)
reference_coverage=reference_pairs.groupby(['architecture','masker','reference']).agg(
    n_pairs_total=('idx','size'),n_real_pairs_matched=('real_pair_eligible','sum'),
    n_jsd_matched=('jsd_match_complete','sum'),n_reference_pairs_matched=('reference_pair_eligible','sum')).reset_index()
for stage,flag in [('real','real_pair_eligible'),('reference','reference_pair_eligible')]:
    eligible=reference_pairs.loc[reference_pairs[flag]]
    counts=eligible.groupby(['architecture','masker','reference']).agg(
        **{'n_records_'+stage:('idx','nunique'),'n_subjects_'+stage:('subject','nunique')}).reset_index()
    reference_coverage=reference_coverage.merge(counts,on=['architecture','masker','reference'],how='left',validate='one_to_one')
reference_coverage.to_csv(OUT/'swap_placebo_coverage.csv',index=False)

primary_rows=[];secondary_rows=[];association_rows=[];primary_draws=[]
normalization_rows=[];influence_rows=[];budget_ci_rows=[]
for (arm,masker),full in pair_curves.groupby(['architecture','masker'],sort=True):
    for scenario,gate in [('all_pairs',full.pair_eligible),('stable_output',full.pair_eligible&full.stable_gate)]:
        g=full.loc[gate].copy()
        for endpoint in ('D','G_swap','G_own','abs_D','raw_T_own_minus_swap','D_signed_margin','D_predicted_class_probability'):
            result,draws=estimate_ci(g.loc[np.isfinite(g[endpoint])],weighted_mean_stat(endpoint))
            row=dict(architecture=arm,masker=masker,scenario=scenario,endpoint=endpoint,**result)
            if endpoint=='D' and scenario=='all_pairs':
                finite=draws[np.isfinite(draws)]
                row['approx_centered_bootstrap_p']=(float((1+np.count_nonzero(abs(finite-result['estimate'])>=abs(result['estimate'])))
                                                         /(len(finite)+1)) if result['ci_status']=='reported' else np.nan)
                primary_rows.append(row);primary_draws.append(draws)
            else:secondary_rows.append(row)
        for outcome in ('D','abs_D','D_H','abs_D_H'):
            valid=g.loc[np.isfinite(g[outcome])].copy()
            result,_=estimate_ci(valid,lambda d,w,y=outcome:rank_correlation(d,'js_divergence',y,w))
            association_rows.append(dict(architecture=arm,masker=masker,scenario=scenario,outcome=outcome,
                                        estimator='subject_balanced_weighted_spearman',**result))
    for floor in DENOM_FLOORS:
        valid=full.loc[full.pair_eligible&(full.effect_sum_own>floor)&(full.effect_sum_swap>floor)].copy()
        valid['D_H_floor']=(valid.T_own-valid.R_own)/valid.effect_sum_own-(valid.T_swap-valid.R_swap)/valid.effect_sum_swap
        for endpoint in ('D_H_floor','D'):
            result,_=estimate_ci(valid,weighted_mean_stat(endpoint))
            normalization_rows.append(dict(architecture=arm,masker=masker,floor=floor,endpoint=endpoint,**result))
    for omitted in SEEDS:
        valid=full.loc[full.pair_eligible&(full.target_seed!=omitted)&(full.source_seed!=omitted)].copy()
        weights=balanced_weights(valid) if len(valid) else np.array([])
        value=weighted_mean_stat('D')(valid,weights) if len(valid) else np.nan
        influence_rows.append(dict(architecture=arm,masker=masker,omitted_seed=omitted,mean_D=value,
            n_pairs=len(valid),n_records=valid.idx.nunique(),n_subjects=valid.subject.nunique(),
            status='leave_one_seed_out_influence_not_seed_CI'))
for (arm,masker,fraction),group in pair_budgets.groupby(['architecture','masker','requested_fraction'],sort=True):
    valid=group.loc[group.pair_eligible].copy()
    for endpoint in ('D','G_swap','G_own','raw_T_own_minus_swap'):
        result,_=estimate_ci(valid,weighted_mean_stat(endpoint))
        budget_ci_rows.append(dict(architecture=arm,masker=masker,requested_fraction=fraction,endpoint=endpoint,**result))
primary=pd.DataFrame(primary_rows)
check(set(zip(primary.architecture,primary.masker))=={(a,m) for a in ARMS for m in MASKERS},
      'four planned primary strata retained')
raw_p=primary.approx_centered_bootstrap_p.to_numpy(float)
bookkeeping_p=np.where(np.isfinite(raw_p),raw_p,1.)
order=np.argsort(bookkeeping_p,kind='stable')
adjusted=np.minimum(1.,np.maximum.accumulate((len(order)-np.arange(len(order)))*bookkeeping_p[order]))
restored=np.empty_like(adjusted);restored[order]=adjusted
primary['holm_adjusted_p']=np.where(np.isfinite(raw_p),restored,np.nan)
primary['multiplicity_family']='four exploratory architecture-by-replacement mean-D tests'
primary['ci_multiplicity']='nominal 95%; not simultaneous'
primary.to_csv(OUT/'swap_primary_ci.csv',index=False)
pd.DataFrame(secondary_rows).to_csv(OUT/'swap_secondary_ci.csv',index=False)
associations=pd.DataFrame(association_rows)
associations.to_csv(OUT/'swap_association_ci.csv',index=False)
pd.DataFrame(normalization_rows).to_csv(OUT/'swap_normalization_ci.csv',index=False)
pd.DataFrame(influence_rows).to_csv(OUT/'leave_one_seed_out.csv',index=False)
pd.DataFrame(budget_ci_rows).to_csv(OUT/'swap_budget_ci.csv',index=False)
np.savez_compressed(OUT/'primary_subject_bootstrap_draws.npz',draws=np.asarray(primary_draws),
                    architecture=primary.architecture.to_numpy(str),masker=primary.masker.to_numpy(str))

reference_ci_rows=[]
for (arm,masker,reference),full in reference_pairs.groupby(['architecture','masker','reference'],sort=True):
    valid=full.loc[full.reference_pair_eligible].copy()
    for endpoint in ('G_real_minus_placebo','D_real','D_placebo','abs_D_real_minus_placebo'):
        result,_=estimate_ci(valid,weighted_mean_stat(endpoint))
        reference_ci_rows.append(dict(architecture=arm,masker=masker,reference=reference,endpoint=endpoint,**result))
reference_ci=pd.DataFrame(reference_ci_rows)
reference_ci.to_csv(OUT/'swap_placebo_ci.csv',index=False)

# Empirical finite-pool control reweighting: separate from subject inference.
mc_rows=[];mc_pair_rows=[];mc_group_draws=[]
for (arm,masker),full in pair_curves.groupby(['architecture','masker'],sort=True):
    valid=full.loc[full.pair_eligible].copy();weights=balanced_weights(valid) if len(valid) else np.array([])
    matrix=[]
    for row in valid.itertuples(index=False):
        values=mc_D[(row.target_model_id,int(row.idx),int(row.source_seed),masker)]
        matrix.append(values)
        mc_pair_rows.append(dict(architecture=arm,masker=masker,target_model_id=row.target_model_id,
            idx=row.idx,source_seed=row.source_seed,D=row.D,control_reweighting_sd=float(values.std(ddof=1)),
            control_reweighting_q025=float(np.quantile(values,.025)),control_reweighting_q975=float(np.quantile(values,.975))))
    mean_draws=np.average(np.asarray(matrix),axis=0,weights=weights) if matrix else np.full(MC_DRAWS,np.nan)
    finite=mean_draws[np.isfinite(mean_draws)]
    point=weighted_mean_stat('D')(valid,weights) if len(valid) else np.nan
    mc_rows.append(dict(architecture=arm,masker=masker,mean_D=point,n_pairs=len(valid),n_subjects=valid.subject.nunique(),
        n_reweightings=MC_DRAWS,conditional_reweighting_sd=float(finite.std(ddof=1)) if len(finite)>1 else np.nan,
        conditional_reweighting_q025=float(np.quantile(finite,.025)) if len(finite) else np.nan,
        conditional_reweighting_q975=float(np.quantile(finite,.975)) if len(finite) else np.nan,
        interpretation='finite accepted-pool sensitivity; not a population CI or fresh-search uncertainty'))
    mc_group_draws.append(mean_draws)
pd.DataFrame(mc_pair_rows).to_csv(OUT/'pair_control_reweighting_sensitivity.csv',index=False)
control_sensitivity=pd.DataFrame(mc_rows)
control_sensitivity.to_csv(OUT/'control_reweighting_sensitivity.csv',index=False)
np.savez_compressed(OUT/'control_reweighting_group_draws.npz',draws=np.asarray(mc_group_draws),
                    architecture=control_sensitivity.architecture.to_numpy(str),masker=control_sensitivity.masker.to_numpy(str))

# Exact replacement intersection for fixed-model D; keep distinct from old bridge contrast.
shared_keys=[k for k in PAIR_KEYS if k!='masker']
median=pair_curves[pair_curves.masker=='train_median']
donor=pair_curves[pair_curves.masker=='train_donor']
replacement=median.merge(donor,on=shared_keys,how='outer',validate='one_to_one',suffixes=('_median','_donor'),indicator=True)
check((replacement['_merge']=='both').all(),'swap replacement planned identities align')
replacement['both_complete']=replacement.pair_eligible_median&replacement.pair_eligible_donor
replacement['both_normalized']=replacement.normalization_eligible_median&replacement.normalization_eligible_donor
replacement['D_median_minus_donor']=replacement.D_median-replacement.D_donor
replacement['D_H_median_minus_donor']=replacement.D_H_median-replacement.D_H_donor
replacement.to_csv(OUT/'swap_replacement_common_support.csv',index=False)
replacement_ci=[]
for arm,group in replacement.groupby('architecture',sort=True):
    for endpoint,flag in [('D_median_minus_donor','both_complete'),('D_H_median_minus_donor','both_normalized')]:
        valid=group.loc[group[flag]&np.isfinite(group[endpoint])].copy()
        result,_=estimate_ci(valid,weighted_mean_stat(endpoint))
        replacement_ci.append(dict(architecture=arm,endpoint=endpoint,n_pairs_total=len(group),**result))
pd.DataFrame(replacement_ci).to_csv(OUT/'swap_replacement_contrast_ci.csv',index=False)
timings=pd.read_csv(OUT/'computation_timings.csv')
timings.groupby('phase').agg(n_events=('seconds','size'),total_seconds=('seconds','sum'),
                             median_seconds=('seconds','median')).reset_index().to_csv(OUT/'timing_summary.csv',index=False)
run_manifest.update(analysis_spec=SWAP_SPEC,completion_status='figures_running',
                    n_target_checkpoints=len(models),n_checkpoint_record_evaluations=len(record_cache_rows),
                    n_distinct_recordings=len(records),n_subjects=len(subjects),
                    n_unique_directed_record_pairs=len(pair_curves[[k for k in PAIR_KEYS if k!='masker']].drop_duplicates()),
                    n_directed_pair_replacement_rows=len(pair_curves),
                    pair_row_unit='architecture x target seed x source seed x recording x replacement',
                    training_performed=False,
                    inference='subject conditional on fixed bank; seed influence diagnostic separate',
                    control_reweighting='finite accepted-pool diagnostic, not population confidence intervals')
display(primary)
display(coverage)
display(reference_coverage)
display(control_sensitivity)
'''


def plotting_source():
    return r'''
ARM_LABEL={'gru':'BiGRU','mamba2':'BiMamba-2'}
MASK_LABEL={'train_median':'Training median','train_donor':'Training donor'}
MASK_COLOR={'train_median':'#744c8c','train_donor':'#28776b'}
MARKER={'gru':'o','mamba2':'s'}
strata=[(arm,masker) for arm in ARMS for masker in MASKERS]
labels=[ARM_LABEL[a]+' / '+MASK_LABEL[m] for a,m in strata]
fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained',sharey=True)
secondary=pd.DataFrame(secondary_rows)
for j,(endpoint,title) in enumerate([('D','Own minus transferred advantage'),('G_swap','Transferred advantage over RMS controls')]):
    source=primary if endpoint=='D' else secondary[(secondary.endpoint=='G_swap')&(secondary.scenario=='all_pairs')]
    for i,(arm,masker) in enumerate(strata):
        row=source[(source.architecture==arm)&(source.masker==masker)].iloc[0]
        if np.isfinite(row.estimate):
            axes[j].plot(row.estimate,i,MARKER[arm],color=MASK_COLOR[masker],ms=7)
            if np.isfinite(row.ci_low):axes[j].plot([row.ci_low,row.ci_high],[i,i],color=MASK_COLOR[masker],lw=1.5)
    axes[j].axvline(0,color='.5',ls=':');axes[j].set(title=title,xlabel='Absolute-logit AOPC contrast (nominal subject CI)')
count_labels=[]
for arm,masker in strata:
    row=primary[(primary.architecture==arm)&(primary.masker==masker)].iloc[0]
    count_labels.append(ARM_LABEL[arm]+' / '+MASK_LABEL[masker]
                        +f'\n{int(row.n_pairs)} pairs, {int(row.n_subjects)} subjects')
axes[0].set_yticks(range(len(strata)),count_labels,fontsize=9)
save_figure(fig,'fixed_model_swap_primary')

fig,axes=plt.subplots(2,2,figsize=(10,7.5),layout='constrained')
for i,arm in enumerate(ARMS):
    for j,masker in enumerate(MASKERS):
        g=pair_curves[(pair_curves.architecture==arm)&(pair_curves.masker==masker)&pair_curves.pair_eligible]
        axes[i,j].scatter(g.js_divergence,g.D,s=9,alpha=.2,color=MASK_COLOR[masker],rasterized=True)
        axes[i,j].axhline(0,color='.5',ls=':')
        axes[i,j].set(xlim=(0,1),xlabel='JSD (base-2 divergence)',ylabel='D: own minus transferred advantage',
                      title=ARM_LABEL[arm]+' / '+MASK_LABEL[masker])
save_figure(fig,'jsd_vs_signed_swap_effect')
fig,axes=plt.subplots(2,2,figsize=(10,7.5),layout='constrained')
for i,arm in enumerate(ARMS):
    for j,masker in enumerate(MASKERS):
        g=pair_curves[(pair_curves.architecture==arm)&(pair_curves.masker==masker)&pair_curves.pair_eligible]
        axes[i,j].scatter(g.js_divergence,g.abs_D,s=9,alpha=.2,color=MASK_COLOR[masker],rasterized=True)
        axes[i,j].set(xlim=(0,1),ylim=(0,None),xlabel='JSD (base-2 divergence)',ylabel='Absolute change in advantage |D|',
                      title=ARM_LABEL[arm]+' / '+MASK_LABEL[masker])
save_figure(fig,'jsd_vs_absolute_swap_effect')

fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
position=np.arange(len(strata))
for i,(arm,masker) in enumerate(strata):
    row=coverage[(coverage.architecture==arm)&(coverage.masker==masker)].iloc[0]
    matched=100*row.n_pairs_matched/row.n_pairs_total
    before=100*row.n_pairs_stable_before_matching/row.n_pairs_total
    after=100*row.n_pairs_stable_after_matching/row.n_pairs_total
    axes[0].barh(i,matched,color=MASK_COLOR[masker])
    axes[0].text(min(matched+1,92),i,f'{int(row.n_pairs_matched)}/{int(row.n_pairs_total)}',va='center',fontsize=8)
    axes[1].barh(i-.15,before,height=.28,color='.75',label='Before RMS matching' if i==0 else None)
    axes[1].barh(i+.15,after,height=.28,color=MASK_COLOR[masker],label='After RMS matching' if i==0 else None)
    axes[1].annotate(f'{int(row.n_pairs_stable_after_matching)} matched',(after,i+.15),xytext=(4,0),
                     textcoords='offset points',va='center',fontsize=8)
axes[0].set(xlim=(0,100),xlabel='Percent of all planned directed pairs',title='Full-curve matching coverage')
maximum=max([100*r.n_pairs_stable_before_matching/r.n_pairs_total for r in coverage.itertuples(index=False)],default=0)
axes[1].set(xlim=(0,max(10,maximum*1.4)),xlabel='Percent of all planned directed pairs',title='Stable-output subset (separate scale)')
for ax in axes:ax.set_yticks(position,labels)
axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.18),frameon=False,ncol=1)
save_figure(fig,'swap_matching_and_stable_coverage')

fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
reference_plot=reference_ci[reference_ci.endpoint=='G_real_minus_placebo']
legend_done=set();yticks=[]
for i,row in enumerate(reference_plot.itertuples(index=False)):
    color='#2a628f' if row.reference=='circular_shift' else '#a55a28'
    label=row.reference.replace('_',' ') if row.reference not in legend_done else None
    if np.isfinite(row.estimate):
        ax.plot(row.estimate,i,MARKER[row.architecture],color=color,label=label)
        if np.isfinite(row.ci_low):ax.plot([row.ci_low,row.ci_high],[i,i],color=color)
    legend_done.add(row.reference)
    yticks.append(ARM_LABEL[row.architecture]+' / '+MASK_LABEL[row.masker]+' / '+row.reference.replace('_',' '))
ax.set_yticks(range(len(yticks)),yticks);ax.axvline(0,color='.5',ls=':')
ax.set(xlabel='Transferred minus placebo advantage (common support; nominal subject CI)')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.14),ncol=2,frameon=False)
save_figure(fig,'swap_vs_jsd_matched_references')
'''


if __name__ == '__main__':
    write_notebook(build_notebook(),'20_fixed_model_ranking_swap.ipynb')
