"""Generate the new exploratory ECE/Brier notebook; never execute notebook cells.
All research definitions are emitted into the notebook. No original source is edited.
"""
from pathlib import Path
import nbformat as nbf


def build_notebook():
    cells = []
    def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
    def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))
    md(r"""
# Subject-level OOF probability audit: ECE and Brier score

**Additional exploratory analysis**, designed 2026-10-14 after E0/E1/E2/bridge
results were known. Identity: `CALIBRATION_OOF_EXPLORATORY_V1`. Historical
preregistration and all previous results remain unchanged. This notebook reads
stored predictions only: **no training, model loading, inference, interventions
or fitting of a calibrator**. It does not modify any manuscript.

Run from the project kernel **Skripsi (.venv)**: Restart Kernel → Run All.
The input banks and hashes are fixed, with no automatic latest-run selection.
Each run writes a fresh directory under `results/calibration_oof/` and
`figures/calibration_oof/`. Confirm alignment before interpreting any scores.

Primary: mean of per-seed ECE/BS from mean recording probability per subject.
This is not a metric of the mean probability across seeds. PD event probability
is binned, not maximum-class confidence. Ten equal-width bins are primary;
five equal-width and ten quantile bins are predetermined sensitivities.
Optional ensemble evaluation defaults off and always has separate tables.

Intervals resample subjects within observed classes, preserving all architecture
and seed outputs; they are conditional on available models/splits/seeds and class
composition. All intervals are exploratory and unadjusted for multiplicity.
""")
    code(r"""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import json, platform, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'src/audit_e2.py').is_file())
PROJECT = ROOT/'Q1-amin/manuscript_array_latex_v4'
ANALYSIS_KIND = 'CALIBRATION_OOF_EXPLORATORY_V1'
IMPLEMENTATION_REVISION = 'v4_path_alignment_and_handoff'
DESIGN_DATE = '2026-10-14'
N_BOOT, BOOT_SEED, CONFIDENCE = 5000, 1914, .95
MIN_CI_SUBJECTS, MIN_FINITE_FRACTION = 10, .95
INCLUDE_ENSEMBLE = False  # separate optional estimand; never replaces seed-average scores
SCHEMES = {'equal_width_10':('equal_width',10), 'equal_width_5':('equal_width',5),
           'quantile_10':('quantile',10)}
PRIMARY_SCHEME = 'equal_width_10'
E0 = ROOT/'results/e0_e1/20261012T035813806939Z'
E2 = ROOT/'results/e2/20261012T050124548170Z'
EXPECTED_EVALUATION = 'c4b1d743f22d6b9bbbc5b2a2a8fcd7d913ba2543108895f8e70de9b4c1b79823'
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUT = ROOT/'results/calibration_oof'/RUN_ID
FIG = ROOT/'figures/calibration_oof'/RUN_ID
OUT.mkdir(parents=True,exist_ok=False); FIG.mkdir(parents=True,exist_ok=False)
manifest = dict(analysis_kind=ANALYSIS_KIND,analysis_design_date=DESIGN_DATE,run_id=RUN_ID,
  implementation_revision=IMPLEMENTATION_REVISION,manuscript_project=str(PROJECT.relative_to(ROOT)),
  analysis_status='additional_exploratory_after_E0_E1_E2_RMS_known',completion_status='initializing',
  training_performed=False,model_inference_performed=False,new_interventions_performed=False,
  calibrator_fitted=False,old_preregistration_changed=False,notebook_execution_by='user',
  bootstrap=dict(n=N_BOOT,seed=BOOT_SEED,confidence=CONFIDENCE,sampling_unit='subject',
    stratification='observed class counts retained',paired_architectures_and_seeds=True),
  estimand='record probability mean within subject; metric per seed; mean metrics across fixed seeds',
  probability_binned='P(PD), positive-class event probability, not maximum-class confidence',
  schemes=SCHEMES,primary_scheme=PRIMARY_SCHEME,include_ensemble=INCLUDE_ENSEMBLE,
  quantile_policy='recompute edges per seed and bootstrap replicate; collapse duplicates',
  empty_bin_policy='zero metric weight; undefined frequency/mean; count zero',
  multiplicity_adjustment='none; exploratory',historical_model_provenance='incomplete training-time identities',
  python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)
checks=[]
def persist():
    (OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2))
    pd.DataFrame(checks).to_csv(OUT/'alignment_checks.csv',index=False)
def check(condition,name):
    checks.append(dict(check=name,passed=bool(condition)))
    if not condition:
        manifest.update(completion_status='alignment_failed',failed_check=name);persist()
        raise ValueError(name)
def phase(name):
    manifest['completion_status']=name
    persist()
persist()
print('Run:',RUN_ID,'; results:',OUT,'; no model execution')
""")
    md('## Input alignment and provenance\nNew-bank recording predictions are recovered from the two sides of the stored E1 pairs, with duplicates checked before subject aggregation. Historical subject probabilities are used as already aggregated; their training-time model/fold provenance cannot be reconstructed by this notebook.')
    code(r"""
expected_inputs=json.loads((PROJECT/'analysis/calibration_inputs.json').read_text())
input_rows=[]
for relative,expected in expected_inputs.items():
    path=ROOT/relative
    check(path.is_file(),'input exists: '+relative)
    observed=sha256(path.read_bytes()).hexdigest()
    check(observed==expected,'input SHA256: '+relative)
    input_rows.append(dict(path=relative,sha256=observed,bytes=path.stat().st_size))
pd.DataFrame(input_rows).to_csv(OUT/'input_inventory.csv',index=False)
builder=PROJECT/'analysis/build_ece_brier_notebook.py'
notebook=PROJECT/'analysis/19_ece_brier_oof.ipynb'
manifest['document_sources']={p.name:sha256(p.read_bytes()).hexdigest() for p in (builder,notebook)}
notebook_payload=json.loads(notebook.read_text())
manifest['notebook_source_sha256']=sha256(json.dumps(
    [(c['cell_type'],c['source']) for c in notebook_payload['cells']],sort_keys=True).encode()).hexdigest()
historical_manifest=json.loads((E0/'analysis_manifest.json').read_text())
check(historical_manifest['run_id']==E0.name,'historical bank run identity')
check(historical_manifest['completion_status']=='complete','historical bank complete')
check(set(historical_manifest['seeds'])==set(range(10)),'historical bank ten fixed seeds')
base_manifest=json.loads((E2/'analysis_manifest.json').read_text())
check(base_manifest['evaluation_id']==EXPECTED_EVALUATION,'current checkpoint bank evaluation identity')
check(base_manifest['completion_status']=='complete','current bank complete')
manifest['source_banks']={
 'historical_E0':dict(run_id=E0.name,cohorts=['uci395','newhandpd'],n_seeds=10,
                    training_time_identities='incomplete historical provenance'),
 'current_E1_E2':dict(run_id=E2.name,evaluation_id=EXPECTED_EVALUATION,
                    cohorts=['newhandpd'],n_seeds=3,n_checkpoints=30)}

def read_csv(path):
    return pd.read_csv(path,dtype={'subject':str,'record_id':str,'architecture':str,'cohort':str,
                                  'model_id':str,'e1_model_id_a':str,'e1_model_id_b':str})
models=read_csv(E2/'model_inventory.csv'); records=read_csv(E2/'record_manifest.csv')
pairs=read_csv(E2/'e1_pairs_new.csv')
check(not models.duplicated(['architecture','seed','fold']).any(),'unique model per architecture/seed/fold')
check(len(models)==30 and set(models.seed)=={0,1,2},'30 checkpoints and three fixed seeds')
check(set(map(tuple,models[['architecture','seed','fold']].to_numpy()))==
      {(a,s,f) for a in ('gru','mamba2') for s in range(3) for f in range(5)},
      'complete architecture/seed/fold checkpoint grid')
check(not records.record_id.duplicated().any(),'unique recording identities')
check(len(records)==264 and records.subject.nunique()==66,'264 recordings and 66 current-bank subjects')
check(records.groupby('subject').fold_actual.nunique().eq(1).all(),'subject belongs to one fold')
for row in models.itertuples():
    checkpoint=ROOT/row.checkpoint
    check(checkpoint.is_file(),'checkpoint exists: '+row.model_id)
    check(sha256(checkpoint.read_bytes()).hexdigest()==row.checkpoint_sha256,'checkpoint SHA256: '+row.model_id)
parts=[]
for side in ('a','b'):
    part=pairs[['cohort','architecture','idx','record_id','subject','label','fold_actual',
                'seed_'+side,'probability_'+side,'e1_model_id_'+side]].rename(columns={
        'fold_actual':'fold','seed_'+side:'seed','probability_'+side:'probability','e1_model_id_'+side:'model_id'})
    parts.append(part)
record_predictions=pd.concat(parts,ignore_index=True)
keys=['architecture','seed','record_id']
for key,group in record_predictions.groupby(keys):
    check(group.subject.nunique()==1 and group.label.nunique()==1 and group.fold.nunique()==1
          and group.model_id.nunique()==1,'duplicated-pair identity '+str(key))
    check(np.allclose(group.probability,group.probability.iloc[0],atol=1e-12,rtol=1e-12),
          'duplicated-pair probability '+str(key))
record_predictions=record_predictions.drop_duplicates(keys)
joined=record_predictions.merge(records[['record_id','subject','label','fold_actual']],on='record_id',
   validate='many_to_one',suffixes=('','_manifest'),how='outer',indicator=True)
check(joined['_merge'].eq('both').all(),'new-bank recording manifest intersection')
check(joined.subject.eq(joined.subject_manifest).all() and joined.label.eq(joined.label_manifest).all()
      and joined.fold.eq(joined.fold_actual).all(),'new-bank subject/label/fold alignment')
joined=record_predictions.merge(models[['architecture','seed','fold','model_id']],
   on=['architecture','seed','fold'],validate='many_to_one',suffixes=('','_inventory'),how='left',indicator=True)
check(joined['_merge'].eq('both').all(),'every current prediction has a checkpoint identity')
check(joined.model_id.eq(joined.model_id_inventory).all(),'prediction-to-checkpoint identity')
check(record_predictions.groupby(['architecture','seed']).size().eq(len(records)).all(),
      'one prediction per recording per current-bank model seed')
recovered=record_predictions.groupby(['cohort','architecture','seed','subject','label'],as_index=False).agg(
    probability=('probability','mean'),n_records=('record_id','size'))
current=read_csv(E2/'new_probability_subject_seed.csv')
key=['cohort','architecture','seed','subject','label']
j=current.merge(recovered,on=key,how='outer',validate='one_to_one',suffixes=('','_recovered'),indicator=True)
check(j['_merge'].eq('both').all(),'current subject-probability support')
check(np.allclose(j.probability,j.probability_recovered,atol=1e-12,rtol=1e-12)
      and j.n_records.eq(j.n_records_recovered).all(),'record mean -> subject probability replay')
current['bank']='current_E1_E2'; current['prediction_kind']='per_seed'
historical=read_csv(E0/'e0_probability_subject_seed.csv')
historical['bank']='historical_E0'; historical['prediction_kind']='per_seed'
probs=pd.concat([historical,current],ignore_index=True)
check(np.isfinite(probs.probability).all() and probs.probability.between(0,1).all(),'finite event probabilities')
check(set(probs.label)=={0,1},'binary PD event labels')
check(not probs.duplicated(['bank','cohort','architecture','seed','subject']).any(),'unique subject/seed probabilities')
EXPECTED_SUPPORT={('historical_E0','uci395'):(77,62,15,10),
                  ('historical_E0','newhandpd'):(66,31,35,10),
                  ('current_E1_E2','newhandpd'):(66,31,35,3)}
check(set(probs.groupby(['bank','cohort']).groups)==set(EXPECTED_SUPPORT),
      'exact historical/current bank and cohort combinations')
denominators=[]
for (bank,cohort),group in probs.groupby(['bank','cohort']):
    count,n_pd,n_hc,n_seeds=EXPECTED_SUPPORT[(bank,cohort)];seeds=range(n_seeds)
    check(set(group.architecture)=={'gru','mamba2'},'both architectures: '+bank+'/'+cohort)
    check(set(group.seed)==set(seeds),'fixed seed set: '+bank+'/'+cohort)
    check(group.groupby(['architecture','seed']).size().eq(count).all(),'complete subject support: '+bank+'/'+cohort)
    check(group.groupby('subject').label.nunique().eq(1).all(),'consistent labels: '+bank+'/'+cohort)
    subjects=set(group.subject)
    check(all(set(g.subject)==subjects for _,g in group.groupby(['architecture','seed'])),
          'identical subject identities across all models/seeds: '+bank+'/'+cohort)
    for (architecture,seed),h in group.groupby(['architecture','seed']):
        check(int(h.label.sum())==n_pd and int((h.label==0).sum())==n_hc,
              'PD/HC denominator: '+bank+'/'+cohort+'/'+architecture+'/'+str(seed))
        denominators.append(dict(bank=bank,cohort=cohort,architecture=architecture,seed=int(seed),
            n_subjects=len(h),n_pd=n_pd,n_hc=n_hc,n_seeds_in_bank=n_seeds,
            n_record_probabilities=int(h.n_records.sum()),support='all OOF subjects; no RMS-matching filter'))
record_predictions.to_csv(OUT/'current_record_probability_identity.csv',index=False)
probs.to_csv(OUT/'subject_probabilities_used.csv',index=False)
models.to_csv(OUT/'current_model_inventory.csv',index=False)
j.to_csv(OUT/'current_subject_probability_replay.csv',index=False)
pd.DataFrame(denominators).to_csv(OUT/'bank_denominators.csv',index=False)
manifest['subject_support']={bank+'/'+cohort:dict(n_subjects=n,n_pd=pd_n,n_hc=hc_n,n_seeds=s)
                            for (bank,cohort),(n,pd_n,hc_n,s) in EXPECTED_SUPPORT.items()}
phase('aligned')
display(probs.groupby(['bank','cohort','architecture','seed']).size().rename('n_subjects').reset_index())
""")
    md(r"""
## Definitions fixed before inspecting calibration results

For each seed, BS = mean((P(PD) − y)²). ECE sums bin frequency weights times
|mean probability − observed PD proportion|. Lower scores are preferable for
these estimands. Brier includes discrimination/sharpness as well as calibration.
Equal-width intervals are [left,right), with 1 in the final bin. Empty bins are
not plotted as zero PD frequencies. Quantile edges depend on probabilities only;
duplicate boundaries collapse, and effective bin count is recorded. This is
estimation of calibration error, **not fitting or applying a calibration map**.
""")
    code(r"""
def bin_edges(probability,kind,bins):
    if kind=='equal_width': return np.linspace(0.,1.,bins+1)
    return np.unique(np.r_[0.,np.quantile(probability,np.arange(1,bins)/bins),1.])

def bin_summary(y,p,scheme):
    kind,requested=SCHEMES[scheme]; edges=bin_edges(p,kind,requested)
    b=np.clip(np.searchsorted(edges,p,side='right')-1,0,len(edges)-2)
    n=np.bincount(b,minlength=len(edges)-1).astype(int)
    sp=np.bincount(b,weights=p,minlength=len(n)); sy=np.bincount(b,weights=y,minlength=len(n))
    mean_p=np.divide(sp,n,out=np.full(len(n),np.nan),where=n>0)
    frequency=np.divide(sy,n,out=np.full(len(n),np.nan),where=n>0)
    table=pd.DataFrame(dict(bin=np.arange(len(n)),left=edges[:-1],right=edges[1:],
       n_subjects=n,mean_probability=mean_p,pd_frequency=frequency,
       empty=n==0,requested_bins=requested,effective_bins=len(n)))
    ece=float(np.nansum(n/len(y)*np.abs(mean_p-frequency)))
    return ece,table

def metric_vector(y,p):
    out={'brier':float(np.mean((p-y)**2))}
    for scheme in SCHEMES:out['ece_'+scheme]=bin_summary(y,p,scheme)[0]
    return out

metrics=['brier', *['ece_'+scheme for scheme in SCHEMES]]

def seed_average_metrics(y,matrix):
    vectors=[metric_vector(y,matrix[:,s]) for s in range(matrix.shape[1])]
    return np.mean([[v[m] for m in metrics] for v in vectors],axis=0)
scores=[]; reliability=[]
phase('computing_per_seed_scores')
for (bank,cohort,architecture,seed),g in probs.groupby(['bank','cohort','architecture','seed']):
    y=g.label.to_numpy(float); probability=g.probability.to_numpy(float)
    for metric,estimate in metric_vector(y,probability).items():
        scores.append(dict(bank=bank,cohort=cohort,architecture=architecture,seed=int(seed),
           prediction_kind='per_seed',metric=metric,estimate=estimate,n_subjects=len(g)))
    for scheme in SCHEMES:
        _,bins=bin_summary(y,probability,scheme)
        bins=bins.assign(bank=bank,cohort=cohort,architecture=architecture,seed=int(seed),scheme=scheme)
        reliability.append(bins)
scores=pd.DataFrame(scores); reliability=pd.concat(reliability,ignore_index=True)
scores.to_csv(OUT/'per_seed_scores.csv',index=False)
reliability.to_csv(OUT/'reliability_bins_per_seed.csv',index=False)
display(scores)
phase('per_seed_scores_complete')
""")
    md('## Paired subject inference\nWithin each bank/cohort, the same class-stratified subject draws are used for both architectures and every fixed seed. Each draw recomputes each seed metric before averaging. No seed bootstrap, paired-record independence assumption, multiplicity correction or calibration fitting is used. Cross-bank or cross-cohort probability-quality differences are not pooled into a primary claim.')
    code(r"""
summary_rows=[]; contrast_rows=[]; bootstrap_rows=[]; ensemble_rows=[]
phase('subject_bootstrap_running')
rng=np.random.default_rng(BOOT_SEED)
for (bank,cohort),g in probs.groupby(['bank','cohort'],sort=True):
    subject_order=sorted(g.subject.unique())
    labels=g.drop_duplicates('subject').set_index('subject').loc[subject_order,'label'].to_numpy(float)
    seeds=sorted(g.seed.unique()); architectures=['gru','mamba2']
    arrays={a:g[g.architecture==a].pivot(index='subject',columns='seed',values='probability')
              .loc[subject_order,seeds].to_numpy(float) for a in architectures}
    strata=[np.flatnonzero(labels==c) for c in (0,1)]
    draws=np.vstack([np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in strata])
                     for _ in range(N_BOOT)])
    np.savez_compressed(OUT/(bank+'_'+cohort+'_subject_bootstrap.npz'),indices=draws,
                       subjects=np.asarray(subject_order),labels=labels)
    estimates={a:seed_average_metrics(labels,arrays[a]) for a in architectures}
    boot={a:np.empty((N_BOOT,len(metrics))) for a in architectures}
    for b,ix in enumerate(draws):
        for a in architectures:
            boot[a][b]=seed_average_metrics(labels[ix],arrays[a][ix,:])
        if (b+1)%1000==0:print(bank,cohort,b+1,'/',N_BOOT)
    def interval(values):
        valid=values[np.isfinite(values)]
        ok=len(subject_order)>=MIN_CI_SUBJECTS and len(valid)>=MIN_FINITE_FRACTION*N_BOOT
        low,high=np.quantile(valid,[(1-CONFIDENCE)/2,1-(1-CONFIDENCE)/2]) if ok else (np.nan,np.nan)
        return float(low),float(high),len(valid),'reported' if ok else 'suppressed'
    for a in architectures:
        for j,m in enumerate(metrics):
            lo,hi,n,status=interval(boot[a][:,j])
            summary_rows.append(dict(bank=bank,cohort=cohort,architecture=a,metric=m,
                prediction_kind='mean_of_per_seed_metric',estimate=float(estimates[a][j]),
                ci_low=lo,ci_high=hi,n_subjects=len(subject_order),n_seeds=len(seeds),
                n_boot_valid=n,ci_status=status))
    for j,m in enumerate(metrics):
        values=boot['mamba2'][:,j]-boot['gru'][:,j];lo,hi,n,status=interval(values)
        contrast_rows.append(dict(bank=bank,cohort=cohort,contrast='BiMamba-2 minus BiGRU',metric=m,
            prediction_kind='paired_difference_of_mean_per_seed_metrics',
            estimate=float(estimates['mamba2'][j]-estimates['gru'][j]),ci_low=lo,ci_high=hi,
            n_subjects=len(subject_order),n_seeds=len(seeds),n_boot_valid=n,ci_status=status))
    for b in range(N_BOOT):
        for j,m in enumerate(metrics):
            bootstrap_rows.append(dict(bank=bank,cohort=cohort,draw=b,metric=m,
                gru=boot['gru'][b,j],mamba2=boot['mamba2'][b,j],contrast=boot['mamba2'][b,j]-boot['gru'][b,j]))
    if INCLUDE_ENSEMBLE:
        for a in architectures:
            ensemble=arrays[a].mean(axis=1)
            e=metric_vector(labels,ensemble)
            eb=np.asarray([seed_average_metrics(labels[ix],ensemble[ix,None]) for ix in draws])
            for j,m in enumerate(metrics):
                lo,hi,n,status=interval(eb[:,j])
                ensemble_rows.append(dict(bank=bank,cohort=cohort,architecture=a,metric=m,
                   prediction_kind='metric_of_seed_mean_probability',estimate=e[m],ci_low=lo,ci_high=hi,
                   n_subjects=len(subject_order),n_seeds=len(seeds),n_boot_valid=n,ci_status=status))
summary=pd.DataFrame(summary_rows); contrasts=pd.DataFrame(contrast_rows)
summary.to_csv(OUT/'mean_per_seed_scores_ci.csv',index=False)
contrasts.to_csv(OUT/'paired_architecture_contrasts_ci.csv',index=False)
pd.DataFrame(bootstrap_rows).to_csv(OUT/'bootstrap_scores.csv',index=False)
pd.DataFrame(ensemble_rows,columns=summary.columns).to_csv(OUT/'ensemble_scores_ci.csv',index=False)
score_checks=[]
for row in summary.itertuples():
    h=scores[(scores.bank==row.bank)&(scores.cohort==row.cohort)&
             (scores.architecture==row.architecture)&(scores.metric==row.metric)]
    passed=len(h)==row.n_seeds and np.isclose(h.estimate.mean(),row.estimate,atol=1e-12,rtol=1e-12)
    score_checks.append(dict(check='per-seed mean replay',bank=row.bank,cohort=row.cohort,
        architecture=row.architecture,metric=row.metric,passed=bool(passed)))
    check(passed,'summary per-seed mean replay: '+row.bank+'/'+row.cohort+'/'+row.architecture+'/'+row.metric)
for row in contrasts.itertuples():
    h=summary[(summary.bank==row.bank)&(summary.cohort==row.cohort)&(summary.metric==row.metric)].set_index('architecture')
    passed=np.isclose(h.loc['mamba2','estimate']-h.loc['gru','estimate'],row.estimate,atol=1e-12,rtol=1e-12)
    score_checks.append(dict(check='paired architecture contrast replay',bank=row.bank,cohort=row.cohort,
        architecture='mamba2 minus gru',metric=row.metric,passed=bool(passed)))
    check(passed,'paired contrast replay: '+row.bank+'/'+row.cohort+'/'+row.metric)
pd.DataFrame(score_checks).to_csv(OUT/'score_consistency_checks.csv',index=False)
phase('subject_bootstrap_complete')
display(summary);display(contrasts)
""")
    md('## Reliability diagrams and uncertainty tables\nSeed curves and bin occupancy are displayed separately, without pooling predictions across seeds. Quantile diagrams use each seed\'s own edges. The displayed bin frequencies are descriptive; subject-bootstrap uncertainty is reported for the scalar scores and paired contrasts, not fabricated as per-bin bands.')
    code(r"""
MODEL_NAMES={'gru':'BiGRU','mamba2':'BiMamba-2'}
phase('rendering_reliability_diagrams')
for (bank,cohort,architecture,scheme),g in reliability.groupby(['bank','cohort','architecture','scheme']):
    fig,(ax,count_ax)=plt.subplots(2,1,figsize=(7.2,6.0),gridspec_kw={'height_ratios':[3,1]},layout='constrained')
    ax.plot([0,1],[0,1],color='black',ls='--',lw=1,label='Perfect calibration')
    for seed,h in g.groupby('seed'):
        observed=h[h.n_subjects>0]
        ax.scatter(observed.mean_probability,observed.pd_frequency,s=20,
                alpha=.75,label='Seed '+str(seed))
        count_ax.step(h.bin,h.n_subjects,where='mid',alpha=.7,label='Seed '+str(seed))
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean subject P(PD) in bin',ylabel='Observed PD proportion',
        title=MODEL_NAMES[architecture]+' | '+bank+' | '+cohort+' | '+scheme)
    ax.legend(fontsize=7,ncol=3,loc='best')
    count_ax.set(xlabel='Bin index (zero-based; edges in CSV)',ylabel='Subjects per bin',ylim=(0,None))
    count_ax.grid(axis='y',alpha=.2)
    stem=bank+'_'+cohort+'_'+architecture+'_'+scheme
    fig.savefig(FIG/(stem+'.pdf'));fig.savefig(FIG/(stem+'.svg'));plt.close(fig)
for metric in metrics:
    t=summary[summary.metric==metric].reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(8.0,4.0),layout='constrained')
    for i,row in t.iterrows():
        if np.isfinite(row.ci_low) and np.isfinite(row.ci_high):
            ax.plot([row.ci_low,row.ci_high],[i,i],color='black',lw=1.2)
        ax.plot(row.estimate,i,'o',color='#0072B2' if row.architecture=='gru' else '#D55E00')
    ax.set_yticks(range(len(t)),[r.bank+' / '+r.cohort+' / '+MODEL_NAMES[r.architecture] for r in t.itertuples()])
    ax.set(xlabel='Mean per-seed '+metric+' (95% subject-bootstrap CI)',title='Probability quality; fixed models and splits')
    ax.grid(axis='x',alpha=.2);fig.savefig(FIG/(metric+'_scores.pdf'));plt.close(fig)
manifest.update(completion_status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
    n_subject_probability_rows=len(probs),n_current_record_probability_rows=len(record_predictions),
    n_checkpoints_verified=len(models),n_alignment_checks=len(checks),all_alignment_checks_passed=all(c['passed'] for c in checks),
    outputs=sorted({f.name for f in OUT.iterdir() if f.is_file()}|{'output_inventory.csv'}),
    figures=str(FIG.relative_to(ROOT)),ensemble_estimand_executed=INCLUDE_ENSEMBLE)
persist()
inventory=[]
for directory,kind in ((OUT,'result'),(FIG,'figure')):
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name!='output_inventory.csv':
            inventory.append(dict(kind=kind,path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,
                                  sha256=sha256(path.read_bytes()).hexdigest()))
pd.DataFrame(inventory).to_csv(OUT/'output_inventory.csv',index=False)
import zipfile
review_bundle=OUT.parent/('calibration_review_'+RUN_ID+'.zip')
try:
    with zipfile.ZipFile(review_bundle,'x',zipfile.ZIP_DEFLATED) as archive:
        for directory,kind in ((OUT,'results'),(FIG,'figures')):
            for path in sorted(directory.iterdir()):
                if path.is_file():archive.write(path,kind+'/'+path.name)
except Exception:
    manifest['completion_status']='handoff_packaging_failed'
    persist()
    raise
print('Complete. Results are exploratory; no manuscript has been updated:',OUT)
print('Review bundle to provide:',review_bundle)
""")
    md(r"""
## Interpretation checklist

- Verify all alignment checks, bank identities and seed/subject denominators.
- Report the primary 10-equal-width estimator before both sensitivities, including
  empty/effective bins; never select the binning with the smallest ECE.
- Read reliability curves with their subject counts. Small bins are noisy; low ECE
  alone does not prove clinical calibration, and Brier is not calibration alone.
- A contrast CI including zero does not establish equivalence. These intervals do
  not include fresh training, a new cohort, new splits or a seed-population bootstrap.
- If ensemble evaluation is enabled, report its metric-of-mean-probability estimand
  separately from the mean-of-per-seed metrics. Do not replace the E0 estimator.
- Send `calibration_review_<run_id>.zip` for incorporation into v4; no code here
  inserts numbers into the manuscript. Calibration does not establish attention
  faithfulness or physiological validity.

Journal methodological references: Silva Filho et al. (2023),
https://doi.org/10.1007/s10994-023-06336-7; Van Calster et al. (2019),
https://doi.org/10.1186/s12916-019-1466-7.
""")
    notebook=nbf.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Skripsi (.venv)','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.11'},
        'analysis_identity':'CALIBRATION_OOF_EXPLORATORY_V1',
        'analysis_status':'exploratory; pending user execution'})
    return notebook


if __name__=='__main__':
    destination=Path(__file__).resolve().parent/'19_ece_brier_oof.ipynb'
    nbf.write(build_notebook(),destination)
    print('Generated notebook only; no cells executed:',destination)
