"""Generate notebook 19 without executing any scientific cell.

Existing builder source is parsed as text; it is never imported or executed.
Verified legacy preflight functions are embedded in the new notebook, which
contains the complete scientific source. New outputs never replace old runs.
"""
from pathlib import Path
import ast
import hashlib
import json

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "plans/attention_audit_extension_20261016"


def legacy_cells(number):
    path = ROOT / f"scripts/build_nb_{number}.py"
    pins = json.loads((PLAN / "source_pins.json").read_text())
    expected = pins["input_sha256"][str(path.relative_to(ROOT))]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Legacy builder source changed: {path}")
    tree = ast.parse(path.read_text())
    cells = []
    for name in ("build_notebook", "_finish_notebook"):
        for fn in tree.body:
            if not isinstance(fn, ast.FunctionDef) or fn.name != name:
                continue
            for node in fn.body:
                if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name)
                        and node.value.func.id in {"md", "code"}):
                    cells.append((node.value.func.id,
                                  ast.literal_eval(node.value.args[0]).strip()))
    return cells


def legacy_functions(number, names):
    found = {}
    for kind, source in legacy_cells(number):
        if kind != "code":
            continue
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                if node.name in found:
                    raise ValueError(f"Duplicate legacy definition: {node.name}")
                found[node.name] = ast.get_source_segment(source, node)
    if set(found) != set(names):
        raise ValueError(f"Missing legacy functions: {set(names) - set(found)}")
    return "\n\n".join(found[name] for name in names)


def markdown(cells, source):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(cells, source):
    cells.append(nbf.v4.new_code_cell(source.strip()))


def add_preflight(cells, notebook_name, builder_name, family, analysis_kind):
    """Embed the checked notebook-18 preflight, without its analyses or execution."""
    code(cells, r'''
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
from itertools import combinations
import json, pickle, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.spatial.distance import jensenshannon
from scipy.stats import rankdata
from IPython.display import display

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
            if (p/'src/audit_e2.py').is_file())
PLAN = ROOT/'plans/attention_audit_extension_20261016'
BASELINE = ROOT/'results/e2/20261012T050124548170Z'
AMPLITUDE = ROOT/'results/e2_amplitude/20261014T024232464643Z'
BRIDGE = ROOT/'results/e1_e2_rms_bridge/20261014T041914953496Z'
EXPECTED_EVALUATION = 'c4b1d743f22d6b9bbbc5b2a2a8fcd7d913ba2543108895f8e70de9b4c1b79823'
EXPECTED_CONTROL = '1d839c2be1e4cc601eeaffc85acc7e380265bf076940f2a62b76648bee9a8ac7'
NOTEBOOK = ROOT/'notebooks/NOTEBOOK_NAME'
BUILDER = ROOT/'scripts/BUILDER_NAME'
ANALYSIS_KIND = 'ANALYSIS_NAME'
FAMILY = 'FAMILY_NAME'
DESIGN_DATE = '2026-10-16'
N_BOOT, BOOT_SEED, CONFIDENCE = 5000, 1902, .95
MIN_CI_SUBJECTS = 10
DENOM_FLOORS = (1e-8, 1e-6, 1e-4)
PRIMARY_FLOOR = DENOM_FLOORS[0]
METRICS = ('abs_delta_logit', 'signed_margin_drop', 'predicted_class_prob_drop')
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUT, FIG = ROOT/'results'/FAMILY/RUN_ID, ROOT/'figures'/FAMILY/RUN_ID
OUT.mkdir(parents=True, exist_ok=False)
FIG.mkdir(parents=True, exist_ok=False)
run_manifest = dict(run_id=RUN_ID,analysis_kind=ANALYSIS_KIND,analysis_design_date=DESIGN_DATE,
    completion_status='initializing',analysis_status='additional_exploratory',
    old_preregistration_changed=False,public_preregistration=False,training_performed=False,
    model_inference_performed=False,new_interventions_performed=False,
    baseline_run=BASELINE.name,amplitude_run=AMPLITUDE.name,bridge_run=BRIDGE.name)
(OUT/'analysis_manifest.json').write_text(json.dumps(run_manifest,indent=2),encoding='utf-8')
print('New exploratory output:', OUT)
'''.replace("NOTEBOOK_NAME", notebook_name).replace("BUILDER_NAME", builder_name)
        .replace("ANALYSIS_NAME", analysis_kind).replace("FAMILY_NAME", family))
    old = legacy_cells(18)
    code(cells, old[3][1])
    code(cells, r'''
pins = json.loads((PLAN/'source_pins.json').read_text())
track(PLAN/'source_pins.json')
track(PLAN/'PROTOCOL.md', pins['protocol_sha256'])
for rel, expected in pins['input_sha256'].items():
    track(ROOT/rel, expected)
track(ROOT/'scripts/build_nb_19.py')
''')
    for idx in (4, 5, 7):
        if old[idx][0] != "code":
            raise ValueError("Legacy preflight layout changed")
        code(cells, old[idx][1])
    code(cells, r'''
bridge_manifest = json.loads((BRIDGE/'analysis_manifest.json').read_text())
bspec = bridge_manifest['analysis_spec']
check(bridge_manifest['completion_status']=='complete'
      and bspec['baseline_evaluation_id']==EXPECTED_EVALUATION
      and bspec['amplitude_control_id']==EXPECTED_CONTROL, 'pinned bridge complete and same bank')
check(source_digest(ROOT/bridge_manifest['scientific_source'])
      ==bridge_manifest['notebook_source_sha256'], 'saved bridge scientific source')
for rel, expected in bspec['input_hashes'].items():
    track(ROOT/rel, expected)
pairs = read_csv(BRIDGE/'bridge_pairs_aopc.csv')
budget_pairs = read_csv(BRIDGE/'bridge_pairs_budget.csv')
for name in ('pair_eligible','normalization_eligible','matched_curve_complete_a','matched_curve_complete_b'):
    bool_column(pairs,name)
''')
    code(cells, legacy_functions(18, ("pair_table",)))
    code(cells, r'''
rebuilt = pair_table(contrasts,level='aopc',value_prefix='aopc_',complete_column='matched_curve_complete')
PAIR_ID = ['architecture','idx','record_id','subject','label','fold','seed_a','seed_b',
           'model_id_a','model_id_b','masker']
unique(pairs,PAIR_ID,'stored bridge pair ID')
unique(rebuilt,PAIR_ID,'rebuilt bridge pair ID')
comparison = pairs.merge(rebuilt,on=PAIR_ID,how='outer',validate='one_to_one',
                         suffixes=('_stored','_rebuilt'),indicator=True)
check((comparison['_merge']=='both').all(),'saved/rebuilt bridge identical pair support')
for field in ('pair_eligible','normalization_eligible'):
    check(np.array_equal(comparison[field+'_stored'],comparison[field+'_rebuilt']),
          'saved/rebuilt eligibility '+field)
for field in ('js_divergence','abs_delta_logit','abs_delta_prob','effect_sum_a','effect_sum_b',
              'abs_delta_gain_abs_delta_logit','abs_delta_relative_gain'):
    finite_close(comparison[field+'_stored'],comparison[field+'_rebuilt'],'saved/rebuilt '+field)
del rebuilt,comparison
for row in pairs[PAIR_ID].itertuples(index=False):
    check(row.idx in specs[row.model_id_a]['split']['test_indices']
          and row.idx in specs[row.model_id_b]['split']['test_indices'], 'pair shares held-out recording')
SUBJECT_POOL = subjects.subject.to_numpy()
SUBJECT_INDEX = {s:i for i,s in enumerate(SUBJECT_POOL)}
pd.DataFrame(checks).to_csv(OUT/'alignment_checks.csv',index=False)
print('Historical bank, E1 maps, intervention controls and bridge identities verified.')
''')


def inference_source(*, directed=False):
    names=("cluster_counts", "weighted_corr", "tie_groups", "weighted_midrank")
    if not directed:
        names=("cluster_counts", "balanced_weights", "weighted_corr", "tie_groups", "weighted_midrank")
    return legacy_functions(18, names) + r'''


def rank_correlation(data, x, y, weights, adjustment=None):
    xr = weighted_midrank(tie_groups(data[x].to_numpy(float)),weights)
    yr = weighted_midrank(tie_groups(data[y].to_numpy(float)),weights)
    if adjustment is None:
        return weighted_corr(xr,yr,weights)
    zr = weighted_midrank(tie_groups(data[adjustment].to_numpy(float)),weights)
    active = weights>0
    if active.sum()<3 or np.ptp(zr[active])==0:
        return np.nan
    design = np.column_stack((np.ones(len(data)),zr))
    sw = np.sqrt(weights)
    if np.linalg.matrix_rank(design[active]*sw[active,None])<2:
        return np.nan
    bx = np.linalg.lstsq(design*sw[:,None],xr*sw,rcond=None)[0]
    by = np.linalg.lstsq(design*sw[:,None],yr*sw,rcond=None)[0]
    rx,ry=xr-design@bx,yr-design@by
    if (np.linalg.norm(rx*sw)<=1e-12*max(np.linalg.norm(xr*sw),1.)
            or np.linalg.norm(ry*sw)<=1e-12*max(np.linalg.norm(yr*sw),1.)):
        return np.nan
    return weighted_corr(rx,ry,weights)


def estimate_ci(data, statistic, *, bootstrap_counts=None):
    nsub = data.subject.nunique()
    if data.empty:
        return dict(estimate=np.nan,ci_low=np.nan,ci_high=np.nan,n_pairs=0,n_records=0,
                    n_subjects=0,n_boot_valid=0,ci_status='empty_support'), np.full(N_BOOT,np.nan)
    weights = balanced_weights(data)
    estimate = statistic(data,weights)
    indices = np.array([SUBJECT_INDEX[s] for s in data.subject],int)
    counts = BOOT_COUNTS if bootstrap_counts is None else bootstrap_counts
    draws = np.array([statistic(data,weights*count[indices]) for count in counts])
    finite = np.isfinite(draws)
    allowed = nsub>=MIN_CI_SUBJECTS and finite.sum()>=np.ceil(.95*len(counts)) and np.isfinite(estimate)
    low,high = (np.quantile(draws[finite],[(1-CONFIDENCE)/2,1-(1-CONFIDENCE)/2])
                if allowed else (np.nan,np.nan))
    reason = ('reported' if allowed else 'fewer_than_10_subjects' if nsub<MIN_CI_SUBJECTS
              else 'nonfinite_estimator_or_too_few_valid_draws')
    return dict(estimate=float(estimate),ci_low=float(low),ci_high=float(high),n_pairs=len(data),
                n_records=data.idx.nunique(),n_subjects=nsub,n_boot_valid=int(finite.sum()),
                ci_status=reason),draws


def weighted_mean_stat(column):
    def statistic(data,weights):
        if weights.sum()<=0:return np.nan
        return float(np.average(data[column].to_numpy(float),weights=weights))
    return statistic


def save_figure(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.png'),dpi=220,bbox_inches='tight')
    plt.show()
    plt.close(fig)
'''


def add_finish(cells):
    code(cells, r'''
check(source_digest(NOTEBOOK)==SOURCE_DIGEST,'new notebook scientific source unchanged')
inventory=[]
for path,before in sorted(tracked.items(),key=lambda item:str(item[0])):
    after=file_hash(path)
    inventory.append(dict(path=str(path.relative_to(ROOT)),sha256_before=before,
                          sha256_after=after,unchanged=before==after))
inventory=pd.DataFrame(inventory)
inventory.to_csv(OUT/'input_inventory.csv',index=False)
check(inventory.unchanged.all(),'all original input/source files unchanged')
pd.DataFrame(checks).to_csv(OUT/'alignment_checks.csv',index=False)
run_manifest.update(completion_status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
    scientific_source=str(NOTEBOOK.relative_to(ROOT)),notebook_source_sha256=SOURCE_DIGEST,
    source_pins_sha256=file_hash(PLAN/'source_pins.json'),protocol_sha256=file_hash(PLAN/'PROTOCOL.md'),
    versions={name:version(name) for name in ('numpy','pandas','scipy','matplotlib')},
    python=platform.python_version(),tables=sorted(p.name for p in OUT.glob('*.csv')),
    figures=sorted(p.name for p in FIG.glob('*')),old_preregistration_changed=False,
    alignment_checks_total=len(checks),alignment_checks_passed=sum(int(c['passed']) for c in checks),
    n_input_files_tracked=len(inventory),bank=dict(cohort=recipe['cohort'],architectures=list(ARMS),
        seeds=list(SEEDS),n_checkpoints=len(models),n_records=len(records),n_subjects=len(subjects)))
(OUT/'analysis_manifest.json').write_text(json.dumps(run_manifest,indent=2),encoding='utf-8')
print('COMPLETE:',OUT)
print('Additional exploratory results; historical estimates and manuscript untouched.')
''')


def build_notebook():
    cells = []
    markdown(cells, r'''
# Attention audit: map references and paired bridge sensitivities

Additional exploratory analyses, designed 2026-10-16 after the historical
E1/E2/RMS/bridge results were known. No training, inference or new intervention.
The old primary estimator and historical preregistration remain unchanged.

Read `plans/attention_audit_extension_20261016/PROTOCOL.md` first. Open this notebook,
select **Skripsi (.venv)**, then **Restart Kernel → Run All**. CPU only. Run All
creates a new run under `results/attention_audit_diagnostics/` and corresponding
`figures/attention_audit_diagnostics/`. No output exists until you run the cells.
Do not run notebooks 16/17/18 again to supply inputs; their pinned complete runs
are already available. Stop on failed alignment; do not change tolerances.

Builder: `.venv/bin/python scripts/build_nb_19.py` (generation only).
All scientific source is embedded below; the agent does not execute these cells.
''')
    add_preflight(cells, "19_attention_audit_diagnostics.ipynb", "build_nb_19.py",
                  "attention_audit_diagnostics", "ATTENTION_STORED_DIAGNOSTICS_V1")
    markdown(cells, r'''
## Conditional subject inference

Weights: equal subject → equal eligible recording → equal unordered seed pair.
Class-stratified subject draws preserve all record/pair dependence. Recompute
weighted midranks and adjustment fits in every draw. Intervals condition on
available checkpoints, split, seeds and matching. They are exploratory, nominal
95% intervals across multiple sensitivities. No result establishes independence,
equivalence or a causal adjustment effect.
''')
    code(cells, inference_source())
    code(cells, r'''
BOOT_COUNTS=cluster_counts(subjects,N_BOOT,BOOT_SEED)
np.savez_compressed(OUT/'subject_bootstrap_counts.npz',counts=BOOT_COUNTS,
                    subjects=SUBJECT_POOL.astype(str),labels=subjects.label.to_numpy())
run_manifest.update(completion_status='analysis_running',
    bootstrap=dict(n_boot=N_BOOT,seed=BOOT_SEED,unit='subject',strata='HC/PD',conditional=True),
    diagnostics=dict(reference_draws=199,reference_seed=1900,primary_reference='nonzero circular shift',
                     sensitivity_reference='permutation',partial_adjustment='weighted midrank abs_delta_logit',
                     alternative_adjustment='weighted midrank abs_delta_prob',floor=PRIMARY_FLOOR))
(OUT/'analysis_manifest.json').write_text(json.dumps(run_manifest,indent=2),encoding='utf-8')
''')
    markdown(cells, r'''
## Concentration and map-reference distributions

Circular shifts preserve weights and cyclic local order; permutations preserve
weights while disrupting order. These are reference constructions, not a claim
that a shifted map is a true explanation. Zero shift is excluded; periodic maps
can still produce an unchanged map and remain in the reference. No reference
p-value is interpreted as a population test. Entropy is in bits and effective
support is 2**entropy. Budget boundary gaps contextualize changes of top-k rank.
''')
    code(cells, r'''
REFERENCE_DRAWS,REFERENCE_SEED=199,1900

def normalized_alpha(values):
    values=np.asarray(values,float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or (values<0).any() or values.sum()<=0:
        raise AlignmentError('invalid attention mass')
    return values/values.sum()


def js_divergence(a,b):
    return float(np.clip(jensenshannon(normalized_alpha(a),normalized_alpha(b),base=2)**2,0,1))


def map_concentration(values):
    a=normalized_alpha(values);positive=a[a>0];w=len(a)
    entropy=float(-np.sum(positive*np.log2(positive)))
    out=dict(n_patches=w,entropy_bits=entropy,normalized_entropy=entropy/np.log2(w) if w>1 else np.nan,
             effective_support=2**entropy,effective_support_fraction=2**entropy/w,max_mass=float(a.max()))
    ordered=np.sort(a)[::-1]
    for fraction in FRACTIONS:
        k=max(1,int(np.ceil(w*fraction)));key=f'{fraction:g}'
        out['boundary_gap_'+key]=float(ordered[k-1]-ordered[k]) if k<w else np.nan
        out['boundary_ties_'+key]=int((a==ordered[k-1]).sum())
    return out

map_rows=[]
for (model_id,idx),entry in sorted(maps.items()):
    rec=records.iloc[idx];spec=specs[model_id]
    map_rows.append(dict(architecture=spec['architecture'],seed=spec['seed'],fold=spec['fold'],
        model_id=model_id,idx=idx,record_id=rec.record_id,subject=rec.subject,label=int(rec.label),
        **map_concentration(entry['alpha'])))
concentration=pd.DataFrame(map_rows)
concentration.to_csv(OUT/'map_concentration.csv',index=False)

reference_rows=[];reference_summaries=[]
for pair in e1.itertuples(index=False):
    a=normalized_alpha(maps[(pair.e1_model_id_a,int(pair.idx))]['alpha'])
    b=normalized_alpha(maps[(pair.e1_model_id_b,int(pair.idx))]['alpha'])
    actual=js_divergence(a,b)
    identity=dict(architecture=pair.architecture,idx=int(pair.idx),record_id=pair.record_id,
        subject=pair.subject,label=int(pair.label),fold=int(pair.fold_actual),seed_a=int(pair.seed_a),
        seed_b=int(pair.seed_b),model_id_a=pair.e1_model_id_a,model_id_b=pair.e1_model_id_b,
        n_patches=len(a),actual_jsd=actual)
    for reference in ('circular_shift','permutation'):
        rng=np.random.default_rng(keyed_seed(REFERENCE_SEED,pair.architecture,pair.record_id,
                                             pair.seed_a,pair.seed_b,reference))
        offsets=(np.sort(rng.choice(np.arange(1,len(b)),size=min(REFERENCE_DRAWS,len(b)-1),replace=False))
                 if reference=='circular_shift' else np.arange(REFERENCE_DRAWS))
        values=[]
        for draw,candidate in enumerate(offsets):
            null=np.roll(b,int(candidate)) if reference=='circular_shift' else b[rng.permutation(len(b))]
            value=js_divergence(a,null);values.append(value)
            reference_rows.append(dict(**identity,reference=reference,draw=draw,candidate_id=int(candidate),
                                       reference_jsd=value,unchanged_map=bool(np.array_equal(null,b))))
        values=np.asarray(values,float)
        reference_summaries.append(dict(**identity,reference=reference,n_reference_draws=len(values),
            reference_mean=float(values.mean()) if len(values) else np.nan,
            reference_q025=float(np.quantile(values,.025)) if len(values) else np.nan,
            reference_q975=float(np.quantile(values,.975)) if len(values) else np.nan,
            actual_minus_reference_mean=actual-float(values.mean()) if len(values) else np.nan))
reference_draws=pd.DataFrame(reference_rows)
reference_summary=pd.DataFrame(reference_summaries)
reference_draws.to_csv(OUT/'map_reference_draws.csv',index=False)
reference_summary.to_csv(OUT/'map_reference_pairs.csv',index=False)
ref_ci=[]
for (arm,reference),group in reference_summary.groupby(['architecture','reference'],sort=True):
    valid=group.loc[np.isfinite(group.actual_minus_reference_mean)].copy()
    row,_=estimate_ci(valid,weighted_mean_stat('actual_minus_reference_mean'))
    ref_ci.append(dict(architecture=arm,reference=reference,endpoint='actual_minus_reference_mean',**row))
reference_ci=pd.DataFrame(ref_ci)
reference_ci.to_csv(OUT/'map_reference_subject_ci.csv',index=False)
del reference_rows
display(reference_ci)
''')
    markdown(cells, r'''
## Weighted output-change sensitivity

This partial rank correlation residualizes weighted midranks on one adjustment
variable. It is an operational, linear adjustment in rank space. It is not a
general conditional-independence test. Report unadjusted and adjusted estimates
on identical finite support, with raw and normalized outcomes separately.
''')
    code(cells, r'''
sensitivity_rows=[]
for (arm,masker),group in pairs.groupby(['architecture','masker'],sort=True):
    for outcome,eligibility,scope in [('abs_delta_gain_abs_delta_logit','pair_eligible','raw_native_support'),
                                ('abs_delta_relative_gain','normalization_eligible','normalized_support'),
                                ('abs_delta_gain_abs_delta_logit','normalization_eligible','raw_on_normalized_support')]:
        for adjustment in ('abs_delta_logit','abs_delta_prob'):
            valid=group.loc[group[eligibility]&np.isfinite(group[outcome])
                &np.isfinite(group.js_divergence)&np.isfinite(group[adjustment])].copy()
            for estimator,control in [('weighted_spearman',None),('weighted_partial_rank',adjustment)]:
                stat=lambda d,w,c=control,y=outcome:rank_correlation(d,'js_divergence',y,w,c)
                result,_=estimate_ci(valid,stat)
                sensitivity_rows.append(dict(architecture=arm,masker=masker,outcome=outcome,
                    adjustment=adjustment,estimator=estimator,scope=scope,**result))
conditional=pd.DataFrame(sensitivity_rows)
conditional.to_csv(OUT/'output_change_sensitivity.csv',index=False)
display(conditional)
''')
    markdown(cells, r'''
## Direct median-minus-donor association contrasts

Intersect complete pairs across replacements before estimation. Normalized support
requires all four T+R denominators to exceed the floor. Raw-on-normalized-support
isolates changes of support from changes of outcome scale. Each bootstrap draw
uses the same subjects for both replacement correlations and then subtracts them.
''')
    code(cells, r'''
JOIN_PAIR=[c for c in PAIR_ID if c!='masker']
left=pairs.loc[pairs.masker=='train_median'].copy()
right=pairs.loc[pairs.masker=='train_donor'].copy()
common=left.merge(right,on=JOIN_PAIR,how='outer',validate='one_to_one',
                  suffixes=('_median','_donor'),indicator=True)
check((common['_merge']=='both').all(),'replacement planned pairs identical')
finite_close(common.js_divergence_median,common.js_divergence_donor,'replacement shared JSD')
finite_close(common.abs_delta_logit_median,common.abs_delta_logit_donor,'replacement shared output change')
common['js_divergence']=common.js_divergence_median
common['abs_delta_logit']=common.abs_delta_logit_median
common['both_complete']=common.pair_eligible_median&common.pair_eligible_donor
common['both_normalized']=common.both_complete&common.normalization_eligible_median&common.normalization_eligible_donor
common.to_csv(OUT/'replacement_common_pair_support.csv',index=False)
replacement_rows=[];replacement_coverage=[]
for arm,full in common.groupby('architecture',sort=True):
    for scope,outcome in [('raw_common','abs_delta_gain_abs_delta_logit'),
                          ('normalized_common','abs_delta_relative_gain'),
                          ('raw_on_normalized_common','abs_delta_gain_abs_delta_logit')]:
        gate=full.both_complete if scope=='raw_common' else full.both_normalized
        valid=full.loc[gate&np.isfinite(full[outcome+'_median'])&np.isfinite(full[outcome+'_donor'])].copy()
        replacement_coverage.append(dict(architecture=arm,scope=scope,n_pairs_total=len(full),
            n_pairs_common=len(valid),n_records_total=full.idx.nunique(),n_records_common=valid.idx.nunique(),
            n_subjects_total=len(subjects),n_subjects_common=valid.subject.nunique()))
        for adjustment in (None,'abs_delta_logit'):
            def contrast_stat(d,w):
                return (rank_correlation(d,'js_divergence',outcome+'_median',w,adjustment)
                        -rank_correlation(d,'js_divergence',outcome+'_donor',w,adjustment))
            result,_=estimate_ci(valid,contrast_stat)
            weights=balanced_weights(valid) if len(valid) else np.array([])
            rm=rank_correlation(valid,'js_divergence',outcome+'_median',weights,adjustment) if len(valid) else np.nan
            rd=rank_correlation(valid,'js_divergence',outcome+'_donor',weights,adjustment) if len(valid) else np.nan
            replacement_rows.append(dict(architecture=arm,scope=scope,outcome=outcome,
                adjustment='none' if adjustment is None else adjustment,rho_median=rm,rho_donor=rd,
                contrast='rho_median_minus_rho_donor',**result))
replacement_ci=pd.DataFrame(replacement_rows)
replacement_ci.to_csv(OUT/'replacement_contrast_ci.csv',index=False)
pd.DataFrame(replacement_coverage).to_csv(OUT/'replacement_coverage.csv',index=False)
display(replacement_ci)
''')
    markdown(cells, r'''
## Figures and outputs

Figures use the full JSD range and show reference means as such. They do not
silently expand the narrow observed range. Check completion/alignment first,
then common-support coverage, followed by effect sizes and estimator sensitivities.
The reference and partial-correlation results are new exploratory analyses.
''')
    code(cells, r'''
ARM_LABEL={'gru':'BiGRU','mamba2':'BiMamba-2'}
MASK_LABEL={'train_median':'Training median','train_donor':'Training donor'}
fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
for i,arm in enumerate(ARMS):
    for j,reference in enumerate(('circular_shift','permutation')):
        g=reference_summary[(reference_summary.architecture==arm)&(reference_summary.reference==reference)]
        axes[i,j].scatter(g.actual_jsd,g.reference_mean,s=8,alpha=.25,rasterized=True)
        axes[i,j].plot([0,1],[0,1],color='.4',ls='--',lw=1)
        axes[i,j].set(xlim=(0,1),ylim=(0,1),xlabel='Observed JSD (base-2 divergence)',
            ylabel='Mean reference JSD',title=ARM_LABEL[arm]+' / '+reference.replace('_',' '))
save_figure(fig,'observed_vs_reference_jsd')
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for arm,color in zip(ARMS,('#26547c','#b04c24')):
    g=concentration[concentration.architecture==arm]
    axes[0].hist(g.normalized_entropy,bins=np.linspace(0,1,26),histtype='step',label=ARM_LABEL[arm],color=color)
    axes[1].hist(g.effective_support_fraction,bins=np.linspace(0,1,26),histtype='step',label=ARM_LABEL[arm],color=color)
axes[0].set(xlabel='Entropy / log2(number of patches)',ylabel='Saved maps',xlim=(0,1))
axes[1].set(xlabel='Effective support / number of patches',ylabel='Saved maps',xlim=(0,1))
axes[1].legend(frameon=False)
save_figure(fig,'attention_concentration')
fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
g=conditional[(conditional.scope=='raw_native_support')&(conditional.adjustment=='abs_delta_logit')]
labels=[]
for i,row in enumerate(g.itertuples(index=False)):
    if np.isfinite(row.estimate):
        ax.plot(row.estimate,i,'o',color='#26547c' if row.estimator=='weighted_spearman' else '#9d3548')
        if np.isfinite(row.ci_low):ax.plot([row.ci_low,row.ci_high],[i,i],color='.35')
    labels.append(ARM_LABEL[row.architecture]+' / '+MASK_LABEL[row.masker]+' / '+row.estimator.replace('weighted_',''))
ax.set_yticks(range(len(labels)),labels);ax.axvline(0,color='.5',ls=':')
ax.set(xlabel='Rank association with JSD (nominal subject CI)',xlim=(-1,1))
save_figure(fig,'output_change_sensitivity')
fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
g=replacement_ci[replacement_ci.adjustment=='none']
labels=[]
for i,row in enumerate(g.itertuples(index=False)):
    if np.isfinite(row.estimate):
        ax.plot(row.estimate,i,'o',color='#26547c')
        if np.isfinite(row.ci_low):ax.plot([row.ci_low,row.ci_high],[i,i],color='.35')
    labels.append(ARM_LABEL[row.architecture]+' / '+row.scope.replace('_',' '))
ax.set_yticks(range(len(labels)),labels);ax.axvline(0,color='.5',ls=':')
ax.set(xlabel='Median-minus-donor rank association (paired subject CI)',xlim=(-2,2))
save_figure(fig,'replacement_contrast')
''')
    add_finish(cells)
    return nbf.v4.new_notebook(cells=cells)


def write_notebook(notebook, name):
    notebook.metadata = {
        "kernelspec": {"display_name": "Skripsi (.venv)", "language": "python", "name": "skripsi"},
        "language_info": {"name": "python", "version": "3.11.8"},
        "analysis_status": "additional exploratory; not executed by builder",
    }
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"{name}:cell{index}", "exec")
    nbf.validate(notebook)
    path = ROOT / "notebooks" / name
    if path.exists():
        from datetime import datetime, timezone
        import shutil
        archive = path.parent / "archive"
        archive.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = archive / f"{path.stem}_{stamp}.ipynb"
        shutil.copy2(path, backup)
        print("Previous notebook preserved:", backup)
    nbf.write(notebook, path)
    print(f"Generated {path} ({len(notebook.cells)} cells); no cells executed.")


if __name__ == "__main__":
    write_notebook(build_notebook(), "19_attention_audit_diagnostics.ipynb")
