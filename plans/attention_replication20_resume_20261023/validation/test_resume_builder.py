"""Synthetic software regression tests; no study model, CUDA or bootstrap."""
from __future__ import annotations

import ast
import builtins
import importlib.util
import json
import symtable
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('resume_builder',ROOT/'scripts/build_nb_22_24_resume.py')
builder=importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class FixtureAlignmentError(ValueError):pass


def definitions(source,names,ns):
    found={n.name:n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    if set(names)-set(found):raise AssertionError('Missing definitions: '+str(set(names)-set(found)))
    tree=ast.fix_missing_locations(ast.Module(body=[found[n] for n in names],type_ignores=[]))
    exec(compile(tree,'synthetic_fixture','exec'),ns)


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def basic_namespace():
    return dict(np=np,json=json,Path=Path,sha256=sha256,spec_digest=digest,
        AlignmentError=FixtureAlignmentError,FRACTIONS=(.05,.1,.2,.3),PATCH=7,
        ranking_order=lambda a,record_id:np.argsort(-np.asarray(a),kind='stable'),
        SESSION='synthetic',EXECUTION_ID='synthetic_execution',ALIGNMENT_REPEATS=2,
        MAX_ALIGNMENT_PROFILES=128)


class ResumeContracts(unittest.TestCase):
    def test_schema_static_compilation_and_global_dependencies(self):
        generated=[builder.build_notebook(c) for c in ('newhandpd','uci395')]+[builder.build_combined()]
        for nb,name in generated:
            builder.fast.validate_notebook(nb,name)
            source='\n'.join(c.source for c in nb.cells if c.cell_type=='code')
            table=symtable.symtable(source,name,'exec')
            bound={s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
            allowed=bound|set(dir(builtins));missing=set()
            def visit(scope):
                for symbol in scope.get_symbols():
                    if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in allowed:
                        missing.add(symbol.get_name())
                for child in scope.get_children():visit(child)
            visit(table)
            self.assertEqual(missing,set(),name)
            self.assertTrue(all(c.execution_count is None and not c.outputs for c in nb.cells if c.cell_type=='code'))
            self.assertNotIn('notebooks/22_replication20_newhandpd_fast.ipynb',source)

    def test_science_and_training_sources_are_preserved(self):
        for cohort in ('newhandpd','uci395'):
            original,_=builder.fast.build_notebook(cohort)
            resumed,_=builder.build_notebook(cohort)
            sources=[c.source for c in resumed.cells if c.cell_type=='code']
            self.assertIn(original.cells[6].source,sources)  # unchanged training / OOF stage
            self.assertIn(original.cells[8].source,sources)  # matching, RMS and geometry
            self.assertIn(original.cells[9].source,sources)  # ranking / AOPC summaries
            self.assertIn(original.cells[16].source,sources) # weighted correlations
            self.assertIn(original.cells[17].source,sources) # inferential functions
            self.assertIn(original.cells[18].source,sources) # estimators / strata
            all_source='\n'.join(sources)
            self.assertIn('len(pair_curves)==expected',all_source)
            self.assertIn('early_alignment_probe()',all_source)
            self.assertIn('release_record_runtime(predictor)',all_source)

    def test_one_boundary_patch_swap_is_not_accepted(self):
        ns=basic_namespace();definitions(builder.ALIGNMENT_METHODS,('alignment_report',),ns)
        alpha=np.arange(12,0,-1,dtype=float);alpha/=alpha.sum()
        fresh=alpha.copy();fresh[[3,4]]=fresh[[4,3]]
        identity=dict(record_id='fixture_record')
        report=ns['alignment_report'](1.+1e-5,fresh,dict(alpha=alpha,logit=1.),identity)
        self.assertTrue(report['anchor_passed'])
        self.assertFalse(report['passed'])
        self.assertEqual(len(report['mask_mismatches']),1)
        self.assertEqual(report['mask_mismatches'][0]['fraction'],.3)
        self.assertEqual(report['mask_mismatches'][0]['removed'],[3])
        self.assertEqual(report['mask_mismatches'][0]['added'],[4])
        np.testing.assert_array_equal(alpha,np.arange(12,0,-1)/78.)
        self.assertFalse(ns['alignment_report'](1.1,alpha,dict(alpha=alpha,logit=1.),identity)['passed'])
        with self.assertRaises(FixtureAlignmentError):
            ns['alignment_report'](1.,alpha[:-1],dict(alpha=alpha,logit=1.),identity)

    def test_alignment_diagnostics_are_valid_json_lines(self):
        ns=basic_namespace();definitions(builder.ALIGNMENT_METHODS,('alignment_event',),ns)
        with tempfile.TemporaryDirectory() as tmp:
            ns['OUT']=Path(tmp)
            ns['alignment_event'](dict(model_id='fixture'),phase='first')
            ns['alignment_event'](dict(model_id='fixture'),phase='second')
            rows=[json.loads(line) for line in (Path(tmp)/'runtime_alignment_trials.jsonl').read_text().splitlines()]
            self.assertEqual([r['phase'] for r in rows],['first','second'])

    def test_partial_replay_checks_all_targets_and_controls(self):
        ns=basic_namespace()
        def apply(x,bg,selected,patch):
            result=x.copy()
            for i in selected:result[i*patch:(i+1)*patch]=bg[i*patch:(i+1)*patch]
            return result
        ns['apply_mask']=apply;ns['forward_one']=lambda predictor,x:float(x.sum())
        def anchor(a,b,where):
            if not np.isclose(a,b,atol=1e-4,rtol=1e-4):raise FixtureAlignmentError(where)
        ns['assert_logit']=anchor
        definitions(builder.ALIGNMENT_METHODS,('replay_partial_components',),ns)
        x=np.arange(15,dtype=float)[:,None];bg=np.zeros_like(x)
        component=dict(background_sha256='fixture_bg',original_logit=float(x.sum()),selected=np.array([0]),
            target_logit=float(apply(x,bg,[0],7).sum()),controls=[dict(selected=np.array([1]),
                perturbed_logit=float(apply(x,bg,[1],7).sum()))])
        args=(None,x,dict(logit=float(x.sum())),[dict(background_sha256='fixture_bg',background=bg)],{0:[component]})
        report=ns['replay_partial_components'](*args)
        self.assertTrue(report['passed']);self.assertEqual(report['n_points'],2)
        component['controls'][0]['perturbed_logit']+=1.
        self.assertFalse(ns['replay_partial_components'](*args)['passed'])

    def test_recovery_is_gated_persisted_and_restored(self):
        ns=basic_namespace();events=[]
        bad={'kernel':{'mode':'bad'}};good={'kernel':{'mode':'good'}}
        obj=SimpleNamespace(configs=[bad['kernel'],good['kernel']],cache={'original':'cache'},best_config=bad['kernel'])
        state={'profile':bad}
        ns.update(runtime_kernels=lambda:({'kernel':obj},{'kernel':[bad['kernel'],good['kernel']]},{'fixture':'sha'}),
            config_payload=lambda config:config,
            alignment_candidates=lambda registered,native,known:[bad,good],known_alignment_profiles=lambda:[],
            profile_binding=lambda *args:{'fixture_binding':'same'},
            replay_partial_components=lambda *args:dict(passed=True,n_points=0,max_abs_difference=0.),
            alignment_event=lambda identity,**row:events.append(row),
            atomic_json=lambda path,value:(path.parent.mkdir(parents=True,exist_ok=True),path.write_text(json.dumps(value))))
        def pin(objects,registered,profile):
            state['profile']=profile;objects['kernel'].configs=[profile['kernel']];objects['kernel'].cache.clear()
        ns['pin_runtime']=pin
        definitions(builder.ALIGNMENT_METHODS,('alignment_report','runtime_snapshot','restore_runtime',
            'check_alignment_profile','verified_original','release_record_runtime'),ns)
        alpha=np.arange(12,0,-1,dtype=float)/78.;wrong=alpha.copy();wrong[[3,4]]=wrong[[4,3]]
        class FixturePredictor:
            def original(self,x):return 1.,alpha.copy() if state['profile']==good else wrong.copy()
        identity=dict(architecture='mamba2',model_id='fixture_model',idx=0,record_id='fixture_record',seed=7,fold=4)
        with tempfile.TemporaryDirectory() as tmp:
            ns['OUT']=Path(tmp);predictor=FixturePredictor()
            result=ns['verified_original'](predictor,np.zeros((84,1)),dict(alpha=alpha,logit=1.),identity,[],{})
            np.testing.assert_array_equal(result[1],alpha)
            path=Path(tmp)/'runtime_alignment_profiles/fixture_model/0.json'
            saved=json.loads(path.read_text());before=path.read_bytes()
            self.assertEqual(saved['spec']['kernels'],good)
            self.assertEqual(saved['profile_id'],predictor.alignment_profile_id)
            accepted=[e for e in events if e.get('phase')=='recovery' and e.get('passed')]
            self.assertEqual(len(accepted),2)
            ns['release_record_runtime'](predictor)
            self.assertEqual(obj.configs,[bad['kernel'],good['kernel']]);self.assertEqual(obj.cache,{'original':'cache'})
            # A bound accepted profile must fail closed if the saved binding differs.
            ns['profile_binding']=lambda *args:{'fixture_binding':'changed'}
            with self.assertRaisesRegex(FixtureAlignmentError,'different binding'):
                ns['verified_original'](FixturePredictor(),np.zeros((84,1)),dict(alpha=alpha,logit=1.),identity,[],{})
            self.assertEqual(path.read_bytes(),before)

    def test_native_success_does_not_search_or_pin(self):
        ns=basic_namespace();alpha=np.array([.4,.3,.2,.1]);events=[]
        ns.update(alignment_event=lambda *args,**kwargs:events.append(kwargs),
            replay_partial_components=lambda *args:dict(passed=True,n_points=0),
            runtime_kernels=lambda:(_ for _ in ()).throw(AssertionError('native pass must not search')))
        definitions(builder.ALIGNMENT_METHODS,('alignment_report','verified_original'),ns)
        with tempfile.TemporaryDirectory() as tmp:
            ns['OUT']=Path(tmp)
            predictor=SimpleNamespace(original=lambda x:(1.,alpha.copy()))
            identity=dict(architecture='gru',model_id='fixture_model',idx=0,record_id='fixture')
            result=ns['verified_original'](predictor,None,dict(alpha=alpha,logit=1.),identity,[],{})
            np.testing.assert_array_equal(result[1],alpha)
            self.assertEqual(predictor.alignment_profile_id,'native_autotuned')

    def test_bounded_candidates_are_registered_deterministic_and_unique(self):
        ns=basic_namespace();ns['config_payload']=lambda c:c.payload
        definitions(builder.ALIGNMENT_METHODS,('alignment_candidates',),ns)
        def config(**kwargs):
            return SimpleNamespace(kwargs=kwargs,num_stages=3,num_warps=4,
                payload=dict(kwargs=kwargs,num_stages=3,num_warps=4))
        matrix=[config(BLOCK_SIZE_M=m,BLOCK_SIZE_N=n,BLOCK_SIZE_K=k)
                for m,n,k in ((32,64,32),(64,32,32),(64,64,64))]
        registered=dict(bmm=matrix,chunk_state=matrix,chunk_scan=matrix,
            cumsum=[config(BLOCK_SIZE_H=i) for i in range(1,11)],
            state_passing=[config(BLOCK_SIZE=i) for i in (16,32,64,128,256,512)])
        native={name:items[0].payload for name,items in registered.items()}
        profiles=ns['alignment_candidates'](registered,native,[])
        self.assertEqual(len(profiles),128)
        self.assertEqual(profiles,ns['alignment_candidates'](registered,native,[]))
        self.assertEqual(profiles[0],native)
        self.assertEqual(len({digest(p) for p in profiles}),len(profiles))
        for profile in profiles:
            self.assertTrue(all(profile[name] in [c.payload for c in items] for name,items in registered.items()))

    def test_record_wrapper_restores_runtime_on_exception(self):
        nb,_=builder.build_notebook('newhandpd')
        source=next(c.source for c in nb.cells if c.cell_type=='code' and '_strict_evaluate_checkpoint_record=' in c.source)
        ns=basic_namespace();released=[]
        def fail(*args):raise FixtureAlignmentError('strict fixture failure')
        ns.update(_strict_evaluate_checkpoint_record=fail,release_record_runtime=lambda p:released.append(p))
        definitions(source,('evaluate_checkpoint_record',),ns)
        predictor=object()
        with self.assertRaisesRegex(FixtureAlignmentError,'strict fixture failure'):
            ns['evaluate_checkpoint_record'](predictor,None,None,None,None,None)
        self.assertEqual(released,[predictor])

    def test_cache_readthrough_is_read_only_and_validates_context(self):
        ns=basic_namespace();ns.update(register_reuse=lambda *args:None)
        ns['validate_cached_components']=lambda value,context,partial:value if value['context']==context else (_ for _ in ()).throw(FixtureAlignmentError('context'))
        ns['read_local_pickle']=lambda path:json.loads(path.read_text())
        definitions(builder.CACHE_METHODS,('prediction_path','compatible_record_cache'),ns)
        with tempfile.TemporaryDirectory() as tmp:
            ns['OUT']=Path(tmp)/'resume';parent=Path(tmp)/'fast';legacy=Path(tmp)/'reference'
            ns['CACHE_PARENT_DIRS']=(parent,legacy)
            relative=Path('record_evaluations/model/0.pkl');p=parent/relative
            p.parent.mkdir(parents=True);p.write_text(json.dumps(dict(context={'fixture':1},components={})))
            before=p.read_bytes()
            value=ns['compatible_record_cache'](ns['OUT']/relative,{'fixture':1})
            self.assertEqual(value['context'],{'fixture':1})
            self.assertEqual(p.read_bytes(),before);self.assertFalse(ns['OUT'].exists())
            with self.assertRaises(FixtureAlignmentError):ns['compatible_record_cache'](ns['OUT']/relative,{'fixture':2})


if __name__=='__main__':unittest.main(verbosity=2)
