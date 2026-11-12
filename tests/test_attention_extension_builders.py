"""Synthetic software checks only: no research files, models or bootstrap draws.

Builders may construct notebook objects. Scientific cells are never executed.
Only selected pure function definitions are loaded for tiny synthetic fixtures.
"""
from pathlib import Path
import ast
import hashlib
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import build_nb_19 as b19
import build_nb_20 as b20


def definitions(source, names=None):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
             and (names is None or n.name in names)]
    return ast.Module(body=nodes, type_ignores=[])


def pure_environment():
    env = dict(np=np, pd=pd, json=json, sha256=hashlib.sha256, Path=Path,
               jensenshannon=jensenshannon, expit=expit, AlignmentError=ValueError,
               FRACTIONS=(.25, .5), MASKERS=('train_median',), PATCH=7,
               RANDOM_REPEATS=1, DONOR_REPEATS=1, RMS_REL_TOL=.05, RMS_ABS_TOL=1e-8,
               PRIMARY_FLOOR=1e-8, MC_DRAWS=8, MC_SEED=2003,
               METRICS=('abs_delta_logit','signed_margin_drop','predicted_class_prob_drop'),
               PLACEBO_CANDIDATES=5, PLACEBO_SEED=2001, PLACEBO_JSD_CALIPER=1.,
               recipe={'intervention_seed':1600})
    legacy=b19.legacy_functions(18, ('keyed_seed','mask_hash','file_hash','digest'))
    legacy+='\n'+b19.legacy_functions(17, ('mask_runs','patch_energy'))
    exec(compile(definitions(legacy),'<pure legacy definitions>','exec'),env)
    exec(compile(definitions(b20.swap_methods_source()),'<pure new definitions>','exec'),env)
    return env


class BuilderStaticChecks(unittest.TestCase):
    def test_both_notebooks_have_valid_unexecuted_cells_and_no_training_path(self):
        for build in (b19.build_notebook,b20.build_notebook):
            notebook=build()
            self.assertGreater(len(notebook.cells),15)
            text='\n'.join(c.source for c in notebook.cells if c.cell_type=='code')
            for cell in notebook.cells:
                if cell.cell_type=='code':
                    compile(cell.source,'<unexecuted cell>','exec')
                    self.assertIsNone(cell.execution_count)
                    self.assertEqual(cell.outputs,[])
            tree=ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                    self.assertNotIn(node.func.id,('latih','train_model','fit_calibrator'))
                    if node.func.id=='get_fold_model':
                        flag=next(k.value for k in node.keywords if k.arg=='allow_training')
                        self.assertIs(ast.literal_eval(flag),False)
            self.assertNotIn('# AGGREGATION_SOURCE',text)
            self.assertNotIn('# PLOTTING_SOURCE',text)

    def test_streamed_tables_preserve_header_order(self):
        env=pure_environment()
        with tempfile.TemporaryDirectory() as folder:
            env['OUT']=Path(folder)
            env['stream_table']('toy',[{'a':1,'b':2}])
            env['stream_table']('toy',[{'b':4,'a':3}])
            result=pd.read_csv(Path(folder)/'toy.csv')
            self.assertEqual(result.to_dict('records'),[{'a':1,'b':2},{'a':3,'b':4}])
            with self.assertRaisesRegex(ValueError,'schema changed'):
                env['stream_table']('toy',[{'a':5,'c':6}])

    def test_atomic_cache_uses_path_first_and_refuses_overwrite(self):
        env=pure_environment();env['ANALYSIS_ID']='toy-analysis'
        payload={'identity':{'idx':0}}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'record.pkl'
            env['cache_record'](path,payload)
            self.assertTrue(path.is_file())
            meta=json.loads(path.with_suffix('.json').read_text())
            self.assertEqual(meta['payload_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):env['cache_record'](path,payload)


class PureMethodChecks(unittest.TestCase):
    def test_small_jsd_does_not_imply_identical_top_rank(self):
        env=pure_environment();eps=1e-6
        a=np.array([.5+eps,.5-eps]);b=a[::-1].copy()
        self.assertLess(env['divergence'](a,b),1e-8)
        self.assertNotEqual(int(env['ranking_order'](a,'toy')[0]),int(env['ranking_order'](b,'toy')[0]))

    def test_placebo_preserves_weights_and_reports_no_offset(self):
        env=pure_environment()
        identity=dict(model_id='m',record_id='r',idx=0)
        own=np.array([.6,.25,.1,.05]);source=own[::-1]
        for kind in ('circular_shift','permutation'):
            values,support,rows=env['choose_placebo'](own,source,identity,1,kind)
            self.assertTrue(support['matching_complete'])
            np.testing.assert_array_equal(np.sort(values),np.sort(own))
            self.assertLessEqual(support['absolute_jsd_error'],1.)
            self.assertEqual(sum(r['selected'] for r in rows),1)
        values,support,rows=env['choose_placebo'](np.ones(1),np.ones(1),identity,1,'circular_shift')
        self.assertIsNone(values)
        self.assertFalse(support['matching_complete'])
        self.assertEqual(rows,[])

    def test_identical_masks_reuse_components_without_forward(self):
        env=pure_environment()
        order=np.arange(4);cache={}
        for fraction in env['FRACTIONS']:
            selected=np.sort(order[:max(1,int(np.ceil(len(order)*fraction)))])
            cache[(fraction,env['mask_hash'](selected))]=[{'tag':fraction}]
        def forbidden(*args):raise AssertionError('No forward should occur on a component-cache hit')
        result=env['new_rank_components'](forbidden,np.ones((28,1)),{'logit':0.},order,[],{},cache)
        self.assertEqual(result,[{'tag':.25},{'tag':.5}])

    def test_complete_components_and_aopc_use_all_points(self):
        env=pure_environment()
        x=np.ones((28,1),np.float32)
        library={'masker':'train_median','donor_repeat':0,'background':np.zeros_like(x),'background_sha256':'bg'}
        identity=dict(model_id='m',idx=0,record_id='r',subject='s',label=0,architecture='gru',seed=0,fold=0)
        components=[]
        for fraction,selected,control in [(.25,np.array([0]),np.array([2])),(.5,np.array([0,1]),np.array([2,3]))]:
            components.append(env['make_component'](identity,x,selected,fraction,library,0.,-2.,
                [{'repeat':0,'selected':control,'perturbed_logit':-1.}],{'complete':True},'toy'))
        scores,points=env['summarize_rank'](components,dict(rank_id='own',kind='own'),identity)
        self.assertEqual(len(points),2)
        self.assertAlmostEqual(scores[0]['T_abs_delta_logit'],1.5)
        self.assertAlmostEqual(scores[0]['R_abs_delta_logit'],.75)
        self.assertAlmostEqual(scores[0]['G_abs_delta_logit'],.75)
        self.assertAlmostEqual(scores[0]['H'],1/3)
        a=env['component_reweighting'](components[0])
        b=env['component_reweighting'](components[0])
        np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(a,np.ones(env['MC_DRAWS']))
        with self.assertRaisesRegex(ValueError,'required quota'):
            env['make_component'](identity,x,np.array([0]),.25,library,0.,-2.,[],{'complete':True},'toy')

    def test_directional_weighting_balances_targets_before_sources(self):
        env=pure_environment()
        module=definitions(b20.aggregation_source(),{'balanced_weights'})
        exec(compile(module,'<weighting definition>','exec'),env)
        frame=pd.DataFrame([
            dict(subject='s1',idx=0,target_seed=0,source_seed=1),
            dict(subject='s1',idx=0,target_seed=0,source_seed=2),
            dict(subject='s1',idx=0,target_seed=1,source_seed=0),
            dict(subject='s1',idx=1,target_seed=0,source_seed=1),
            dict(subject='s2',idx=2,target_seed=2,source_seed=0),
            dict(subject='s2',idx=2,target_seed=2,source_seed=1),
        ])
        frame['weight']=env['balanced_weights'](frame)
        np.testing.assert_allclose(frame.groupby('subject').weight.sum(),[1.,1.])
        self.assertAlmostEqual(frame.iloc[0].weight,.125)
        self.assertAlmostEqual(frame.iloc[2].weight,.25)
        self.assertAlmostEqual(frame.iloc[3].weight,.5)

    def test_partial_rank_reports_zero_residual_variance_as_undefined(self):
        env=pure_environment()
        exec(compile(definitions(b19.inference_source()),'<rank definitions>','exec'),env)
        data=pd.DataFrame({'x':[1.,2.,3.,4.],'y':[1.,4.,2.,3.],'z':[1.,2.,3.,4.]})
        value=env['rank_correlation'](data,'x','y',np.ones(4),'z')
        self.assertTrue(np.isnan(value))
        self.assertTrue(np.isfinite(env['rank_correlation'](data,'x','y',np.ones(4))))


if __name__=='__main__':
    unittest.main(verbosity=2)
