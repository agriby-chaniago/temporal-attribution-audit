"""Pemeriksaan sintetis E2 untuk pengguna; tanpa checkpoint/data penelitian/training."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from audit_e0_e1 import AlignmentError
from audit_e2 import apply_mask, donor_background, evaluate_record, e2_record_scores, join_e1_e2


class E2Tests(unittest.TestCase):
    def test_mask_preserves_length_tail_and_original(self):
        x = np.arange(22, dtype=np.float32).reshape(11,2)
        untouched = x.copy()
        altered = apply_mask(x, np.zeros_like(x), [1], patch=3)
        np.testing.assert_array_equal(x, untouched)
        np.testing.assert_array_equal(altered[:3], x[:3])
        np.testing.assert_array_equal(altered[3:6], 0)
        np.testing.assert_array_equal(altered[6:], x[6:])
        with self.assertRaises(AlignmentError): apply_mask(x,np.zeros_like(x),[3],patch=3)

    def test_donor_plan_tracks_contiguous_training_blocks(self):
        x = np.full((11,2), -1, np.float32)
        donor = np.arange(40,dtype=np.float32).reshape(20,2)
        background, plans = donor_background(x,[(7,"train_subject",donor)],3,np.random.default_rng(12),block_patches=2)
        for plan in plans:
            self.assertEqual(plan["donor_subject"],"train_subject")
            a,b,s=plan["target_start"],plan["target_stop"],plan["donor_start"]
            np.testing.assert_array_equal(background[a:b],donor[s:s+b-a])
        np.testing.assert_array_equal(background[9:],x[9:])

    def test_input_can_influence_output_while_opposing_original_class(self):
        x=np.ones((4,1),np.float32)
        coefficient=np.array([-2.,3.,0.,0.])
        predict=lambda batch: (batch[:,:,0]*coefficient).sum(axis=1)
        curves,_=evaluate_record(predict,x,np.array([.7,.2,.08,.02]),1.,
            [(0,"training",np.zeros_like(x))],record_id="toy",patch=1,
            fractions=(.25,),random_repeats=4,donor_repeats=2,block_patches=2,batch_size=1)
        selected=curves[(curves.masker=="train_median")&(curves.ranking=="alpha_top")]
        self.assertEqual(selected.abs_delta_logit.iloc[0],2.)
        self.assertEqual(selected.signed_margin_drop.iloc[0],-2.)
        self.assertTrue((curves.n_masked==1).all())

    def test_anchor_detects_different_original_output(self):
        x=np.ones((4,1),np.float32)
        predict=lambda batch: np.full(len(batch),4.)
        with self.assertRaises(AlignmentError):
            evaluate_record(predict,x,np.full(4,.25),0.,[(0,"train",x)],record_id="toy",
                patch=1,fractions=(.25,),random_repeats=2,donor_repeats=1,block_patches=2)

    def test_batch_sensitive_predictor_uses_singletons_for_every_forward(self):
        x=np.ones((4,1),np.float32)
        calls=[]
        def predict(batch):
            calls.append(batch.shape)
            return batch.sum(axis=(1,2)) + .02*(len(batch)-1)
        curves,_=evaluate_record(predict,x,np.array([.4,.3,.2,.1]),4.,
            [(0,"train",np.zeros_like(x))],record_id="batch_sensitive",patch=1,
            fractions=(.25,),random_repeats=2,donor_repeats=1,block_patches=2)
        self.assertEqual(set(calls),{(1,4,1)})
        self.assertEqual(len(calls),len(curves)+2)
        self.assertTrue((curves.abs_delta_logit==1.).all())
        self.assertTrue((curves.unmasked_abs_delta_max==0.).all())

    def test_larger_batch_rejected_before_any_forward(self):
        x=np.ones((4,1),np.float32)
        def predict(batch):
            self.fail("predictor tidak boleh dipanggil dengan konfigurasi batch tidak valid")
        with self.assertRaisesRegex(ValueError,"batch_size=1"):
            evaluate_record(predict,x,np.full(4,.25),4.,[(0,"train",x)],record_id="toy",
                patch=1,fractions=(.25,),random_repeats=2,donor_repeats=1,
                block_patches=2,batch_size=4)

    def test_anchor_after_detects_model_state_drift(self):
        x=np.ones((4,1),np.float32)
        calls=0
        def predict(batch):
            nonlocal calls
            calls+=1
            return batch.sum(axis=(1,2)) + (0. if calls==1 else .01)
        with self.assertRaisesRegex(AlignmentError,"record_id=state_drift, phase=after"):
            evaluate_record(predict,x,np.full(4,.25),4.,[(0,"train",x*0)],
                record_id="state_drift",patch=1,fractions=(.25,),
                random_repeats=2,donor_repeats=1,block_patches=2)

    def test_random_mask_ids_reproducible_independent_of_model_output(self):
        x=np.ones((4,1),np.float32)
        predict=lambda batch: batch.sum(axis=(1,2))
        kwargs=dict(record_id="record_same",patch=1,fractions=(.25,.50),random_repeats=4,
                    donor_repeats=2,block_patches=2)
        first,_=evaluate_record(predict,x,np.array([.4,.3,.2,.1]),4.,[(0,"train",x*0)],**kwargs)
        second,_=evaluate_record(predict,x,np.array([.1,.2,.3,.4]),4.,[(0,"train",x*0)],**kwargs)
        np.testing.assert_array_equal(first[first.ranking=="random"].mask_sha256,
                                      second[second.ranking=="random"].mask_sha256)

    def test_trapezoidal_aopc_and_paired_contrast(self):
        rows=[]
        for ranking,multiplier in [("alpha_top",2.),("alpha_bottom",0.),("random",1.)]:
            for fraction in (.1,.3):
                rows.append(dict(architecture="gru",seed=0,fold=0,model_id="m0",idx=0,
                    subject="p",label=1,masker="train_median",ranking=ranking,
                    requested_fraction=fraction,actual_fraction=fraction,original_logit=10.,
                    abs_delta_logit=multiplier*fraction,signed_margin_drop=multiplier*fraction,
                    predicted_class_prob_drop=multiplier*fraction))
        _,score=e2_record_scores(pd.DataFrame(rows))
        self.assertAlmostEqual(score.alpha_top_abs_delta_logit.iloc[0],.3)
        self.assertAlmostEqual(score.random_abs_delta_logit.iloc[0],.15)
        self.assertAlmostEqual(score.adv_abs_delta_logit.iloc[0],.15)
        self.assertAlmostEqual(score.relative_gain_abs_delta_logit.iloc[0],1/3)

    def test_join_requires_matching_checkpoint_and_original_logit(self):
        pairs=pd.DataFrame([dict(architecture="gru",seed_a=0,seed_b=1,idx=0,
            subject="s",label=1,logit_a=2.,logit_b=2.01,e1_model_id_a="m0",e1_model_id_b="m1")])
        scores=pd.DataFrame([dict(architecture="gru",seed=s,idx=0,subject="s",label=1,
            masker="train_median",model_id=f"m{s}",original_logit=l,adv_abs_delta_logit=1.,
            adv_signed_margin_drop=.3,relative_gain_abs_delta_logit=.5)
            for s,l in [(0,2.),(1,2.01)]])
        self.assertEqual(join_e1_e2(pairs,scores).abs_delta_relative_gain.iloc[0],0.)
        scores.loc[0,"model_id"]="other"
        with self.assertRaises(AlignmentError): join_e1_e2(pairs,scores)
        scores.loc[0,"model_id"]="m0"
        scores.loc[0,"original_logit"]=3.
        with self.assertRaises(AlignmentError): join_e1_e2(pairs,scores)


if __name__=="__main__": unittest.main()
