"""Software fixtures only: no study data, trained model, CUDA or bootstrap."""
from __future__ import annotations

import ast
import builtins
import gzip
import importlib.util
import json
import os
import pickle
import socket
import symtable
import tempfile
import time
import unittest
from collections import namedtuple
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("fast_builder", ROOT / "scripts/build_nb_22_24_fast.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class FixtureAlignmentError(ValueError):
    pass


def definitions(source, names, namespace):
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise AssertionError("Missing requested synthetic definition")
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "fixture", "exec"), namespace)


def namespace():
    result = dict(np=np, torch=torch, json=json, sha256=sha256, expit=expit,
                  AlignmentError=FixtureAlignmentError, time=time, Path=Path,
                  PATCH=7, FRACTIONS=(.05, .10, .20, .30), MASKERS=("train_median", "train_donor"),
                  RANDOM_REPEATS=20, DONOR_REPEATS=5, MATCH_SEED=2104,
                  RMS_REL_TOL=.05, RMS_ABS_TOL=1e-8,
                  MAX_CANDIDATES_PER_BUDGET=4096, CANDIDATE_BATCH=128,
                  METRICS=("abs_delta_logit", "signed_margin_drop", "predicted_class_prob_drop"),
                  PRIMARY_FLOOR=1e-8, EXECUTION_ID="synthetic_execution", timing_rows=[])
    result["digest"] = lambda value: sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    definitions((ROOT / "src/audit_e2.py").read_text(), ("keyed_seed", "apply_mask"), result)
    definitions(builder.legacy_functions(17, ("mask_hash", "mask_runs", "patch_energy", "geometry_candidates", "matched_candidates")),
                ("mask_hash", "mask_runs", "patch_energy", "geometry_candidates", "matched_candidates"), result)
    definitions(builder.legacy_functions(20, ("output_metrics", "component_context", "make_component", "trapezoid_score", "summarize_rank")),
                ("output_metrics", "component_context", "make_component", "trapezoid_score", "summarize_rank"), result)
    return result


class FastBuilderContracts(unittest.TestCase):
    def test_notebook_schema_sources_and_global_dependencies(self):
        for notebook, name in [builder.build_notebook(c) for c in ("newhandpd", "uci395")] + [builder.build_combined()]:
            builder.validate_notebook(notebook, name)
            source = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
            table = symtable.symtable(source, name, "exec")
            bound = {s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
            allowed = bound | set(dir(builtins))
            missing = set()

            def check(scope):
                for symbol in scope.get_symbols():
                    if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in allowed:
                        missing.add(symbol.get_name())
                for child in scope.get_children():
                    check(child)
            check(table)
            self.assertEqual(missing, set(), name)
            self.assertTrue(all(not c.outputs and c.execution_count is None for c in notebook.cells if c.cell_type == "code"))

    def test_scientific_estimators_and_matching_definitions_are_identical(self):
        parent = builder.parent_constants()
        for cohort in ("newhandpd", "uci395"):
            sources = builder.notebook_sources(cohort)
            self.assertEqual(sources["stat"], parent["STAT_METHODS"])
            self.assertEqual(sources["analysis"], parent["ANALYSIS_CELL"])
            self.assertIn("len(pair_curves)==expected", sources["finish"])
            self.assertNotIn("for relative,expected in LOCK", sources["finish"])
            self.assertIn("for relative,pinned_sha256 in LOCK", sources["finish"])
        self.assertNotIn("torch.cuda.graph(", builder.RESIDENT_METHODS)
        self.assertNotIn("autocast", builder.RESIDENT_METHODS)
        self.assertIn("torch.full((1,),", builder.RESIDENT_METHODS)
        self.assertIn("self.model(current,self.lengths)", builder.RESIDENT_METHODS)

    def test_cpu_tensor_replacement_is_bit_exact_and_keeps_tail_and_sources(self):
        ns = namespace()
        definitions(builder.RESIDENT_METHODS, ("resident_mask_input",), ns)
        rng = np.random.default_rng(101)
        for length in (7, 16, 57, 141):
            x = rng.normal(size=(length, 6))
            background = rng.normal(size=x.shape)
            x[0, 0] = -0.
            background[0, 1] = 1. + 2e-8
            base = torch.from_numpy(np.ascontiguousarray(x[None], dtype=np.float32))
            donor = torch.from_numpy(np.ascontiguousarray(background[None], dtype=np.float32))
            original_base, original_donor = base.clone(), donor.clone()
            buffer = torch.empty_like(base)
            w = length // 7
            for selected in (np.array([], int), np.array([0]), np.arange(w), np.arange(0, w, 2)):
                actual = ns["resident_mask_input"](buffer, base, donor, selected, 7)[0].numpy()
                expected = np.ascontiguousarray(ns["apply_mask"](x, background, selected, 7), dtype=np.float32)
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                np.testing.assert_array_equal(base.numpy().view(np.uint32), original_base.numpy().view(np.uint32))
                np.testing.assert_array_equal(donor.numpy().view(np.uint32), original_donor.numpy().view(np.uint32))
            with self.assertRaises(FixtureAlignmentError):
                ns["resident_mask_input"](buffer, base, donor, [0, 0], 7)
            with self.assertRaises(FixtureAlignmentError):
                ns["resident_mask_input"](buffer, base, donor, [w], 7)

    def test_energy_cache_preserves_candidates_components_and_summaries(self):
        reference = namespace()
        fast = namespace()
        reference["timed_forward"] = lambda predictor, array, phase, identity: float(np.square(array).sum())
        definitions(builder.legacy_functions(20, ("new_rank_components",)), ("new_rank_components",), reference)
        definitions(builder.notebook_sources("newhandpd")["make_component"], ("make_component",), fast)
        definitions(builder.RESIDENT_METHODS, ("new_rank_components",), fast)

        class FixturePredictor:
            backend = "synthetic_no_model"

            @staticmethod
            def mask_logits(x, library, masks, identity, fraction):
                return np.array([np.square(fast["apply_mask"](x, library["background"], mask, 7)).sum() for mask in masks])

        rng = np.random.default_rng(3)
        x = rng.normal(size=(213, 6))
        libraries = []
        for masker, count in (("train_median", 1), ("train_donor", 5)):
            for donor_repeat in range(count):
                background = np.zeros_like(x) if masker == "train_median" else rng.normal(size=x.shape)
                libraries.append(dict(masker=masker, donor_repeat=donor_repeat, background=background,
                    background_sha256=sha256(background.tobytes()).hexdigest(),
                    _patch_energy=fast["patch_energy"](x, background, 7)))
        identity = dict(model_id="fixture_model", idx=0, record_id="fixture:0", architecture="gru", seed=3,
                        fold=0, subject="fixture_subject", label=0)
        p = dict(logit=1.5)
        order = np.random.default_rng(99).permutation(len(x) // 7)
        a = reference["new_rank_components"](None, x, p, order, libraries, identity, {})
        b = fast["new_rank_components"](FixturePredictor(), x, p, order, libraries, identity, {})
        self.assertEqual(len(a), len(b))
        for left, right in zip(a, b):
            for key in left:
                if isinstance(left[key], np.ndarray):
                    np.testing.assert_array_equal(left[key], right[key])
                elif isinstance(left[key], float) and np.isnan(left[key]):
                    self.assertTrue(np.isnan(right[key]))
                elif key == "controls":
                    for lc, rc in zip(left[key], right[key]):
                        self.assertEqual(lc.keys(), rc.keys())
                        for field in lc:
                            if isinstance(lc[field], np.ndarray):
                                np.testing.assert_array_equal(lc[field], rc[field])
                            else:
                                self.assertEqual(lc[field], rc[field])
                else:
                    self.assertEqual(left[key], right[key], key)
        for left, right in zip(reference["summarize_rank"](a, {}, identity), fast["summarize_rank"](b, {}, identity)):
            self.assertEqual(json.dumps(left, sort_keys=True), json.dumps(right, sort_keys=True))

    def test_reference_prediction_cache_is_read_through_not_overwritten(self):
        ns = dict(Path=Path)
        definitions(builder.EXECUTION_METHODS, ("prediction_path",), ns)
        with tempfile.TemporaryDirectory() as tmp:
            ns["OUT"] = Path(tmp) / "fast"
            ns["LEGACY_OUT"] = Path(tmp) / "old"
            relative = Path("predictions/model/0.pkl")
            old = ns["LEGACY_OUT"] / relative
            old.parent.mkdir(parents=True)
            old.write_bytes(b"synthetic old cache")
            self.assertEqual(ns["prediction_path"]("model", 0), old)
            self.assertEqual(old.read_bytes(), b"synthetic old cache")
            self.assertEqual(ns["prediction_path"]("model", 1), ns["OUT"] / "predictions/model/1.pkl")

    def test_record_reuse_checks_identity_checksum_and_execution_namespace(self):
        ns = namespace()
        ns.update(Path=Path, pickle=pickle, gzip=gzip, os=os, json=json,
                  EXECUTION_ID="fixture_execution", ROOT=Path("/"))
        definitions(builder.parent_constants()["COMMON"], ("atomic_pickle", "read_local_pickle"), ns)
        registered = []
        ns["register_reuse"] = lambda path, kind: registered.append((path, kind))
        definitions(builder.EXECUTION_METHODS, ("validate_cached_components", "compatible_record_cache"), ns)
        context = dict(model_id="fixture", idx=0, record_id="fixture:0", input_sha256="input",
                       backgrounds=["background"], source_ids={"4": "source_model"})
        with tempfile.TemporaryDirectory() as tmp:
            ns["OUT"] = Path(tmp) / "fast"
            ns["LEGACY_OUT"] = Path(tmp) / "old"
            path = ns["OUT"] / "record_evaluations/fixture/0.progress.pkl"
            legacy = ns["LEGACY_OUT"] / path.relative_to(ns["OUT"])
            saved = dict(context=context, components={})
            ns["atomic_pickle"](legacy, saved)
            checksum = sha256(legacy.read_bytes()).hexdigest()
            actual = ns["compatible_record_cache"](path, context, partial=True)
            self.assertEqual(actual, saved)
            self.assertFalse(path.exists())
            self.assertEqual(sha256(legacy.read_bytes()).hexdigest(), checksum)
            self.assertEqual(registered[0][1], "partial_record")
            with self.assertRaises(FixtureAlignmentError):
                ns["compatible_record_cache"](path, dict(context, record_id="different"), partial=True)
            ns["atomic_pickle"](path, dict(saved, execution_id="another_execution"))
            with self.assertRaises(FixtureAlignmentError):
                ns["compatible_record_cache"](path, context, partial=True)

    def test_active_reference_lock_is_refused_without_mutation(self):
        ns = dict(Path=Path, json=json, socket=socket, os=os, DESIGN_ID="fixture_design", RUNTIME_ID="fixture_runtime")
        definitions(builder.EXECUTION_METHODS, ("assert_reference_runs_idle",), ns)
        with tempfile.TemporaryDirectory() as tmp:
            ns["ROOT"] = Path(tmp)
            path = ns["ROOT"] / "results/attention_replication20/newhandpd/fixture_design/fixture_runtime/interventions.active.lock"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(host=socket.gethostname(), pid=os.getppid())))
            original = path.read_bytes()
            with self.assertRaises(RuntimeError):
                ns["assert_reference_runs_idle"]()
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
