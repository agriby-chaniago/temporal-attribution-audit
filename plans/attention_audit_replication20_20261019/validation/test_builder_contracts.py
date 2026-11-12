"""Synthetic/software checks only. No study data, model, bootstrap or inference.

This file is implementation validation, not a scientific analysis entry point.
Scientific changes belong to scripts/build_nb_21_24.py.
"""
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
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("replication_builder", ROOT/"scripts/build_nb_21_24.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class SyntheticAlignmentError(ValueError):
    pass


def definitions(source, names, namespace):
    tree = ast.parse(source)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise AssertionError("Synthetic check requested missing definitions")
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, "synthetic_definitions", "exec"), namespace)


class BuilderContracts(unittest.TestCase):
    def test_schedule_is_balanced_reciprocal_and_new_seed_set_is_separate(self):
        data = builder.schedule_rows()
        self.assertEqual(len(data), 400)
        for cohort in ("newhandpd", "uci395"):
            for arm in ("gru", "mamba2"):
                g = [r for r in data if r["cohort"] == cohort and r["architecture"] == arm]
                edges = {(r["target_seed"], r["source_seed"]) for r in g}
                self.assertEqual(len(edges), 100)
                self.assertTrue(all(a != b and (b, a) in edges for a, b in edges))
                for seed in range(20):
                    self.assertEqual(sum(r["target_seed"] == seed for r in g), 5)
                    self.assertEqual(sum(r["source_seed"] == seed for r in g), 5)
                if cohort == "newhandpd":
                    self.assertTrue(all(r["target_seed"] >= 3 and r["source_seed"] >= 3
                                        for r in g if r["primary_pair"]))

    def test_notebook_schema_compilation_and_global_dependencies(self):
        for cohort in ("newhandpd", "uci395"):
            notebook, name = builder.build_notebook(cohort)
            builder.validate_notebook(notebook, name)
            source = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
            table = symtable.symtable(source, name, "exec")
            bound = {s.get_name() for s in table.get_symbols()
                     if s.is_assigned() or s.is_imported() or s.is_namespace()}
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

    def test_exact_geometry_rms_and_failed_quota_do_not_fallback(self):
        ns = {"np": np, "AlignmentError": SyntheticAlignmentError}
        source = builder.old_function_sources(17, ("mask_runs", "geometry_candidates", "matched_candidates"))
        definitions(source, ("mask_runs", "geometry_candidates", "matched_candidates"), ns)
        selected = np.array([2, 3, 4, 18])
        result, diagnostics = ns["matched_candidates"](
            np.ones((2, 30)), selected, 180, [3, 3], np.random.default_rng(7), .05, 1e-8, 128, 16)
        self.assertTrue(all(d["complete"] for d in diagnostics))
        for controls in result:
            for control in controls:
                self.assertEqual(ns["mask_runs"](control["selected"])[2], (1, 3))
                self.assertFalse(np.array_equal(control["selected"], selected))
        _, failed = ns["matched_candidates"](
            np.ones((1, 3)), np.array([0, 1]), 18, [20], np.random.default_rng(1), .05, 1e-8, 64, 16)
        self.assertFalse(failed[0]["complete"])
        self.assertEqual(failed[0]["search_status"], "insufficient_matches_within_search_budget")

    def test_hierarchical_weights_keep_subject_weight_equal(self):
        ns = {"np": np}
        definitions(builder.STAT_METHODS, ("balanced_weights",), ns)
        frame = pd.DataFrame([
            ("A", 0, 3, 5), ("A", 0, 3, 6), ("A", 0, 4, 5),
            ("A", 1, 3, 5), ("B", 2, 4, 5),
        ], columns=["subject", "idx", "target_seed", "source_seed"])
        weights = ns["balanced_weights"](frame)
        np.testing.assert_allclose(weights, [.125, .125, .25, .5, 1.])
        np.testing.assert_allclose(frame.assign(weight=weights).groupby("subject").weight.sum(), [1., 1.])

    def test_cache_checksum_and_concurrent_lock(self):
        ns = dict(Path=Path, os=os, json=json, pickle=pickle, gzip=gzip, sha256=sha256,
                  socket=socket, datetime=datetime, timezone=timezone, contextmanager=contextmanager,
                  AlignmentError=SyntheticAlignmentError)
        definitions(builder.COMMON, ("atomic_pickle", "read_local_pickle", "stage_lock"), ns)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            path = directory/"synthetic.pkl"
            ns["atomic_pickle"](path, {"marker": "synthetic", "values": np.array([1, 2, 3])})
            np.testing.assert_array_equal(ns["read_local_pickle"](path)["values"], [1, 2, 3])
            with path.open("rb") as stream:
                envelope = pickle.load(stream)
            envelope["payload"] += b"tampered"
            with path.open("wb") as stream:
                pickle.dump(envelope, stream)
            with self.assertRaises(SyntheticAlignmentError):
                ns["read_local_pickle"](path)
            with ns["stage_lock"](directory, "synthetic"):
                with self.assertRaises(RuntimeError):
                    with ns["stage_lock"](directory, "synthetic"):
                        pass
            self.assertFalse((directory/"synthetic.active.lock").exists())

    def test_dose_gap_is_symmetric_and_zero_zero_is_defined(self):
        ns={"np":np,"AlignmentError":SyntheticAlignmentError}
        definitions(builder.RANK_METHODS,("paired_dose",),ns)
        base=dict(masker="train_median",requested_fraction=.1,donor_repeat=0,
                  target_run_lengths="[2]",selected=np.array([1,2]))
        a=dict(base,target_rms=1.)
        b=dict(base,target_rms=1.04,selected=np.array([2,3]))
        forward=ns["paired_dose"]([a],[b],"train_median")[0]
        backward=ns["paired_dose"]([b],[a],"train_median")[0]
        self.assertAlmostEqual(forward["relative_dose_gap"],backward["relative_dose_gap"])
        self.assertLess(forward["relative_dose_gap"],.05)
        self.assertAlmostEqual(forward["mask_jaccard"],1/3)
        a["target_rms"]=b["target_rms"]=0.
        self.assertEqual(ns["paired_dose"]([a],[b],"train_median")[0]["relative_dose_gap"],0.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
