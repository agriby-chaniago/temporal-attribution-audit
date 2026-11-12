# Implementation validation — FAST execution version

The builder generated new notebooks; none of their scientific or benchmark cells
were executed by the agent. The parent builder, sealed plan, notebooks, weights
and research-result files were not edited.

## Completed

Seven software-only tests passed on 20 October 2026:

1. Generated notebook schema, Python-source compilation and global dependencies.
2. Identical parent statistical code/analysis blocks; no model batching, mixed
   precision or CUDA Graphs; correction of the final checksum/row-count shadowing.
3. Exact replacement bits on CPU tensors, including dtype conversion, negative
   zero, incomplete tails, empty/full/noncontiguous masks and source immutability.
4. Identical original versus memoized energies, candidate selection, component
   fields and summaries on synthetic arrays with a non-model arithmetic fixture.
5. Read-through prediction paths and preservation of old fixture bytes.
6. Partial record identity, checksum/namespace validation and no copying/deletion
   of the original fixture cache.
7. Refusal of a live reference lock without mutating that lock.

The tests allocate CPU tensors only; they do not call a trained network, use
study recordings, run CUDA, bootstrap study data or estimate a research effect.
The test command and exact scope are in README.md and test_fast_builder.py.

## Pending author execution

- CUDA input-bit and logit comparisons on actual short/long recordings.
- End-to-end reference/resident timing, separately by architecture.
- Runtime choice of resident versus reference; numerical fallback checks.
- Full intervention/analysis completion and result-level validation.

No speedup or new scientific result is claimed. A backend is retained by the
notebook only after the user-run numerical gate and measured improvement.

## CUDA Graphs decision

Static review found host operations in both unchanged encoders: the GRU path
copies lengths to CPU; Mamba-2 converts lengths to a Python list. Full-forward
capture is therefore not enabled. No graph capture or encoder rewrite was made.
The policy follows the static-shape/control-flow and host-operation restrictions
in the [PyTorch 2.11 CUDA Graphs documentation](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs).
Batching and cross-backend numerical identity are not assumed; see
[PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html).

## Provenance

BUILD_CHECKS.json records the generated notebooks and static status.
execution_lock.json binds the new builder and execution amendment to the parent
scientific design. All science-changing settings and original source hashes
remain the sealed parent's. The new runtime stores its validation profile and
checksums before using a faster execution path.
