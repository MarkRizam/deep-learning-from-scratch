# AIR-004 V1.1 — scientific audit

Baseline: accdf8ae5c9845459dc8e708a34ee41a178ce80d. Original [hosted CI](https://github.com/MarkRizam/deep-learning-from-scratch/actions/runs/45535401841) passed. Reviewed layer equations, training loop, optimizer state, experiment runner and protocol.

## Findings and corrections

- Backward gradients now require exact shape and finite real inputs; no implicit broadcasting across batch or feature dimensions.
- Dense input must be rank 2 with the expected feature count; convolution and flatten shapes are validated. Positive integer layer dimensions, epochs and batch sizes are enforced; invalid initialization scales are rejected.
- Dense, activation and convolution cache copies of inputs. A dense regression test confirms that caller mutation cannot alter cached input gradients.
- Optimizer updates are staged and validated before any parameter or state mutation, including exact gradient shape, finite writable float64 parameters and rejection of overlapping or duplicate parameter storage. Overflow raises rather than committing a partial update. This is single-step validation, not an entire-training transaction.
- Training validates the complete label vector before optimizer updates. MSE overflow and nonfinite cross-entropy raise explicitly.

## Executed verification

Local Python 3.13.5, NumPy 2.3.5 and CPU PyTorch 2.10.0: 53/53 tests passed (23 new cases); Python compilation passed. Committed-result verification passed. Two fresh runs matched all 63 original records exactly locally: maximum absolute difference 0.0 for both baseline-to-fresh and fresh-to-fresh, at declared tolerance 1e-9. Original benchmark file unchanged. One expected RuntimeWarning (overflow in logit subtraction) remains in the extreme cross-entropy test before the explicit exception. No warning-free claim.

## Scientific interpretation

Dense and convolution forward/backward comparisons use float64 PyTorch references on small tensors. The central difference check covers a selected weight; it is not exhaustive gradient verification. ReLU inactive units count toward the reported low-derivative fraction. XOR accuracy is training accuracy on four points, not a generalization estimate. Optimizer and initialization experiments use first-epoch losses as references, not untrained initial losses or compute-matched budgets. Regularization results use a shared held-out comparison set with no selected winner or independent final test; L2 is coupled and includes biases. The CNN and MLP synthetic image comparison has different parameter counts. Means and sample SD over three seeds are descriptive, not confidence intervals. No external dataset or clinical claim.

## Remaining limitations

Fresh hosted CI required before release. Full tensor autodiff, shared/reused layer graphs, general cache lifecycle and mutation handling, broader extreme-value arithmetic, pooling, normalization, recurrent layers, GPU, Docker, lint and runtime/memory benchmarking remain outside verified scope. The library is educational and not a replacement for production frameworks.

Reproduce: `python -m pytest -q`, then `python scripts/verify_committed_results.py`.
