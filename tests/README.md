# SAHF Unit Tests

This folder will contain the unit tests for the decoupled SAHF architecture. Because we have restructured the pipeline into distinct stages, our testing strategy will be highly modular.

## Planned Test Structure

```text
tests/
├── test_stage_1_amplitude.py      # Tests logits -> amplitude conversion and Euclidean norms
├── test_stage_2_gate.py           # Tests entropy and divergence thresholds
├── test_stage_5_fusion.py         # Tests fast mean averaging mathematics
├── test_stage_6_7_escalation.py   # Tests outlier masking and Geometric Median convergence
└── test_stage_8_reconciler.py     # Tests tree traversal and conditional probability decoding
```

*Note: The adversarial tree math and vocabulary detection tests are already implemented in `../audit/audit_prefix_tree.py`.*
