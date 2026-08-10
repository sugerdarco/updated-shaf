# SAHF Benchmarks

This folder will contain the performance measurement tools for the cross-tokenizer pipeline. 

## Planned Benchmark Structure

```text
benchmarks/
├── README.md                      # This documentation
├── benchmark_topk_union.py        # Measures RAM/Latency for dynamic step-by-step tree building
├── benchmark_full_static_tree.py  # Offline evaluation of building a static tree over all vocabularies
└── benchmark_escalation_cost.py   # Measures the exact millisecond delay of triggering Stages 5-7
```

Because the original benchmark data relies on the old monolithic architecture, we will rebuild these scripts to import directly from `../pipeline/orchestrator_pipeline.py`.
