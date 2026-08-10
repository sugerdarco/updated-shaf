# SAHF Runners

This directory contains the execution layer of the SAHF framework. These scripts are responsible for taking user input, loading the configuration and models, and executing the cross-tokenizer decoding loop defined in the `pipeline` directory.

## File Structure

```text
runners/
├── README.md                 # This documentation file
├── run_sheaf.py              # Main CLI entry point for running a single prompt
├── run_batch_prompts.py      # Batch executor for running multiple prompts
├── demo_sheaf_mock_run.py    # A mock run script for testing without heavy GPUs
└── sheaf_orchestrator.py     # The wrapper that bridges CLI args to the pipeline
```

## How It Works
The `run_sheaf.py` script acts as the primary user interface. It parses `config_sheaf.yaml`, instantiates the PyTorch models, and hands them to the `SheafOrchestrator` (`sheaf_orchestrator.py`). The Orchestrator then spins up the core mathematical loop located in `../pipeline/orchestrator_pipeline.py`.
