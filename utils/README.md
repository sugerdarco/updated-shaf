# SAHF Utilities

This directory contains shared helper tools and utilities used across the framework.

## File Structure

```text
utils/
├── README.md                 # This documentation file
└── logger.py                 # Outlier and escalation tracking system
```

## Details
The `logger.py` module tracks step-by-step metrics during generation. It records when the Divergence Gate opens/closes, when an agent is flagged as an outlier (Stage 6), and how many iterations the Geometric Median took to converge (Stage 7). These metrics are dumped into JSON files for offline analysis.
