# Patched Run (Post-Generation String Cleaner) Metrics
*Date: 2026-08-13 | Subset: 250 Prompts | Configuration: 4 Models (Mistral, Llama-2, Yi, Qwen) with `output_cleaner.py`*

## Overview
We applied a post-generation string filter (`utils/output_cleaner.py`) to the end of the `generate()` loop in `sheaf_orchestrator.py`. This filter was designed to strip out the visible artifacts caused by the byte-fusion pipeline:
1. Repeated punctuation from Yi (e.g. `'''`)
2. Doubled/tripled letters from byte-bloat (e.g. `secretedd`)
3. Infinite repetition loops
4. Long trailing hallucination paragraphs.

## Results (Exact Match Accuracy)
* **Pre-patch (Uncleaned) Ensemble Accuracy:** 37.60%
* **Post-patch (Cleaned) Ensemble Accuracy:** **24.40%**
* *(Target Baseline: Llama-2 at 67.60%)*

## Analysis & Key Findings
The accuracy dropped significantly. While the visual artifacts (bad spelling, weird quotes) were successfully removed, the aggressive truncation of infinite loops and trailing paragraphs destroyed the exact-match score. 

The unpatched ensemble was generating 1024-byte walls of text. Because the evaluation script uses a simple substring check (`alias in output`), the ensemble was being rewarded for generating massive amounts of text, as the correct answer would often randomly appear somewhere within the noise.

By cleaning and truncating the output to the first reasonable sentence, we removed the ensemble's "guess and check" advantage. This reveals that the **true, underlying accuracy** of the corrupted fusion algorithm is much closer to ~24% than 37%.

## Conclusion
Post-hoc string manipulation is a band-aid that cannot fix algorithmic degradation. The **Stage 8 Byte Fusion** logic (which uses the Weiszfeld geometric median) is fundamentally causing the 4 distinct tokenizers to fight over token lengths, misaligning the byte stream. To surpass the **67.60%** baseline, the root cause must be patched directly in the mathematical fusion layer (`stage_8_sheaf_reconciliation.py`).
