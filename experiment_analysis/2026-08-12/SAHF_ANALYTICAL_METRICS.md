# SAHF Cross-Tokenizer Analytical Metrics Framework

This document outlines the metrics necessary to deeply understand the behavior of the Sheaf-based Amplitude Fusion (SAHF) cross-tokenizer pipeline and to extract key findings from the 1,000 prompt batch experiment.

## I. Baseline Pipeline Routing Metrics (Already Calculated)

These are the fundamental topological routing metrics output directly by the SAHF pipeline for every token generated.

| Metric | Definition | Mathematical Condition | Result (1K Prompts) |
| :--- | :--- | :--- | :--- |
| **Total Tokens** | The absolute number of decoding steps executed across all prompts. | N/A | 61,068 (100%) |
| **Fast-Path** | Tokens where all models had near-perfect agreement, bypassing complex fusion. | Divergence $D < 0.25$ | 1,084 (1.78%) |
| **Fusion** | Tokens requiring standard spherical averaging (Chordal Mean) on the byte tree. | Entropy $H < 1.5$ & $D \ge 0.25$ | 59,984 (98.22%) |
| **Escalated** | Tokens requiring robust outlier rejection and Weiszfeld Geometric Median calculation. | $MAD_{score} > 3.0$ | 54,688 (89.55%) |

---

## II. Proposed Advanced Analytical Metrics (To Be Calculated)

To uncover *why* the pipeline behaves the way it does, we must parse the internal step-by-step trace logs (`steps.jsonl`) and calculate the following 5 advanced metrics:

### 1. The "Rebel" Outlier Metric (Model Disagreement Distance)
* **What it measures:** During the 54,688 Escalated tokens, which specific model (Mistral, Llama, or Yi) was most frequently identified by the MAD (Median Absolute Deviation) gate as the outlier?
* **Key Finding Goal:** Determines if one specific model's architecture or training data consistently rebels against the ensemble consensus. It reveals the weak link or the most "unique" model in the group.

### 2. Escalation-to-Entropy Correlation
* **What it measures:** The mathematical correlation between the ensemble's average Byte-Prefix Tree Entropy ($H$) and the routing path taken. 
* **Key Finding Goal:** Proves whether "cross-model disagreement" (Escalation) is directly caused by "high individual uncertainty" (Entropy). If escalated tokens have massive entropy spikes, it means the models aren't just disagreeing with each other; they are individually confused by the prompt.

### 3. Prompt Complexity vs. Consensus Ratio
* **What it measures:** Grouping prompts by their individual Escalation Percentage to see which types of questions cause the highest disagreement.
* **Key Finding Goal:** Tests if the SAHF Divergence Gate acts as an implicit "Complexity Detector". We expect factual questions to have higher consensus (more Fast-Path/Fusion) and subjective/complex reasoning questions to have near 100% Escalation.

### 4. Cross-Tokenizer Byte-Bloat Ratio
* **What it measures:** The average number of raw bytes generated per decoding step compared to a standard token.
* **Key Finding Goal:** Because the models have completely different vocabularies, they are forced to agree byte-by-byte on a shared tree. This metric reveals if the cross-tokenizer constraint forces the models into a "stuttering" effect, taking much smaller, sub-word byte chunks rather than full-word tokens.

### 5. Pairwise Amplitude Divergence Distribution
* **What it measures:** A histogram of the Divergence ($D$) scores across all 61,000 steps.
* **Key Finding Goal:** Visualizes the severity of the disagreement. If the distribution heavily skews towards extreme divergence, it means the models are attempting to take completely opposite semantic paths. If it hovers just above the $0.25$ threshold, it means the models are only slightly disagreeing on exact word choice (synonyms).

---

## III. Summary Table of Advanced Metrics to Compute

| Proposed Metric | Data Source (`steps.jsonl`) | Output Format | Purpose |
| :--- | :--- | :--- | :--- |
| **Model Outlier Frequency** | `mad_outliers` array | Percentage per model | Identify the most contrarian LLM |
| **Average Node Entropy** | `entropy` float | Float (Bits) | Measure model uncertainty |
| **Per-Prompt Escalation %** | Aggregated `path` string | Float (0.0 - 1.0) | Detect prompt complexity |
| **Byte-Bloat Ratio** | `byte_length` int | Float (Bytes/Step) | Measure cross-tokenizer inefficiency |
| **Divergence Severity** | `divergence` float | Histogram / Float | Measure semantic disagreement severity |
