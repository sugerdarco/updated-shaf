# SAHF 4-Model Analytical Metrics
*Date: 2026-08-13 | 4 Models: Mistral, Llama, Yi, Qwen | max_new_bytes: 1024 | Prompts: 250*

## 1. Rebel Outlier Metric (MAD Flagging)
* **Mistral:** 0
* **Llama-2:** 0
* **Yi:** 0
* **Qwen:** 0

**Analysis:** Adding a 4th model mathematically diluted the Median Absolute Deviation (MAD). Because there are now 4 competing opinions, the variance in the Weiszfeld geometric median increased, meaning no single model was far enough away from the median to trigger the `3.0x` MAD threshold. The ensemble lost its ability to confidently veto "bad" models.

## 2. Entropy vs. Divergence
* **Avg Entropy (Fast-Path):** `0.0313`
* **Avg Entropy (Escalated):** `0.3217`
* **Average Pairwise Divergence (D):** `1.3140`

**Analysis:** The average divergence skyrocketed (1.3140 is massive). The 4 models strongly disagreed on almost every byte generated. The divergence gate successfully caught this (routing high entropy to the escalated path), but the models simply couldn't agree.

## 3. Escalation and Complexity
* **Prompts with >95% Escalated Steps:** `234 / 250 (93.6%)`

**Analysis:** Almost every single prompt was escalated to the slow Weiszfeld Geometric Median path. The "Fast-Path" effectively died because adding Qwen to the mix introduced too much cross-tokenizer conflict.

## 4. Dataset Accuracy (Exact Match)
* **Evaluated:** 250 prompts
* **Correct:** 94
* **Accuracy:** `37.60%`

**Analysis:** Despite increasing `max_new_bytes` to `1024` so the models could finish their sentences, the accuracy *dropped* from the 3-model baseline of `40.40%`. Why? Because of **Byte-Level Interference**. The models fought over the spelling of words (e.g., generating *"Insuling is secretedd by thee pancreas"*). This bizarre spelling phenomenon means the final text no longer exact-matches the correct dataset answers, artificially destroying the score. This proves that linear byte-by-byte fusion across 4 radically different tokenizers creates spelling artifacts!
