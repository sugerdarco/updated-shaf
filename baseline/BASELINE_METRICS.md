# Individual Model Baseline Metrics
*Date: 2026-08-13 | Subset: 250 Prompts | Configuration: Independent Standard Inference (max_new_tokens=300)*

## Overview
To establish a true baseline to compare the SAHF ensemble architecture against, we ran the 4 models independently on the exact same 250-prompt subset used in the ensemble testing. This determines the "Best Single Model" accuracy, which the DeepEn paper asserts an ensemble must exceed (by at least +2.05%) to be considered state-of-the-art.

## Results (Exact Match Accuracy)
* **LLAMA-2** (`meta-llama/Llama-2-13b-chat-hf`): **67.60%**  🏆 *(Best Single Model)*
* **MISTRAL** (`mistralai/Mistral-7B-Instruct-v0.2`): **66.00%**
* **YI** (`01-ai/Yi-6B-Chat`): **57.60%**
* **QWEN** (`Qwen/Qwen1.5-7B-Chat`): **34.00%**

## The Target
The Best Single Model is **Llama-2 at 67.60%**.

## Comparison to SAHF Ensemble
When these exact same 4 models were combined in the cross-tokenizer SAHF byte-fusion pipeline on the same prompts, the ensemble score plummeted to **37.60%**. 

This severe performance degradation highlights the **Byte-Bloat Phenomenon**: the models mathematically fought over token alignments at the byte level, resulting in severe spelling artifacts (e.g., *"Insuling is secretedd"*). These spelling artifacts destroyed the ability to evaluate exact-match strings. 

## Next Steps
To beat DeepEn and prove the SAHF architecture, the pipeline must be upgraded to resolve the byte-bloat interference layer. The target for the repaired SAHF ensemble is **> 67.60%**.
