# Strategic Goal: Challenging the DeepEn (NeurIPS) SOTA

This document outlines the concrete steps and goals required for the SAHF (Sheaf-based Amplitude Fusion) architecture to officially challenge and defeat the current State-of-the-Art cross-tokenizer ensemble framework, **DeepEn**.

## The Current SOTA Landscape
* **DeepEn's Performance:** Achieves ~65% to 74.7% on TriviaQA (depending on the base models).
* **DeepEn's Core Metric:** Demonstrates an average **+2.05% accuracy boost** over the Best Single Model (BSM) by mapping probability distributions into a universal relative space.
* **SAHF's Current Baseline:** 40.40% on TriviaQA (artificially suppressed due to a strict 256 byte generation limit).

## Goal: The Blueprint to Beat DeepEn

To prove that SAHF's topological byte-prefix tree architecture is superior to DeepEn's relative-space mapping, the next phase of experiments must accomplish the following three objectives:

### 1. Remove Artificial Bottlenecks
* **Action:** Increase `max_new_bytes` in `config_sheaf.yaml` from `256` to `1024` (or higher).
* **Why:** Cross-tokenizer generation forces the models to step byte-by-byte. A 256-byte limit forcibly truncates outputs mid-sentence (at ~256 characters), destroying accuracy on complex factual questions. Giving the models room to finish their answers is mandatory for a fair SOTA comparison.

### 2. Establish the BSM (Best Single Model) Baseline
* **Action:** Run the exact same TriviaQA evaluation subset on Mistral-7B, Llama-2-13B, and Yi-6B completely independently (without the SAHF fusion pipeline).
* **Why:** You cannot beat DeepEn by comparing absolute numbers (e.g., 74%), because the absolute number depends entirely on how smart the base models are. We must find out which of our 3 models is the smartest, and record its individual accuracy. That number becomes our BSM baseline.

### 3. Prove the "Wisdom of the Crowd" Delta
* **Action:** Run the fully uncapped SAHF ensemble and compare its accuracy to the BSM.
* **Why:** The SOTA victory condition is the *delta*. If SAHF boosts the BSM by **more than +2.05%**, it mathematically proves that fusing raw amplitudes in a byte-prefix tree (SAHF) extracts more synergy from heterogeneous models than projecting them into a relative space (DeepEn). 

### Secondary Victory Condition: Execution Speed
If SAHF achieves a similar accuracy boost to DeepEn but generates tokens at a faster rate (less computational overhead per step), SAHF still claims the SOTA title for real-time inference efficiency. Our current sustained speed of **~140ms per token (7.1 tokens/sec)** across 3 massive models is a highly competitive metric to lean on.
