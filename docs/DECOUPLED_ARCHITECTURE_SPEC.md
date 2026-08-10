# SAHF Decoupled Architecture Specification

This document provides the formal architectural specification for the Decoupled SAHF Cross-Tokenizer Multi-Agent Pipeline.

## 1. Overview

The SAHF (Sheaf-based Amplitude Fusion & Escalation) architecture resolves token-level and distribution-level disagreements among heterogeneous LLM agents operating under mismatched tokenizers (e.g., Llama-3, Qwen, Mistral).

## 2. Pipeline Stages

- **Stage 0: Agent Ensemble** (`pipeline/stage_0_agent_ensemble.py`): Wraps Hugging Face LLM models to emit raw logits per decoding step.
- **Stage 1: Amplitude Interception Kernel** (`pipeline/stage_1_amplitude_interception.py`): Converts logit probability vectors $P_i(v)$ into normalized amplitude vectors $\psi_i = \sqrt{P_i(v)}$ on the unit sphere.
- **Stage 2: Divergence Gate** (`pipeline/stage_2_divergence_gate.py`): Evaluates ensemble divergence and entropy to determine if fast-path sampling can be used.
- **Stage 3: Fast Passthrough** (`pipeline/stage_3_fast_passthrough.py`): Bypasses full fusion when high consensus is detected.
- **Stage 4: Allgather Vectors** (`pipeline/stage_4_allgather_vectors.py`): Gathers complete probability vectors across distributed agent instances.
- **Stage 5: Fast Mean Fusion** (`pipeline/stage_5_fast_mean_fusion.py`): Computes chordal mean fusion on the amplitude sphere.
- **Stage 6: Outlier Check** (`pipeline/stage_6_and_7_robust_escalation.py`): Robust MAD-based outlier filtering to detect hallucinated or adversarial model output.
- **Stage 7: Escalation Tier** (`pipeline/stage_6_and_7_robust_escalation.py`): Computes Riemannian Weiszfeld Geometric Median with a 50% breakdown point.
- **Stage 8: Sheaf Reconciliation** (`pipeline/stage_8_sheaf_reconciliation.py`): Cross-tokenizer byte-level sheaf fusion mapping token distributions onto a shared byte prefix tree.

## 3. Byte Prefix Tree Space

Cross-tokenizer fusion uses a shared byte base space constructed in `prefix_tree_build/`:
- **`vocab.py`**: Extracts raw byte tokens across SentencePiece, BPE, and WordPiece schemes.
- **`prefix_tree.py`**: Constructs the shared Byte Prefix Trie and propagates probability mass.
- **`build_prefix_tree.py`**: Offline CLI builder for pre-computing tree artifacts (`prefix_tree.npz`).
