# SAHF Pipeline

This directory contains the fully decoupled, stage-by-stage implementation of the SAHF architecture. The system orchestrates multiple language models, mathematically resolving their disagreements to produce a single, robust consensus generation.

## Architecture Flow

```text
                  ┌─────────────────────────────────────────┐
                  │       Stage 0: Agent Ensemble           │
                  │         (Llama, Qwen, Mistral)          │
                  └───────────────────┬─────────────────────┘
                                      │
                                      ▼
                  ┌─────────────────────────────────────────┐
                  │   Stage 1: Amplitude Interception       │
                  │        (Logits -> Amplitudes)           │
                  └───────────────────┬─────────────────────┘
                                      │
                                      ▼
                  ┌─────────────────────────────────────────┐
                  │      Stage 4: Allgather Ψ Vectors       │
                  └───────────────────┬─────────────────────┘
                                      │
   ┌──────────────────────────────────┼──────────────────────────────────┐
   │                                  ▼                                  │
   │     [STAGE 8: SHEAF RECONCILIATION (Cross-Tokenizer Byte Tree)]     │
   │                                                                     │
   │                   ┌─────────────────────────────┐                   │
   │                   │  Stage 2: Divergence Gate   │                   │
   │                   │      (Do they agree?)       │                   │
   │                   └───────┬─────────────┬───────┘                   │
   │                           │             │                           │
   │                   [ YES ] │             │ [ NO ]                    │
   │                           ▼             ▼                           │
   │  ┌───────────────────────────┐    ┌──────────────────────────────┐  │
   │  │ Stage 3: Fast Passthrough │    │  Stage 5: Fast Mean Fusion   │  │
   │  └───────────┬───────────────┘    └─────────────┬────────────────┘  │
   │              │                                  │                   │
   │              │                                  ▼                   │
   │              │                    ┌──────────────────────────────┐  │
   │              │                    │    Stage 6: Outlier Check    │  │
   │              │                    └───────┬──────────────┬───────┘  │
   │              │                            │              │          │
   │              │                 [NO OUTLIER]              │ [YES]    │
   │              │                            │              ▼          │
   │              │                            │ ┌────────────────────┐  │
   │              │                            │ │ Stage 7: Geom. Med │  │
   │              │                            │ └────────────┬───────┘  │
   │              │                            │              │          │
   │              ▼                            ▼              ▼          │
   │            ( ──────────────── Consensus Bytes ───────────────── )   │
   └──────────────────────────────────────────┬──────────────────────────┘
                                              │
                                              ▼
                  ┌─────────────────────────────────────────┐
                  │        Decode to Tokens & Continue      │
                  └─────────────────────────────────────────┘
```

## File Structure

```text
pipeline/
├── README.md                                 # This documentation file
├── orchestrator_pipeline.py                  # The main conductor loop connecting all stages
├── stage_0_agent_ensemble.py                 # Agent wrapping and generation (Hugging Face)
├── stage_1_amplitude_interception.py         # Logits to amplitude vectors
├── stage_2_divergence_gate.py                # Heuristic gate to skip heavy math
├── stage_3_fast_passthrough.py               # Fast passthrough (bypassing fusion)
├── stage_4_allgather_vectors.py              # Gathering probability distributions
├── stage_5_fast_mean_fusion.py               # Mean fusion algorithm
├── stage_6_and_7_robust_escalation.py        # Outlier checks & Geometric Median math
└── stage_8_sheaf_reconciliation.py           # Cross-tokenizer byte-level sheaf fusion
```

## Architectural Stages

### Stage 0 — Agent Ensemble (`stage_0_agent_ensemble.py`)
Wraps the Hugging Face models (e.g., Llama, Qwen, Mistral). Requires at least 3 models to enable majority voting.

### Stage 1 — Amplitude Interception Kernel (`stage_1_amplitude_interception.py`)
Intercepts the raw output probabilities (logits) from the models right before they pick a token, converting them into normalized Euclidean "amplitude vectors".

### Stage 2 — Divergence Gate (`stage_2_divergence_gate.py`)
A fast heuristic check that evaluates the top choices from each model. If all models strongly agree, it opens the gate to skip the complex math.

### Stage 3 — Path A: Fast Passthrough (`stage_3_fast_passthrough.py`)
If Stage 2 indicates consensus, this stage executes a fast shortcut, directly sampling the agreed-upon token and skipping Stages 4-7 entirely.

### Stage 4 — Allgather Ψ Vectors (`stage_4_allgather_vectors.py`)
If models disagree (Stage 2 closed), this stage gathers the complete 100k+ probability vectors from every agent for deep mathematical analysis.

### Stage 5 — Fast Mean Fusion (`stage_5_fast_mean_fusion.py`)
The initial attempt at resolving disagreement by calculating the mathematical average of all agents' probability distributions. Fast, but vulnerable to poisoned models.

### Stage 6 — Outlier Check (`stage_6_and_7_robust_escalation.py`)
Checks if any agent's probability vector is suspiciously distant from the Stage 5 average, acting as a detection mechanism for hallucinations or poisoning.

### Stage 7 — Escalation Tier: Geometric Median (`stage_6_and_7_robust_escalation.py`)
Triggered only if Stage 6 detects an outlier. Uses heavy Weiszfeld Geometric Median math to find the true center of the ensemble, mathematically neutralizing the poisoned agent.

### Stage 8 — Sheaf Reconciliation (`stage_8_sheaf_reconciliation.py`)
The universal cross-tokenizer engine. Because different models have different vocabularies, they cannot be directly averaged. This stage acts as a universal translator:
1. Translates model tokens into universal raw bytes via the `prefix_tree_build` module.
2. Runs Stages 5, 6, and 7 *inside* the byte tree to find the consensus.
3. Translates the winning bytes back into model-specific tokens for the next step.
