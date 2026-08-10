# Experimental Cross-Tokenizer Architecture

> [!WARNING]
> **DISCLAIMER:** This branch is purely for experiment purposes and nothing else. We do not take reference from this for any outside project used.

## Goal

The main branch contains our working, fully decoupled cross-tokenizer pipeline where the fusion happens inside the Stage 8 Byte-Prefix Tree sandbox. 

However, the goal of this specific experimental branch is to explore and attempt to create a **proper stage-by-stage architecture implementation** that perfectly mirrors the flow of the same-tokenizer baseline, but applied strictly to cross-tokenizers. 

This is a sandbox to see if we can force the cross-tokenizer math to walk the exact same physical sequence as the mono-tokenizer setup.

**Note:** For stable reference code, refer to the `main` branch.

## Context: Same-Tokenizer vs. Cross-Tokenizer

If you are reading this and are confused by what "same-tokenizer" means, here is the context:

### The "Same-Tokenizer" Baseline
In a **Same-Tokenizer** ensemble, every AI agent shares the exact same vocabulary dictionary (e.g., they are all fine-tunes of Llama-3). Because they all speak the exact same "token language", their output probability vectors are identical in length and perfectly aligned. 
* **The Architecture:** Because the vectors align perfectly, the architecture is simple. You can take the output from Agent A and mathematically average it with Agent B directly on the tensors (Stage 5/6/7), append the winning token ID, and pass it back.

### The "Cross-Tokenizer" Challenge
In a **Cross-Tokenizer** ensemble (which this project solves), the agents are totally different (e.g., Llama-3, Qwen, Mistral). They do not share a vocabulary.
* **The Architecture:** You **cannot** average their output vectors because Token ID 1500 means something completely different to Llama-3 than it does to Qwen. Therefore, the pipeline cannot just loop linearly. Instead, it must project all the different tokens into a shared "Byte-Prefix Tree" (Stage 8). The fusion math (Stages 5/6/7) must happen *inside* the nodes of this tree, operating on bytes rather than tokens. 

The goal of this branch is to see if we can forcefully structure the complex Cross-Tokenizer byte-tree math so that it reads and looks like the simple, linear Same-Tokenizer architecture.
