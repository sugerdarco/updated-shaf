# Experimental Cross-Tokenizer Architecture

> [!WARNING]
> **DISCLAIMER:** This branch is purely for experiment purposes and nothing else. We do not take reference from this for any outside project used.

## Goal

The main branch contains our working, fully decoupled cross-tokenizer pipeline where the fusion happens inside the Stage 8 Byte-Prefix Tree sandbox. 

However, the goal of this specific experimental branch is to explore and attempt to create a **proper stage-by-stage architecture implementation** that perfectly mirrors the flow of the same-tokenizer baseline, but applied strictly to cross-tokenizers. 

This is a sandbox to see if we can force the cross-tokenizer math to walk the exact same physical sequence as the mono-tokenizer setup.

**Note:** For stable reference code, refer to the `main` branch.
