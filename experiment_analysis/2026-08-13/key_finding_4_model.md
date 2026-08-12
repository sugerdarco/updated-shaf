# Key Findings: The 4-Model Cross-Tokenizer Bottleneck
*Date: 2026-08-13 | 4 Models: Mistral, Llama, Yi, Qwen | max_new_bytes: 1024 | Prompts: 250*

## The Experiment Goal
We attempted to beat the 40.40% baseline accuracy by addressing two issues:
1. Increasing `max_new_bytes` to `1024` so the models could finish their sentences without being prematurely cut off.
2. Adding a 4th powerful model (`Qwen/Qwen1.5-7B-Chat`) to the ensemble to increase the collective intelligence of the cluster.

## The Surprising Result
Despite giving the models more room to speak and more intelligence, the exact-match accuracy **dropped to 37.60%**.

## Why did it fail? The "Byte Bloat" Spelling Phenomenon
The logs revealed a fascinating artifact of cross-tokenizer topological fusion. When 4 entirely different models (from different base languages and tokenizer algorithms) are forced to fuse their output probability distributions *byte-by-byte*, they physically fight over the spelling of words.

For example, when attempting to output the word "Insulin", the differing vocabularies pulled the Weiszfeld geometric median in multiple directions, resulting in the generation of text like: 
> *"Insuling is secretedd by thee pancreas too regulated"*

Because the generated text was littered with these bizarre, averaged misspellings, the automated TriviaQA evaluation script could no longer find exact substring matches for the ground-truth entities, artificially tanking our accuracy score.

## The DeepEn SOTA Insight
This experiment proves exactly why the **DeepEn (NeurIPS)** framework utilizes Minimum Bayes Risk (MBR) decoding at the *sequence level* (generating full sentences and picking the best one) rather than fusing at the *byte level*. SAHF's byte-by-byte fusion mathematically works for identical tokenizers, but when forcing 4 heterogeneous tokenizers together, the continuous probability interference creates spelling artifacts.

## Next Steps
To beat DeepEn using SAHF, we must fix the spelling artifacts. We need an interception layer (possibly `stage_1b_summary_cut.py` or a de-noising pass) that rounds the fused byte distributions to the nearest valid dictionary token *before* the output is committed to the sequence!
