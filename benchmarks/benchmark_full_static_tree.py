"""
Benchmark: Full Static Tree Construction
Measures memory usage and latency when constructing a static BytePrefixTree
across full model token vocabularies.
"""

import time
import numpy as np

from prefix_tree_build.prefix_tree import BytePrefixTree, VocabSpec

def generate_mock_vocab(agent_idx: int, size: int) -> VocabSpec:
    token_bytes = [f"token_{i}".encode("utf-8") for i in range(size)]
    token_ids = list(range(size))
    return VocabSpec(agent_idx=agent_idx, token_bytes=token_bytes, token_ids=token_ids)

def run_benchmark():
    vocab_size = 32000
    n_agents = 3
    print(f"--- Benchmarking Static BytePrefixTree Construction ---")
    print(f"Agents: {n_agents}, Vocab size per agent: {vocab_size}")

    start_time = time.perf_counter()
    vocabs = [generate_mock_vocab(i, vocab_size) for i in range(n_agents)]
    tree = BytePrefixTree(vocabs)
    elapsed = (time.perf_counter() - start_time) * 1000

    print(f"Construction time: {elapsed:.2f} ms")
    print(f"Total Tree Nodes: {tree.n_nodes}")

if __name__ == "__main__":
    run_benchmark()
