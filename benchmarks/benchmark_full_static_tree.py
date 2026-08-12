"""
Benchmark: full static tree construction.

Measures latency and node count when the BytePrefixTree is built once over the
agents' COMPLETE vocabularies (`mode="full"` / a prebuilt artifact), rather than
per step over the top-k union.
"""

import time

import numpy as np

from prefix_tree_build.prefix_tree import BytePrefixTree
from prefix_tree_build.vocab import VocabSpec


def generate_mock_vocab(agent_idx: int, size: int) -> VocabSpec:
    """VocabSpec maps token id -> raw BYTES; it has no agent_idx/token_ids fields.

    Ids are positional in `token_bytes`, and the agent's index is its position in
    the list handed to `from_vocabs`, so neither needs to be stored here.
    """
    rng = np.random.default_rng(agent_idx)
    lengths = rng.integers(1, 9, size=size)
    mapping = {
        i: bytes(rng.integers(32, 127, size=int(n)).tolist()) for i, n in enumerate(lengths)
    }
    return VocabSpec.from_mapping(mapping, name=f"agent_{agent_idx}")


def run_benchmark(vocab_size: int = 32_000, n_agents: int = 3):
    print("--- Benchmarking Static BytePrefixTree Construction ---")
    print(f"Agents: {n_agents}, Vocab size per agent: {vocab_size:,}")

    vocabs = [generate_mock_vocab(i, vocab_size) for i in range(n_agents)]

    start_time = time.perf_counter()
    # BytePrefixTree() takes no arguments -- the trie is built by the
    # `from_vocabs` classmethod, which also builds the per-agent path index.
    tree = BytePrefixTree.from_vocabs(vocabs)
    elapsed = (time.perf_counter() - start_time) * 1000

    print(f"Construction time: {elapsed:.2f} ms")
    print(f"Total tree nodes : {tree.n_nodes:,}")
    print(f"Max depth        : {tree.max_depth_seen}")


if __name__ == "__main__":
    run_benchmark()
