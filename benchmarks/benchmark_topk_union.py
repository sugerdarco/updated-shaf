"""
Benchmark: per-step top-k union tree construction.

`topk_union` mode rebuilds the trie on EVERY decoding step, restricted to the
union of each agent's top-k token ids, so this cost sits directly in the hot
path. Compare against benchmark_full_static_tree.py, which pays the cost once.
"""

import time

import numpy as np

from prefix_tree_build.prefix_tree import BytePrefixTree
from prefix_tree_build.vocab import VocabSpec
from pipeline.stage_0_agent_ensemble import top_k_ids


def mock_vocab(name: str, size: int, seed: int) -> VocabSpec:
    """A byte-native vocabulary of `size` tokens with realistic length spread.

    Token *bytes* are what the trie is built from, so the mock has to produce
    real byte strings -- a name/size pair carries no byte image and the tree has
    nothing to insert.
    """
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, 9, size=size)
    mapping = {
        i: bytes(rng.integers(32, 127, size=int(n)).tolist()) for i, n in enumerate(lengths)
    }
    return VocabSpec.from_mapping(mapping, name=name)


def benchmark_topk_union_tree_build(vocab_size: int = 128_000, k: int = 16, trials: int = 20):
    """Millisecond cost of the dynamic tree build, per decoding step."""
    print("--- Benchmarking Top-K Union Tree Build ---")
    print(f"Agents: 3, Vocab size per agent: {vocab_size:,}, K={k}, trials={trials}")

    vocabs = [mock_vocab(f"Agent{c}", vocab_size, seed) for c, seed in zip("ABC", (0, 1, 2))]
    rng = np.random.default_rng(7)

    # Dirichlet over 128k categories is very slow and irrelevant to what is being
    # timed; an exponential draw normalised to the simplex is the same shape of
    # input for top-k purposes and costs nothing.
    probs = []
    for _ in range(3):
        p = rng.exponential(size=vocab_size)
        probs.append(p / p.sum())

    elapsed, n_nodes = [], 0
    for _ in range(trials):
        t0 = time.perf_counter()
        restrict = [top_k_ids(p, k) for p in probs]
        tree = BytePrefixTree.from_vocabs(vocabs, restrict=restrict, max_depth=8)
        elapsed.append((time.perf_counter() - t0) * 1000)
        n_nodes = tree.n_nodes

    print(f"Tree build: {np.mean(elapsed):.2f} ms / step (min {np.min(elapsed):.2f})")
    print(f"Tree size : {n_nodes} nodes")


if __name__ == "__main__":
    benchmark_topk_union_tree_build()
