import time
import numpy as np

from prefix_tree_build.prefix_tree import BytePrefixTree
from prefix_tree_build.vocab import VocabSpec
from pipeline.stage_0_agent_ensemble import top_k_ids

def benchmark_topk_union_tree_build():
    """
    Measures the exact millisecond cost of building the dynamic tree 
    for `topk_union` mode, which happens on every decoding step.
    """
    # Create mock vocabs (128k tokens each)
    mock_vocabs = {
        "AgentA": (128000, "utf-8"),
        "AgentB": (128000, "utf-8"),
        "AgentC": (128000, "utf-8"),
    }
    
    # Create random probability distributions for the 3 agents
    probs = [
        np.random.dirichlet(np.ones(128000)),
        np.random.dirichlet(np.ones(128000)),
        np.random.dirichlet(np.ones(128000))
    ]
    
    # Restrict to top 16 tokens (the default K)
    K = 16
    t0 = time.perf_counter()
    restrict = [top_k_ids(p, K) for p in probs]
    tree = BytePrefixTree.from_vocabs(mock_vocabs, restrict=restrict, max_depth=8)
    t1 = time.perf_counter()
    
    elapsed_ms = (t1 - t0) * 1000
    print(f"Top-K Union Tree Build (K={K}): {elapsed_ms:.2f} ms")
    print(f"Tree Size: {len(tree.children)} nodes")

if __name__ == "__main__":
    benchmark_topk_union_tree_build()
