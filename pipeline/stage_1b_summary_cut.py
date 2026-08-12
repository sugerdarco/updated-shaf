"""
Stage 1b — Summary Cut
"""
import numpy as np
from dataclasses import dataclass

@dataclass
class AgentSummary:
    agent: int
    ids: np.ndarray
    probs: np.ndarray
    entropy: float
    stop_prob: float
    vocab_size: int
    k_requested: int

def cut_summary(agent_idx: int, p: np.ndarray, stop_prob: float, k: int) -> AgentSummary:
    vocab_size = p.shape[0]
    # entropy exact over full p
    mask = p > 0
    entropy = -float(np.sum(p[mask] * np.log(p[mask])))
    
    # top k
    if k == 0 or k >= vocab_size:
        # full vector
        k_req = 0 if k == 0 else k
        ids = np.flatnonzero(p > 0)
        # return ascending ids
        ids = np.sort(ids)
        return AgentSummary(
            agent=agent_idx,
            ids=ids,
            probs=p[ids],
            entropy=entropy,
            stop_prob=stop_prob,
            vocab_size=vocab_size,
            k_requested=k_req
        )
    
    idx = np.argpartition(-p, k)[:k]
    # filter out zeros
    idx = idx[p[idx] > 0]
    # MUST BE ASCENDING
    ids = np.sort(idx)
    return AgentSummary(
        agent=agent_idx,
        ids=ids,
        probs=p[ids],
        entropy=entropy,
        stop_prob=stop_prob,
        vocab_size=vocab_size,
        k_requested=k
    )
