"""Agreement-Gated Expert Delegation (AGED) — a novel cross-tokenizer decoder.

Inspired by DeePEn's cross-model collaboration but distinct from it: DeePEn (and
the base SAHF pipeline) AVERAGE the models' next-step distributions in a shared
space. On heterogeneous tokenizers that averaging is exactly what manufactures
"byte-bloat" — a blended distribution whose argmax is a byte neither model meant,
so words come out misspelled.

AGED keeps SAHF's shared byte-prefix tree, but uses it only as an *agreement
sensor* (the Stage-2 byte-space divergence gate). The routing rule:

    agents AGREE on the next bytes  -> emit the sheaf consensus (safe; no bloat)
    agents DIVERGE                  -> DELEGATE the step to the single most
                                       confident agent, which emits its own next
                                       token verbatim (clean spelling)

Confidence is the peak mass of an agent's own next-token distribution. Because
leadership is re-decided every step, different specialists lead different spans,
so the ensemble can beat any single model while never averaging a misspelling.
"""
from __future__ import annotations

import time

import numpy as np

from runners.sheaf_orchestrator import SheafOrchestrator


class DelegatingOrchestrator(SheafOrchestrator):
    def __init__(self, *args, delegate_on_agree: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # delegate_on_agree=True  -> pure confidence-routed MoE (always delegate)
        # delegate_on_agree=False -> consensus when agents agree, delegate when not
        self.delegate_on_agree = delegate_on_agree

    def _delegate_chunk(self, probs):
        """Bytes of the most-confident agent's argmax token (or None)."""
        conf = [float(p.max()) for p in probs]
        a = int(np.argmax(conf))
        tid = int(np.argmax(probs[a]))
        tb = self.agents[a].vocab_spec().token_bytes[tid]
        if not tb:
            return None, a, conf[a]
        return bytes(tb), a, conf[a]

    def step(self, context: str, step_idx: int) -> tuple:
        t0 = time.perf_counter()
        probs = [agent.next_token_probs(context) for agent in self.agents]
        decision, tree, mass, stalks, candidates = self._byte_level_gate(probs)
        record = {"step": step_idx, "entropy": decision.entropy, "divergence": decision.divergence}

        pipe = self._path_a if decision.agree else self._path_b
        res = pipe.step_from_probs(
            probs, context=context, tree=tree, mass=mass, stalks=stalks, candidates=candidates
        )
        chunk = res.consensus_bytes
        path = "A_fast_passthrough" if decision.agree else "B_fusion_pipeline"

        if (not decision.agree) or self.delegate_on_agree:
            deleg, a, conf = self._delegate_chunk(probs)
            if deleg:
                chunk, path = deleg, "C_delegate"
                record["leader"] = self.agents[a].name
                record["leader_conf"] = conf

        record.update({
            "path": path,
            "escalated": res.n_escalated > 0,
            "consensus_bytes": chunk.decode("utf-8", errors="replace"),
            "n_bytes": len(chunk),
            "stop_mass": res.stop_mass,
            "elapsed_ms": (time.perf_counter() - t0) * 1e3,
        })
        if self.logger:
            self.logger.log_step(record)
        return chunk, record, res
