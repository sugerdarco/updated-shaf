"""
Stage 8 / part 5 -- end-to-end pipeline (Experimental Stage-by-Stage Architecture).
Execution order: 0 → 1 → 1b → [8 build] → 2 → {3 | 4} → [8: per node 5→6→7, glue] → [8 emit] → back to 0.
"""

from __future__ import annotations

import codecs
import time
from dataclasses import dataclass, field
import numpy as np

from prefix_tree_build.prefix_tree import BytePrefixTree, TreeStats
from .stage_8_sheaf_reconciliation import GlobalSection, SheafReconciler, byte_level_sections
from .stage_0_agent_ensemble import Agent
from .stage_1b_summary_cut import cut_summary, AgentSummary
from .stage_2_divergence_gate import GateThresholds, byte_space_gate

@dataclass
class Stage8Result:
    context: str
    section: GlobalSection
    consensus: list[tuple[bytes, float]]
    coverage: list[float]
    unreachable: dict[str, float]
    mass_ratio: dict[str, float]
    tree_stats: TreeStats
    n_fused_nodes: int
    n_escalated: int
    pruned_mass: float
    elapsed_ms: float
    token_probs: dict[str, np.ndarray] = field(default_factory=dict)
    stop_mass: dict[str, float] = field(default_factory=dict)
    path_taken: str = "A"

    @property
    def should_stop(self) -> bool:
        if not self.stop_mass:
            return False
        votes = sum(1 for v in self.stop_mass.values() if v > 0.5)
        return votes * 2 > len(self.stop_mass)

    @property
    def consensus_bytes(self) -> bytes:
        return self.consensus[0][0] if self.consensus else b""

    def summary(self) -> str:
        top = ", ".join(f"{b!r}:{p:.3f}" for b, p in self.consensus[:5])
        res = ", ".join(f"{k}={v:.2e}" for k, v in self.unreachable.items())
        return (
            f"Path {self.path_taken} | nodes={self.tree_stats.n_nodes} fused={self.n_fused_nodes} "
            f"escalated={self.n_escalated} pruned={self.pruned_mass:.2e} "
            f"{self.elapsed_ms:.1f}ms\n  consensus: {top}\n  unreachable: {res}"
        )


class Stage8Pipeline:
    def __init__(
        self,
        agents: list[Agent],
        *,
        mode: str = "topk_union",
        tree: BytePrefixTree | None = None,
        k: int = 16,
        max_depth: int | None = None,
        fusion: str = "auto",
        eps: float = 1e-6,
        weights: np.ndarray | None = None,
        project_back: bool = True,
        min_support: int | None = None,
        k_escalate: int | None = None,
        theta_D: float = 0.05
    ) -> None:
        self.agents = agents
        self.mode = mode
        self.k = k
        self.k_escalate = k_escalate
        self.max_depth = max_depth
        self.weights = weights
        self.project_back = project_back
        self.min_support = min_support if min_support is not None else max(1, (len(agents) + 1) // 2)
        self.vocabs = [a.vocab_spec() for a in agents]
        self.gate_thresholds = GateThresholds(divergence=theta_D)
        
        # We instantiate a reconciler. In path A it uses mean, in path B it uses auto
        self.reconciler = SheafReconciler(eps=eps, fusion=fusion, max_depth=max_depth)
        self.path_a_reconciler = SheafReconciler(eps=eps, fusion="mean", max_depth=max_depth)

    def step(
        self, context: str, *, beam: int = 8, max_len: int = 16, min_support: int | None = None
    ) -> Stage8Result:
        t0 = time.perf_counter()
        ms = self.min_support if min_support is None else min_support

        # Stages 0 & 1: Agents generate full probability vectors locally.
        probs = [a.next_token_probs(context) for a in self.agents]
        stops = [a.stop_probability(context) if hasattr(a, "stop_probability") else 0.0 for a in self.agents]

        # Stage 1b: Agents cut summary at width k to cross the boundary.
        summaries = [cut_summary(i, probs[i], stops[i], self.k) for i in range(len(self.agents))]
        
        # --- BOUNDARY CROSSED ---

        # Stage 8 (Build Tree): Orchestrator builds shared space from summaries.
        restrict = [s.ids for s in summaries]
        tree = BytePrefixTree.from_vocabs(self.vocabs, restrict=restrict, max_depth=self.max_depth)

        # Precompute covers/terms on the summary ids
        covers, terms = [], []
        for i in range(len(self.agents)):
            c, t = tree.cover_mass(i, probs[i], restrict=restrict[i])
            covers.append(c)
            terms.append(t)

        # Stage 2 (Divergence Gate): Evaluate over the tree.
        root_secs, max_spread, stalk_cache = byte_level_sections(tree, covers, terms)
        gate = byte_space_gate(max_spread, self.gate_thresholds)

        if gate.agree:
            # Stage 3 (Path A: Fast Passthrough)
            path = "A"
            section = self.path_a_reconciler.reconcile(
                tree, probs, weights=self.weights, mass=(covers, terms), stalks=stalk_cache
            )
        else:
            # Stage 4 (Path B: Allgather)
            path = "B"
            if self.k_escalate is not None:
                if self.k_escalate > 0 and self.k_escalate < self.k:
                    raise ValueError("k_escalate must be >= k or 0 (for full vector)")
                # Escalation request: Agent sends wider summary.
                wide_summaries = [cut_summary(i, probs[i], stops[i], self.k_escalate) for i in range(len(self.agents))]
                wide_restrict = [s.ids for s in wide_summaries]
                # Rebuild tree with wider ids
                tree = BytePrefixTree.from_vocabs(self.vocabs, restrict=wide_restrict, max_depth=self.max_depth)
                covers, terms = [], []
                for i in range(len(self.agents)):
                    c, t = tree.cover_mass(i, probs[i], restrict=wide_restrict[i])
                    covers.append(c)
                    terms.append(t)
                stalk_cache = None  # invalidated by new tree

            # Stage 8: Reconcile per node (Stages 5 -> 6 -> 7)
            section = self.reconciler.reconcile(
                tree, probs, weights=self.weights, mass=(covers, terms), stalks=stalk_cache
            )

        # Stage 8 (Emit bytes)
        consensus = section.decode(beam=beam, max_len=max_len, min_support=ms)
        coverage = [float(section.agent_cover[a][0]) for a in range(len(self.agents))]

        unreachable, mass_ratio, token_probs = {}, {}, {}
        if self.project_back:
            for a, agent in enumerate(self.agents):
                q, rep = section.project_to_vocab(a)
                token_probs[agent.name] = q
                unreachable[agent.name] = rep.unreachable
                mass_ratio[agent.name] = rep.mass_ratio

        stop_mass = {ag.name: stops[i] for i, ag in enumerate(self.agents)}

        return Stage8Result(
            context=context,
            section=section,
            consensus=consensus,
            coverage=coverage,
            unreachable=unreachable,
            mass_ratio=mass_ratio,
            stop_mass=stop_mass,
            tree_stats=tree.stats(),
            n_fused_nodes=len(section.conditionals),
            n_escalated=sum(1 for r in section.reports.values() if r.escalated),
            pruned_mass=section.pruned_mass,
            elapsed_ms=(time.perf_counter() - t0) * 1e3,
            token_probs=token_probs,
            path_taken=path
        )

    def generate(
        self, prompt: str, *, max_steps: int = 16, stop: bytes | None = None
    ) -> tuple[str, list[Stage8Result]]:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        text = decoder.decode(prompt.encode("utf-8"))
        trace: list[Stage8Result] = []
        emitted_bytes = bytearray()

        for _ in range(max_steps):
            res = self.step(text)
            trace.append(res)
            if res.should_stop:
                break
            chunk = res.consensus_bytes
            if not chunk:
                break
            emitted_bytes.extend(chunk)
            text += decoder.decode(chunk)
            if stop and stop in bytes(emitted_bytes):
                break

        return text, trace
