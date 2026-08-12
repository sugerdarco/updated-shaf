import torch

from pipeline.stage_2_divergence_gate import GateThresholds, divergence_gate

def test_divergence_gate_agree():
    # Two identical agents (perfect agreement)
    psi_1 = torch.tensor([0.8, 0.6]) # probabilities: [0.64, 0.36]
    psi_2 = torch.tensor([0.8, 0.6])
    
    thresholds = GateThresholds(entropy=1.5, divergence=0.1)
    decision = divergence_gate([psi_1, psi_2], thresholds)
    
    assert decision.agree is True
    assert decision.divergence == 0.0
    # Entropy of [0.64, 0.36] is ~0.65
    assert decision.entropy < 1.5

def test_divergence_gate_disagree():
    # Two completely misaligned agents
    psi_1 = torch.tensor([1.0, 0.0]) # predicts token A
    psi_2 = torch.tensor([0.0, 1.0]) # predicts token B
    
    thresholds = GateThresholds(entropy=2.0, divergence=0.5)
    decision = divergence_gate([psi_1, psi_2], thresholds)
    
    # Distance between [1,0] and [0,1] is sqrt(2) = 1.414, which > 0.5
    assert decision.agree is False
    assert decision.divergence > 1.4

def test_divergence_gate_low_confidence():
    # Agents agree, but they are uniformly unsure
    psi_1 = torch.tensor([0.5, 0.5, 0.5, 0.5]) # p = [0.25]*4, entropy = 1.38
    psi_2 = torch.tensor([0.5, 0.5, 0.5, 0.5])
    
    # Set a very strict entropy threshold
    thresholds = GateThresholds(entropy=1.0, divergence=0.5)
    decision = divergence_gate([psi_1, psi_2], thresholds)
    
    # divergence is 0, but entropy > 1.0, so they fail the gate
    assert decision.agree is False
