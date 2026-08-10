import torch
import pytest

from pipeline.stage_6_and_7_robust_escalation import detect_outliers, weiszfeld_geometric_median

def test_detect_outliers_no_outlier():
    # 3 agents with very close probability distributions
    psi1 = torch.tensor([0.7, 0.7])
    psi2 = torch.tensor([0.71, 0.69])
    psi3 = torch.tensor([0.69, 0.71])
    psi_star = torch.tensor([0.7, 0.7])
    
    outlier_mask, dists = detect_outliers([psi1, psi2, psi3], psi_star, mad_multiplier=3.0)
    
    assert not any(outlier_mask)
    assert len(dists) == 3

def test_detect_outliers_with_adversary():
    # 2 honest agents + 1 poisoned agent
    psi1 = torch.tensor([0.9, 0.1])
    psi2 = torch.tensor([0.89, 0.11])
    psi_poisoned = torch.tensor([0.0, 1.0])
    
    psi_star = (psi1 + psi2 + psi_poisoned) / 3.0
    psi_star = psi_star / torch.norm(psi_star, p=2)
    
    outlier_mask, dists = detect_outliers([psi1, psi2, psi_poisoned], psi_star, mad_multiplier=2.0)
    
    # Agent index 2 should be flagged as outlier
    assert outlier_mask[2] is True
    assert outlier_mask[0] is False
    assert outlier_mask[1] is False

def test_weiszfeld_geometric_median_convergence():
    # 2 honest agents pointing to ~[1, 0], 1 outlier pointing to [0, 1]
    psi1 = torch.tensor([1.0, 0.0])
    psi2 = torch.tensor([0.99, 0.1])
    psi1 = psi1 / torch.norm(psi1)
    psi2 = psi2 / torch.norm(psi2)
    psi_outlier = torch.tensor([0.0, 1.0])
    
    med, iters = weiszfeld_geometric_median([psi1, psi2, psi_outlier], max_iter=50, tol=1e-5)
    
    # Geometric median should be close to honest agents and resist single outlier
    assert med[0] > 0.9
    assert med[1] < 0.3
    assert iters > 0
    assert torch.isclose(torch.norm(med), torch.tensor(1.0), atol=1e-5)
