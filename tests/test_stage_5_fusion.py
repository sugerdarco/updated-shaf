import torch

from pipeline.stage_5_fast_mean_fusion import fast_mean_fusion

def test_fast_mean_fusion_unweighted():
    # Two agents, unweighted (weights default to equal)
    psi_1 = torch.tensor([1.0, 0.0])
    psi_2 = torch.tensor([0.0, 1.0])
    
    # 0.5 * psi_1 + 0.5 * psi_2 = [0.5, 0.5]
    # L2 norm of [0.5, 0.5] is sqrt(0.5^2 + 0.5^2) = sqrt(0.5) = 0.707
    # Normalized: [0.5/0.707, 0.5/0.707] = [0.707, 0.707]
    
    result = fast_mean_fusion([psi_1, psi_2])
    
    expected = torch.tensor([0.70710678, 0.70710678])
    torch.testing.assert_close(result, expected)

def test_fast_mean_fusion_weighted():
    psi_1 = torch.tensor([1.0, 0.0])
    psi_2 = torch.tensor([0.0, 1.0])
    
    # Give agent 1 a huge weight
    weights = [0.99, 0.01]
    
    result = fast_mean_fusion([psi_1, psi_2], weights)
    
    # Result should heavily favor psi_1
    assert result[0] > 0.99
    assert result[1] < 0.15
    # Should be normalized
    assert torch.isclose(torch.norm(result), torch.tensor(1.0))
