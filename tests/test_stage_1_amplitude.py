import torch

from pipeline.stage_1_amplitude_interception import softmax_to_amplitude

def test_softmax_to_amplitude_basic():
    logits = torch.tensor([1.0, 2.0, 3.0])
    probs = torch.softmax(logits, dim=-1)
    expected_amp = torch.sqrt(probs)
    
    result = softmax_to_amplitude(logits)
    
    torch.testing.assert_close(result, expected_amp)

def test_softmax_to_amplitude_norm():
    logits = torch.tensor([2.0, 5.0, 1.0])
    result = softmax_to_amplitude(logits)
    
    norm = torch.norm(result, p=2)
    assert torch.isclose(norm, torch.tensor(1.0))
