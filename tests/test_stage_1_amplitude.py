import numpy as np

from pipeline.stage_1_amplitude_interception import softmax_to_amplitude

def test_softmax_to_amplitude_basic():
    # A standard probability distribution
    p = np.array([0.1, 0.4, 0.5])
    # The amplitude should be the square root of p
    expected = np.sqrt(p)
    
    result, norm = softmax_to_amplitude(p)
    
    np.testing.assert_allclose(result, expected)
    # Norm of a probability distribution should be 1.0 (so sqrt is also 1.0)
    assert np.isclose(norm, 1.0)

def test_softmax_to_amplitude_unnormalized():
    # Logits / unnormalized probabilities
    logits = np.array([1.0, 4.0, 4.0])
    # The function automatically normalizes by the sum
    expected_p = logits / logits.sum()
    expected_amp = np.sqrt(expected_p)
    
    result, norm = softmax_to_amplitude(logits)
    
    np.testing.assert_allclose(result, expected_amp)
    np.testing.assert_allclose(norm, np.sqrt(logits.sum()))

def test_softmax_to_amplitude_zeros():
    # All zeros (e.g., extreme masking)
    logits = np.array([0.0, 0.0, 0.0])
    
    result, norm = softmax_to_amplitude(logits)
    
    # Should safely return zeros without div-by-zero error
    np.testing.assert_allclose(result, np.zeros_like(logits))
    assert norm == 0.0
