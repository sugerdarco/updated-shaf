import numpy as np

from prefix_tree_build.vocab import VocabSpec
from prefix_tree_build.prefix_tree import BytePrefixTree
from pipeline.stage_8_sheaf_reconciliation import SheafReconciler, to_amplitude, to_probability, hellinger

def test_amplitude_and_probability_conversion():
    p = np.array([0.25, 0.75])
    psi = to_amplitude(p)
    np.testing.assert_allclose(psi, np.array([0.5, np.sqrt(0.75)]))
    
    p_back = to_probability(psi)
    np.testing.assert_allclose(p_back, p)

def test_hellinger_distance():
    psi1 = np.array([1.0, 0.0])
    psi2 = np.array([0.0, 1.0])
    
    # Distance between orthobasis vectors on sphere is sqrt(2), hellinger = sqrt(2)/sqrt(2) = 1.0
    h_dist = hellinger(psi1, psi2)
    assert np.isclose(h_dist, 1.0)

def test_sheaf_reconciler_basic():
    # Construct a minimal mock tree with 2 agents using VocabSpec.from_mapping
    vocab1 = VocabSpec.from_mapping({0: b" Paris", 1: b" Berlin"}, name="agent_0")
    vocab2 = VocabSpec.from_mapping({0: b" Paris", 1: b" Rome"}, name="agent_1")
    
    tree = BytePrefixTree.from_vocabs([vocab1, vocab2])
    reconciler = SheafReconciler(fusion="mean")
    
    probs_a = np.array([0.8, 0.2])
    probs_b = np.array([0.7, 0.3])
    
    global_section = reconciler.reconcile(tree, [probs_a, probs_b])
    
    assert global_section is not None
    assert 0 in global_section.conditionals
    
    # Test projection back to agent 0 vocab
    q0, report0 = global_section.project_to_vocab(0)
    assert len(q0) == 2
    assert q0.sum() > 0
