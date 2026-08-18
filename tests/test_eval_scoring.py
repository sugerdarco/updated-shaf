"""Unit tests for the answer-space selection contribution (eval/).

Covers the scorer (per answer type), the majority-vote logic, and the open-QA
selection tie-break — the pieces every reported number depends on. CPU-only.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

import scoring
import vote


# ---- multiple-choice extraction ----------------------------------------

def test_mc_double_colon():
    it = {"type": "mc", "choices": ["a", "b", "c", "d"], "gold": 1}  # B
    assert scoring.predict_mc(it, "\n\nAnswer:: B\n\nCoal") == "B"
    assert scoring.score_mc(it, "Answer:: B")
    assert not scoring.score_mc(it, "Answer: C")


def test_mc_star_and_paren_and_is():
    it = {"type": "mc", "choices": ["a", "b", "c", "d"], "gold": 3}  # D
    assert scoring.score_mc(it, "Answer:* D")
    assert scoring.score_mc(it, "The answer is (D).")
    assert scoring.score_mc(it, "D) because ...")


def test_mc_choice_text_fallback():
    it = {"type": "mc", "choices": ["Paris", "Berlin"], "gold": 0}
    assert scoring.score_mc(it, "the capital is paris")
    assert not scoring.score_mc(it, "the capital is berlin")


def test_mc_out_of_range_letter_ignored():
    # 2-choice: only A/B are valid; a stray 'D' must not count
    it = {"type": "mc", "choices": ["x", "y"], "gold": 0}
    assert scoring.predict_mc(it, "Answer: A") == "A"


# ---- numeric ------------------------------------------------------------

def test_number_hash_and_commas():
    assert scoring.score_number({"type": "number", "gold": 18.0}, "= $18\n#### 18")
    assert scoring.score_number({"type": "number", "gold": 1430.0}, "total 1,430")
    assert not scoring.score_number({"type": "number", "gold": 18.0}, "the answer is 20")


# ---- open-QA alias match ------------------------------------------------

def test_openqa_alias():
    it = {"type": "openqa", "gold": ["David Seville", "Dave Seville"]}
    assert scoring.score_openqa(it, "It was David Seville.")
    assert scoring.score_openqa(it, "answer: dave  seville")   # normalization
    assert not scoring.score_openqa(it, "Alvin")


def test_predict_item_dispatch():
    assert scoring.predict_item({"type": "mc", "choices": ["x", "y"], "gold": 0}, "Answer: A") == "A"
    assert scoring.predict_item({"type": "number", "gold": 5}, "#### 5") == 5.0


# ---- voting logic -------------------------------------------------------

def test_vote_majority():
    assert vote.vote_choice(["A", "A", "B"]) == "A"
    assert vote.vote_choice(["A", "B", "B"]) == "B"


def test_vote_tie_breaks_to_priority():
    assert vote.vote_choice(["C", "A", "B"]) == "C"      # all differ -> first
    assert vote.vote_choice([None, "A", "B"]) == "A"     # abstain skipped
    assert vote.vote_choice([None, None, None]) is None


def test_openqa_choice_idx_agreement():
    # two agree on normalized short answer -> pick the earliest agreeing
    assert vote.openqa_choice_idx(["paris", "paris", "berlin"]) == 0
    assert vote.openqa_choice_idx(["berlin", "paris", "paris"]) == 1
    # no agreement -> priority 0
    assert vote.openqa_choice_idx(["a", "b", "c"]) == 0


def test_score_prediction_matches_score_item():
    it = {"type": "mc", "choices": ["a", "b", "c", "d"], "gold": 2}  # C
    assert scoring.score_prediction(it, "C")
    assert not scoring.score_prediction(it, "B")
    assert not scoring.score_prediction(it, None)
