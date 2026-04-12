import math

import pytest
import torch

from src.models.mlp import MLPDetector


def test_mlp_hidden_size_full():
    model = MLPDetector(n=100, variant="full")
    assert model.hidden_size == 198  # 2*100 - 2


def test_mlp_hidden_size_pruned():
    model = MLPDetector(n=100, variant="pruned")
    assert model.hidden_size == 24  # 4 * floor(log2(100)) = 4*6=24


def test_mlp_hidden_size_pruned_n20():
    model = MLPDetector(n=20, variant="pruned")
    # floor(log2(20)) = 4, so 4*4=16
    assert model.hidden_size == 4 * int(math.floor(math.log2(20)))


def test_mlp_forward_shape():
    model = MLPDetector(n=50, variant="pruned")
    model.eval()
    x = torch.randn(8, 50)
    out = model(x)
    assert out.shape == (8, 1)


def test_mlp_forward_is_logits():
    # Output should be unbounded logits (not clipped to [0,1])
    model = MLPDetector(n=50, variant="pruned")
    model.eval()
    x = torch.randn(100, 50) * 10  # large inputs
    logits = model(x)
    # Logits can exceed [0,1]
    assert logits.min().item() < 0 or logits.max().item() > 1


def test_mlp_predict_binary():
    model = MLPDetector(n=50, variant="pruned")
    model.eval()
    x = torch.randn(32, 50)
    preds = model.predict(x)
    assert preds.shape == (32,)
    assert set(preds.tolist()).issubset({0, 1})


def test_mlp_predict_proba_range():
    model = MLPDetector(n=50, variant="pruned")
    model.eval()
    x = torch.randn(32, 50)
    probs = model.predict_proba(x)
    assert probs.shape == (32,)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_mlp_invalid_variant():
    with pytest.raises(ValueError):
        MLPDetector(n=50, variant="unknown")


def test_mlp_count_parameters():
    n = 50
    model = MLPDetector(n=n, variant="pruned")
    h = model.hidden_size
    expected = n * h + h + h * 1 + 1  # W1, b1, W2, b2
    assert model.count_parameters() == expected
