import pytest
import torch

from src.models.rescnn import ResidualBlock, ResidualCNN


def test_residual_block_same_channels_preserves_length():
    block = ResidualBlock(in_channels=8, out_channels=8, kernel_size=8)
    block.eval()
    x = torch.randn(4, 8, 100)
    out = block(x)
    assert out.shape == (4, 8, 100)


def test_residual_block_channel_expansion_preserves_length():
    block = ResidualBlock(in_channels=8, out_channels=16, kernel_size=8)
    block.eval()
    x = torch.randn(4, 8, 100)
    out = block(x)
    assert out.shape == (4, 16, 100)


def test_residual_block_small_kernel():
    block = ResidualBlock(in_channels=4, out_channels=4, kernel_size=3)
    block.eval()
    x = torch.randn(2, 4, 50)
    out = block(x)
    assert out.shape == (2, 4, 50)


def test_rescnn_forward_shape(tiny_rescnn, small_n):
    tiny_rescnn.eval()
    x = torch.randn(8, 1, small_n)
    out = tiny_rescnn(x)
    assert out.shape == (8, 1)


def test_rescnn_forward_logits(tiny_rescnn, small_n):
    # Verify forward() returns raw logits (not sigmoid probabilities).
    # Property: sigmoid(logits) != logits (they're not pre-sigmoided).
    tiny_rescnn.eval()
    x = torch.randn(8, 1, small_n)
    logits = tiny_rescnn(x)
    probs = torch.sigmoid(logits)
    # sigmoid(logits) should differ from logits unless logits happen to be ~0.5
    assert not torch.allclose(logits, probs), \
        "forward() appears to return probabilities instead of logits"
    # predict_proba should equal sigmoid(forward())
    proba = tiny_rescnn.predict_proba(x)
    torch.testing.assert_close(proba, probs.squeeze(1))


def test_rescnn_predict_binary(tiny_rescnn, small_n):
    tiny_rescnn.eval()
    x = torch.randn(16, 1, small_n)
    preds = tiny_rescnn.predict(x)
    assert preds.shape == (16,)
    assert set(preds.tolist()).issubset({0, 1})


def test_rescnn_predict_proba_range(tiny_rescnn, small_n):
    tiny_rescnn.eval()
    x = torch.randn(16, 1, small_n)
    probs = tiny_rescnn.predict_proba(x)
    assert probs.shape == (16,)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_rescnn_full_architecture():
    model = ResidualCNN(n=100, n_blocks=21, base_channels=32, kernel_size=8)
    model.eval()
    x = torch.randn(2, 1, 100)
    out = model(x)
    assert out.shape == (2, 1)


def test_rescnn_variable_length():
    # Model should work on sequences longer than training n (length-agnostic via AdaptiveAvgPool)
    model = ResidualCNN(n=100, n_blocks=3, base_channels=4, kernel_size=3)
    model.eval()
    for length in [50, 100, 200]:
        x = torch.randn(2, 1, length)
        out = model(x)
        assert out.shape == (2, 1)
