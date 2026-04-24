import torch

from src.models.rescnn import ResidualBlock, ResidualCNN


def build_tiny_rescnn(**overrides) -> ResidualCNN:
    kwargs = {
        "n": 32,
        "n_blocks": 3,
        "base_channels": 4,
        "kernel_size": 3,
        "dropout": 0.0,
    }
    kwargs.update(overrides)
    return ResidualCNN(**kwargs)


def test_residual_block_same_channels_preserves_length() -> None:
    block = ResidualBlock(in_channels=8, out_channels=8, kernel_size=8)
    block.eval()

    x = torch.randn(4, 8, 100)
    out = block(x)

    assert out.shape == (4, 8, 100)


def test_residual_block_channel_expansion_preserves_length() -> None:
    block = ResidualBlock(in_channels=8, out_channels=16, kernel_size=8)
    block.eval()

    x = torch.randn(4, 8, 100)
    out = block(x)

    assert out.shape == (4, 16, 100)


def test_residual_block_small_kernel() -> None:
    block = ResidualBlock(in_channels=4, out_channels=4, kernel_size=3)
    block.eval()

    x = torch.randn(2, 4, 50)
    out = block(x)

    assert out.shape == (2, 4, 50)


def test_rescnn_forward_shape() -> None:
    model = build_tiny_rescnn()
    model.eval()

    x = torch.randn(8, 1, 32)
    out = model(x)

    assert out.shape == (8, 1)


def test_rescnn_forward_returns_logits_not_probabilities() -> None:
    model = build_tiny_rescnn()
    model.eval()

    x = torch.randn(8, 1, 32)
    logits = model(x)
    probs = torch.sigmoid(logits)

    assert not torch.allclose(logits, probs)

    predicted_probs = model.predict_proba(x)
    torch.testing.assert_close(predicted_probs, probs.squeeze(1))


def test_rescnn_predict_binary_output() -> None:
    model = build_tiny_rescnn()
    model.eval()

    x = torch.randn(16, 1, 32)
    preds = model.predict(x)

    assert preds.shape == (16,)
    assert set(preds.tolist()).issubset({0, 1})


def test_rescnn_predict_proba_range() -> None:
    model = build_tiny_rescnn()
    model.eval()

    x = torch.randn(16, 1, 32)
    probs = model.predict_proba(x)

    assert probs.shape == (16,)
    assert (probs >= 0).all()
    assert (probs <= 1).all()


def test_rescnn_full_architecture_forward() -> None:
    model = ResidualCNN(n=100, n_blocks=21, base_channels=32, kernel_size=8, dropout=0.0)
    model.eval()

    x = torch.randn(2, 1, 100)
    out = model(x)

    assert out.shape == (2, 1)


def test_rescnn_variable_length_inputs() -> None:
    model = build_tiny_rescnn()
    model.eval()

    for length in [50, 100, 200]:
        x = torch.randn(2, 1, length)
        out = model(x)
        assert out.shape == (2, 1)


def test_rescnn_multiclass_head_shape() -> None:
    model = ResidualCNN(
        n=100,
        n_blocks=3,
        base_channels=4,
        kernel_size=3,
        dropout=0.0,
        num_classes=30,
    )
    model.eval()

    x = torch.randn(2, 1, 100)
    out = model(x)

    assert out.shape == (2, 30)
