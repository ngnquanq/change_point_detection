import numpy as np
import pytest

from src.data.transforms import (
    minmax_scale,
    trimmed_scale,
    augment_reversed,
    apply_pretransform,
    build_preprocessing_pipeline,
)


def test_minmax_scale_range():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 30)).astype(np.float32)
    X_scaled = minmax_scale(X)
    assert X_scaled.shape == X.shape
    assert X_scaled.min() >= 0.0 - 1e-6
    assert X_scaled.max() <= 1.0 + 1e-6


def test_minmax_scale_constant_sequence():
    X = np.ones((5, 20), dtype=np.float32)
    X_scaled = minmax_scale(X)
    # Constant sequences should not produce NaN
    assert not np.isnan(X_scaled).any()


def test_trimmed_scale_shape():
    rng = np.random.default_rng(1)
    X = rng.standard_cauchy(size=(20, 40)).astype(np.float32)
    X_scaled = trimmed_scale(X, trim_fraction=0.1)
    assert X_scaled.shape == X.shape
    assert not np.isnan(X_scaled).any()


def test_augment_reversed_doubles_n():
    X = np.random.randn(50, 30).astype(np.float32)
    y = np.random.randint(0, 2, size=50).astype(np.int8)
    taus = np.where(y == 1, np.random.randint(1, 29, size=50), 0).astype(np.int64)

    X_aug, y_aug, taus_aug = augment_reversed(X, y, taus)
    assert X_aug.shape == (100, 30)
    assert y_aug.shape == (100,)
    assert taus_aug.shape == (100,)


def test_augment_reversed_sequences_correct():
    X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    y = np.array([1], dtype=np.int8)
    taus = np.array([2], dtype=np.int64)  # 1-indexed tau=2

    X_aug, y_aug, taus_aug = augment_reversed(X, y, taus)
    np.testing.assert_array_equal(X_aug[1], [5.0, 4.0, 3.0, 2.0, 1.0])
    assert y_aug[1] == 1
    # tau_rev = n - tau = 5 - 2 = 3
    assert taus_aug[1] == 3


def test_augment_reversed_no_change_tau_zero():
    X = np.random.randn(10, 20).astype(np.float32)
    y = np.zeros(10, dtype=np.int8)
    taus = np.zeros(10, dtype=np.int64)

    _, _, taus_aug = augment_reversed(X, y, taus)
    assert (taus_aug == 0).all()


def test_apply_pretransform_squared():
    X = np.ones((5, 10), dtype=np.float32) * 2.0
    X_out = apply_pretransform(X, use_squared=True, use_cross_product=False)
    assert X_out.shape == (5, 20)
    # First 10 cols are original, next 10 are x^2=4
    np.testing.assert_allclose(X_out[:, 10:], 4.0)


def test_apply_pretransform_cross_product():
    X = np.array([[1.0, 2.0, 3.0, 4.0]])
    X_out = apply_pretransform(X, use_squared=False, use_cross_product=True)
    assert X_out.shape == (1, 8)
    # Cross-product: [1*2, 2*3, 3*4, 0] = [2, 6, 12, 0]
    np.testing.assert_allclose(X_out[0, 4:], [2.0, 6.0, 12.0, 0.0])


def test_apply_pretransform_none():
    X = np.random.randn(10, 30).astype(np.float32)
    X_out = apply_pretransform(X, use_squared=False, use_cross_product=False)
    np.testing.assert_array_equal(X_out, X)


def test_build_pipeline_s1():
    pipeline = build_preprocessing_pipeline("S1")
    X = np.random.randn(20, 50).astype(np.float32)
    X_out = pipeline(X)
    assert X_out.shape == (20, 50)
    assert X_out.min() >= -1e-6
    assert X_out.max() <= 1.0 + 1e-6


def test_build_pipeline_s3_trimmed():
    pipeline = build_preprocessing_pipeline("S3")
    rng = np.random.default_rng(42)
    X = rng.standard_cauchy(size=(10, 30)).astype(np.float32)
    X_out = pipeline(X)
    assert X_out.shape == (10, 30)
    assert not np.isnan(X_out).any()
