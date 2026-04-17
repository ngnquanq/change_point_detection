"""Tests for the registry pattern."""
from __future__ import annotations

import pytest

from src.registry import Registry, MODEL_REGISTRY, DATASET_REGISTRY


def test_registry_register_and_get():
    reg = Registry("test")

    @reg.register("foo")
    class Foo:
        pass

    assert reg.get("foo") is Foo
    assert "foo" in reg


def test_registry_build():
    reg = Registry("test")

    @reg.register("bar")
    class Bar:
        def __init__(self, x, y=10):
            self.x = x
            self.y = y

    obj = reg.build("bar", x=5, y=20)
    assert isinstance(obj, Bar)
    assert obj.x == 5
    assert obj.y == 20


def test_registry_duplicate_raises():
    reg = Registry("test")

    @reg.register("dup")
    class A:
        pass

    with pytest.raises(ValueError, match="already contains"):
        @reg.register("dup")
        class B:
            pass


def test_registry_missing_raises():
    reg = Registry("test")
    with pytest.raises(KeyError, match="has no entry"):
        reg.get("nonexistent")


def test_registry_keys():
    reg = Registry("test")

    @reg.register("a")
    class A:
        pass

    @reg.register("b")
    class B:
        pass

    assert set(reg.keys()) == {"a", "b"}


def test_registry_repr():
    reg = Registry("myname")

    @reg.register("x")
    class X:
        pass

    assert "myname" in repr(reg)
    assert "x" in repr(reg)


def test_model_registry_has_entries():
    """Models are registered at import time."""
    import src.models.rescnn  # noqa: F401
    import src.models.mlp     # noqa: F401
    assert "rescnn" in MODEL_REGISTRY
    assert "mlp" in MODEL_REGISTRY


def test_dataset_registry_has_entries():
    """Datasets are registered at import time."""
    import src.data.datasets.simulated  # noqa: F401
    import src.data.datasets.hasc       # noqa: F401
    assert "simulated" in DATASET_REGISTRY
    assert "hasc" in DATASET_REGISTRY
