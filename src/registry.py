"""Generic registry for datasets and models.

Usage:
    from src.registry import MODEL_REGISTRY, DATASET_REGISTRY

    @MODEL_REGISTRY.register("rescnn")
    class ResidualCNN(nn.Module): ...

    model = MODEL_REGISTRY.build("rescnn", n=100, n_blocks=21)
"""
from __future__ import annotations

from typing import Any, Dict


class Registry:
    """A generic registry that maps string names to classes.

    Classes are registered via the ``register`` decorator and
    instantiated via ``build``.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: Dict[str, type] = {}

    def register(self, name: str):
        """Decorator that registers a class under *name*.

        Example::

            @REGISTRY.register("foo")
            class Foo: ...
        """
        def decorator(cls):
            if name in self._registry:
                raise ValueError(
                    f"{self._name} registry already contains {name!r} "
                    f"(registered as {self._registry[name].__name__})"
                )
            self._registry[name] = cls
            return cls
        return decorator

    def get(self, name: str) -> type:
        """Return the class registered under *name*."""
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"{self._name} registry has no entry {name!r}. "
                f"Available: [{available}]"
            )
        return self._registry[name]

    def build(self, name: str, **kwargs) -> Any:
        """Look up class by *name* and instantiate with *kwargs*."""
        cls = self.get(name)
        return cls(**kwargs)

    def keys(self):
        return self._registry.keys()

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __repr__(self) -> str:
        entries = ", ".join(sorted(self._registry.keys()))
        return f"Registry({self._name!r}, [{entries}])"


DATASET_REGISTRY = Registry("dataset")
MODEL_REGISTRY = Registry("model")
