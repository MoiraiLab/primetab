"""Build fresh estimators without coupling pipelines to server paths."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from primetab.backends.interfaces import create_tfm


class Model(Protocol):
    def fit(self, X: Any, y: Any) -> Any: ...

    def predict(self, X: Any) -> Any: ...


ModelFactory = Callable[[str], Model]


def create_model_factory(spec: Mapping[str, Any]) -> ModelFactory:
    """Create a callable that returns a new model for each requested task.

    ``ridge`` is a lightweight reference backend for tests and debugging. All
    other names are delegated to the shared TFM registry, whose source and
    weight locations come from ``configs/tfm_backends.json``.
    """

    name = str(spec.get("name", "")).strip().lower()
    if not name:
        raise ValueError("backend config must contain a non-empty 'name'")
    common_kwargs = dict(spec.get("kwargs", {}))
    task_kwargs = {key: dict(value) for key, value in spec.get("task_kwargs", {}).items()}

    def factory(task: str) -> Model:
        kwargs = {**common_kwargs, **task_kwargs.get(task, {})}
        if name == "ridge":
            if task == "regression":
                from sklearn.linear_model import Ridge

                return Ridge(**kwargs)
            if task == "classification":
                from sklearn.linear_model import LogisticRegression

                kwargs.setdefault("max_iter", 2_000)
                return LogisticRegression(**kwargs)
            raise ValueError(f"unsupported task: {task!r}")
        return create_tfm(name, task=task, **kwargs)

    return factory
