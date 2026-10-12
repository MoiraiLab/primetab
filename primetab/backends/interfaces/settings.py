"""Configuration loading for shared server model resources."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CONFIG_ENVIRONMENT_VARIABLE = "TFM_BACKENDS_CONFIG"
# ``settings.py`` now lives at ``primetab/backends/interfaces/``. parents[2] is the
# package root, so the default config is ``primetab/configs/tfm_backends.json``.
# That file is not bundled: server configs contain machine-local paths. Pass
# ``config_path`` or set ``TFM_BACKENDS_CONFIG``.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "tfm_backends.json"


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    """Resolve an explicit path, an environment override, or the package default."""
    selected = config_path or os.environ.get(CONFIG_ENVIRONMENT_VARIABLE) or DEFAULT_CONFIG_PATH
    path = Path(selected).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"TFM backend config not found: {path}")
    return path


def load_settings(config_path: str | Path | None = None) -> dict[str, Any]:
    path = resolve_config_path(config_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "backends" not in payload:
        raise ValueError(f"TFM backend config has no 'backends' object: {path}")
    return payload


def load_backend_settings(
    name: str, config_path: str | Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return one backend section and the shared network section."""
    payload = load_settings(config_path)
    try:
        backend = payload["backends"][name]
    except KeyError as exc:
        raise ValueError(f"TFM backend {name!r} is missing from the config") from exc
    return backend, payload.get("network", {})
