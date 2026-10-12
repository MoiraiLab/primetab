"""Reusable asset download, extraction, checksum, and audit helpers."""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

import requests
import yaml

CHUNK_SIZE = 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    """Download to a partial file and resume when the server supports ranges."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        if existing and response.status_code == 200:
            existing = 0
        response.raise_for_status()
        mode = "ab" if existing and response.status_code == 206 else "wb"
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    handle.write(chunk)
    partial.replace(destination)


def safe_extract_zip(archive: Path, destination: Path) -> None:
    """Extract a ZIP while rejecting entries that escape the destination."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + ".extracting")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            resolved = (staging / member.filename).resolve()
            if staging.resolve() not in resolved.parents and resolved != staging.resolve():
                raise ValueError(f"Unsafe ZIP path: {member.filename}")
        bundle.extractall(staging)
    children = [child for child in staging.iterdir() if child.name != "__MACOSX"]
    source = children[0] if len(children) == 1 and children[0].is_dir() else staging
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {destination}")
    if source == staging:
        staging.replace(destination)
    else:
        source.replace(destination)
        shutil.rmtree(staging)


def iter_assets(manifest: dict) -> dict[str, dict]:
    combined = dict(manifest.get("assets", {}))
    combined.update(manifest.get("model_assets", {}))
    return combined


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def audit_manifest(manifest_path: Path, repository_root: Path) -> list[dict]:
    records: list[dict] = []
    for asset_id, item in iter_assets(load_manifest(manifest_path)).items():
        path_value = item.get("local_path")
        path = repository_root / path_value if path_value else None
        record = {
            "asset_id": asset_id,
            "declared_status": item.get("status"),
            "path": path_value,
            "present": bool(path and path.exists()),
        }
        if path and path.is_file():
            record.update({"bytes": path.stat().st_size, "sha256": sha256(path)})
        extract_value = item.get("extract_to")
        if extract_value:
            record["extract_to"] = extract_value
            record["extracted"] = (repository_root / extract_value).is_dir()
        records.append(record)
    return records
