"""Content-addressed JSON artifact cache used by every pipeline stage."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


CACHE_SCHEMA_VERSION = 1


def cache_key(value: Any) -> str:
    """Hash all relevant, JSON-serializable inputs deterministically."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a file incrementally without loading source media into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VideoCache:
    """Per-video artifact store with atomic JSON writes and keyed reads."""

    def __init__(self, root: Path, video_id: str) -> None:
        safe_video_id = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in video_id
        )
        if not safe_video_id:
            raise ValueError("video_id does not contain a usable cache directory name")
        self.path = root / safe_video_id
        self.path.mkdir(parents=True, exist_ok=True)

    def artifact_path(self, relative_path: str) -> Path:
        """Return an artifact path, refusing paths outside this video's cache."""

        candidate = (self.path / relative_path).resolve()
        root = self.path.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("cache artifact path must stay within the video cache")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def read_json(self, relative_path: str, expected_key: str) -> Any | None:
        """Read a JSON payload only when its content-addressed key matches."""

        path = self.artifact_path(relative_path)
        if not path.is_file():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            envelope.get("schema_version") != CACHE_SCHEMA_VERSION
            or envelope.get("cache_key") != expected_key
            or "payload" not in envelope
        ):
            return None
        return envelope["payload"]

    def write_json(self, relative_path: str, artifact_key: str, payload: Any) -> Path:
        """Atomically persist a keyed JSON cache envelope."""

        path = self.artifact_path(relative_path)
        envelope = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_key": artifact_key,
            "payload": payload,
        }
        data = json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(data)
            temporary_path = Path(temporary.name)
        try:
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink(missing_ok=True)
        return path
