"""
Lightweight file‑system based artifact registry.

The registry is used by the realtime inference engine to resolve paths
to trained models and calibrators. It stores a simple JSON index of
artifacts keyed by name along with their SHA‑256 hash and optional
metadata. External model files can be registered once and later
referenced in the ensemble configuration by name.

If a lookup key begins with ``builtin:``, it is treated specially and
the registry returns a dict containing a ``builtin`` entry. This
allows builtin models (such as ``ema_crossover``) to be referenced
without hitting the file system.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union


class ArtifactRegistry:
    """A simple on‑disk registry for model artifacts.

    Parameters
    ----------
    root : str or Path
        Directory under which the registry index and artifact files
        reside. The index file is stored at ``<root>/index.json``. The
        directory is created if it does not already exist.
    """

    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        # initialise index file if missing
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps({"artifacts": {}}), encoding="utf-8")

    def _load_index(self) -> Dict[str, Any]:
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _save_index(self, idx: Dict[str, Any]) -> None:
        self.index_path.write_text(json.dumps(idx, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _sha256(path: Path) -> str:
        """Compute the SHA‑256 hash of a file lazily."""
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def register(self, name: str, file_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Register a new artifact.

        Saves the SHA of the file at ``file_path`` under the given name in the
        registry index. The actual file is not copied, so it is the caller's
        responsibility to manage the file location. Optional metadata can be
        attached to the entry.

        Returns
        -------
        Dict[str, Any]
            The entry stored in the registry, including path, sha256 and metadata.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact file not found: {path}")
        idx = self._load_index()
        sha = self._sha256(path)
        entry = {"path": str(path.resolve()), "sha256": sha, "metadata": metadata or {}}
        idx.setdefault("artifacts", {})[name] = entry
        self._save_index(idx)
        return entry

    def resolve(self, name_or_uri: str) -> Dict[str, Any]:
        """Resolve an artifact reference to a dictionary describing its location.

        If ``name_or_uri`` starts with ``builtin:``, a dict with a single
        ``builtin`` key is returned, signalling to downstream code that a
        builtin model should be constructed. Otherwise the registry index
        is consulted. If the name does not exist but the string points to
        an existing file, the file path is returned directly. A KeyError
        is raised if the artifact cannot be found.
        """
        # allow builtin references through without touching the index
        if name_or_uri.startswith("builtin:"):
            return {"builtin": name_or_uri}
        idx = self._load_index().get("artifacts", {})
        if name_or_uri not in idx:
            p = Path(name_or_uri)
            if p.exists():
                return {"path": str(p.resolve()), "sha256": self._sha256(p), "metadata": {}}
            raise KeyError(f"Artifact not found: {name_or_uri}")
        return idx[name_or_uri]