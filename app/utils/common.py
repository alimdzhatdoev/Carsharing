"""Shared path helpers."""

from pathlib import Path


def get_project_root() -> Path:
    """Return repository root (parent of `app/`)."""
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """Resolve possibly relative path against project root."""
    p = Path(path)
    if p.is_absolute():
        return p
    base = root or get_project_root()
    return (base / p).resolve()
