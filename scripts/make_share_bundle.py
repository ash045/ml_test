# scripts/make_share_bundle.py
import os, zipfile, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo root (../)
OUT  = ROOT / "share_bundle.zip"

INCLUDE_DIRS = ["src", "scripts", "configs"]
INCLUDE_FILES = ["requirements.txt", "pyproject.toml", "README.md"]

EXCLUDE_DIRS = {
    ".venv", "env", "__pycache__", ".git", ".ipynb_checkpoints",
    "artifacts", "reports", "data", ".mypy_cache", ".pytest_cache"
}
EXCLUDE_EXTS = {".pyc", ".pyo", ".pyd", ".ipynb"}

def should_skip(path: pathlib.Path) -> bool:
    # Skip excluded dirs anywhere in path
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
    # Skip by extension
    if path.suffix.lower() in EXCLUDE_EXTS:
        return True
    return False

def add_path(z: zipfile.ZipFile, path: pathlib.Path, arcroot: pathlib.Path):
    if should_skip(path):
        return
    if path.is_dir():
        for p in path.rglob("*"):
            if p.is_file() and not should_skip(p):
                z.write(p, p.relative_to(arcroot))
    elif path.is_file():
        z.write(path, path.relative_to(arcroot))

def main():
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for d in INCLUDE_DIRS:
            p = ROOT / d
            if p.exists():
                add_path(z, p, ROOT)
        for f in INCLUDE_FILES:
            p = ROOT / f
            if p.exists():
                add_path(z, p, ROOT)
    print(f"Created {OUT} ({OUT.stat().st_size/1_048_576:.2f} MB)")

if __name__ == "__main__":
    main()
