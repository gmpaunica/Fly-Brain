"""Preserve completed and partial experiment records on repeated commands."""

from pathlib import Path


def ensure_new_run_directory(output_dir: Path) -> None:
    """Require an unused or empty destination before any experiment work."""
    path = Path(output_dir)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f'Choose an unused or empty experiment directory: {path}')
