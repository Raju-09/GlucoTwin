"""Raw data discovery and loading for GlucoTwin.

This module discovers dataset files in a directory and loads them
into typed DataFrames. It does NOT validate or modify the data;
that is the responsibility of `validate.py`.

Design rules:
- Raw data is never modified.
- Provenance metadata is attached to each loaded DataFrame as attrs.
- Unknown file formats raise an informative error rather than silently failing.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# File extensions we know how to load, in priority order.
_SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".parquet": "parquet",
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "csv",  # many research datasets use .txt for tab/comma-delimited files
    ".xml": "xml",
    ".json": "json",
    ".xlsx": "excel",
    ".xls": "excel",
}


def discover_files(data_dir: str | Path) -> list[Path]:
    """Walk *data_dir* recursively and return paths to all supported files.

    Parameters
    ----------
    data_dir:
        Root directory to search.

    Returns
    -------
    list[Path]
        Sorted list of file paths with supported extensions.

    Raises
    ------
    FileNotFoundError
        If *data_dir* does not exist.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {data_dir}")

    found: list[Path] = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in _SUPPORTED_EXTENSIONS:
            found.append(path)
            logger.debug("Discovered file: %s", path)

    if not found:
        logger.warning(
            "No supported files found in %s. "
            "Supported extensions: %s",
            data_dir,
            ", ".join(_SUPPORTED_EXTENSIONS),
        )
    else:
        logger.info("Discovered %d file(s) in %s", len(found), data_dir)

    return found


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _sniff_delimiter(path: Path) -> str:
    """Sniff the delimiter for a CSV/TXT file by reading the first line."""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        first_line = fh.readline()
    for sep in (",", "\t", ";", "|"):
        if sep in first_line:
            return sep
    return ","  # fallback


def load_raw(
    path: str | Path,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load a single data file into a DataFrame with provenance metadata.

    The resulting DataFrame's ``.attrs`` dictionary contains:
    - ``source_path``: absolute path string
    - ``sha256``: file hash (for integrity tracking)
    - ``format``: detected file format

    Parameters
    ----------
    path:
        Path to the file to load.
    **kwargs:
        Extra keyword arguments forwarded to the underlying pandas reader.

    Returns
    -------
    pd.DataFrame
        Loaded data with provenance attrs.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    fmt = _SUPPORTED_EXTENSIONS.get(ext)
    if fmt is None:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported: {list(_SUPPORTED_EXTENSIONS)}"
        )

    logger.info("Loading %s (format=%s)", path.name, fmt)

    # Compute hash before loading so we always store the hash of the original.
    sha = _sha256(path)

    if fmt == "parquet":
        df = pd.read_parquet(path, **kwargs)
    elif fmt in ("csv", "tsv"):
        sep = kwargs.pop("sep", None) or kwargs.pop("delimiter", None)
        if sep is None:
            sep = "\t" if fmt == "tsv" else _sniff_delimiter(path)
        df = pd.read_csv(path, sep=sep, low_memory=False, **kwargs)
    elif fmt == "xml":
        df = pd.read_xml(path, **kwargs)
    elif fmt == "json":
        df = pd.read_json(path, **kwargs)
    elif fmt == "excel":
        df = pd.read_excel(path, **kwargs)
    else:  # pragma: no cover — guarded above
        raise ValueError(f"Unhandled format: {fmt}")

    df.attrs["source_path"] = str(path.resolve())
    df.attrs["sha256"] = sha
    df.attrs["format"] = fmt

    logger.info(
        "Loaded %s: %d rows x %d cols (sha256=%s...)",
        path.name,
        len(df),
        len(df.columns),
        sha[:8],
    )
    return df


def load_all(
    data_dir: str | Path,
    **kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """Discover and load all supported files in *data_dir*.

    Parameters
    ----------
    data_dir:
        Root directory to search.
    **kwargs:
        Extra keyword arguments forwarded to :func:`load_raw`.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from file stem to loaded DataFrame.
    """
    files = discover_files(data_dir)
    result: dict[str, pd.DataFrame] = {}
    for path in files:
        try:
            df = load_raw(path, **kwargs)
            result[path.stem] = df
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load %s: %s", path.name, exc)
    return result
