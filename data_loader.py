"""Data loading utilities for converter transformer partial discharge (PD) signals."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class SignalSample:
    """Container for a single PD sample.

    Attributes:
        sample_id: Unique sample identifier (default from file stem).
        signal: Signal array with shape (n_samples,) or (n_samples, n_channels).
        fs: Sampling frequency in Hz.
        label: Optional label used for supervised learning.
        meta: Extra metadata (path, sheet name, channel hints, etc.).
    """

    sample_id: str
    signal: np.ndarray
    fs: float
    label: Optional[Any] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.signal = np.asarray(self.signal, dtype=float)
        if self.signal.ndim not in (1, 2):
            raise ValueError(f"Signal must be 1D or 2D array, got shape={self.signal.shape}")
        if self.fs <= 0:
            raise ValueError("Sampling frequency 'fs' must be positive")


def _select_numeric_columns(df: pd.DataFrame, prefer_columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Return numeric subset of dataframe, with optional preferred columns first."""
    if prefer_columns:
        existing = [c for c in prefer_columns if c in df.columns]
        if existing:
            numeric = df[existing].apply(pd.to_numeric, errors="coerce")
            return numeric.dropna(how="all")

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        converted = df.apply(pd.to_numeric, errors="coerce")
        numeric_df = converted.dropna(axis=1, how="all")
    return numeric_df


def _load_tabular_file(
    file_path: Path,
    fmt: str,
    signal_columns: Optional[Sequence[str]] = None,
    label_column: Optional[str] = None,
    excel_sheet: Optional[str | int] = 0,
) -> tuple[np.ndarray, Optional[Any], Dict[str, Any]]:
    """Load CSV/TXT/Excel and return signal array + optional label."""
    if fmt in {"csv", "txt"}:
        sep = "," if fmt == "csv" else None
        df = pd.read_csv(file_path, sep=sep, engine="python")
    elif fmt in {"xls", "xlsx"}:
        df = pd.read_excel(file_path, sheet_name=excel_sheet)
    else:
        raise ValueError(f"Unsupported tabular format: {fmt}")

    label = None
    if label_column and label_column in df.columns:
        label = df[label_column].iloc[0]

    signal_df = _select_numeric_columns(df, prefer_columns=signal_columns)
    if label_column and label_column in signal_df.columns:
        signal_df = signal_df.drop(columns=[label_column])

    if signal_df.empty:
        raise ValueError(f"No numeric signal columns found in file: {file_path}")

    signal = signal_df.to_numpy()
    if signal.shape[1] == 1:
        signal = signal[:, 0]

    meta = {
        "path": str(file_path),
        "format": fmt,
        "columns": list(signal_df.columns),
        "n_rows": len(signal_df),
    }
    return signal, label, meta


def _load_numpy_file(file_path: Path, key: Optional[str] = None) -> tuple[np.ndarray, Optional[Any], Dict[str, Any]]:
    """Load .npy/.npz signal file."""
    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        signal = np.load(file_path, allow_pickle=True)
        label = None
    elif suffix == ".npz":
        npz = np.load(file_path, allow_pickle=True)
        keys = list(npz.keys())
        if not keys:
            raise ValueError(f"No arrays in NPZ file: {file_path}")
        data_key = key if key in npz else keys[0]
        signal = npz[data_key]
        label = npz["label"] if "label" in npz else None
    else:
        raise ValueError(f"Unsupported numpy file format: {suffix}")

    meta = {"path": str(file_path), "format": suffix.lstrip("."), "array_shape": np.shape(signal)}
    return np.asarray(signal, dtype=float), label, meta


def load_sample(
    file_path: str | Path,
    fs: float,
    signal_columns: Optional[Sequence[str]] = None,
    label_column: Optional[str] = None,
    excel_sheet: Optional[str | int] = 0,
    npz_key: Optional[str] = None,
) -> SignalSample:
    """Load a single PD sample from one file with unified output structure.

    Supported formats: csv/xlsx/xls/txt/npy/npz.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    ext = p.suffix.lower().lstrip(".")
    if ext in {"csv", "txt", "xls", "xlsx"}:
        signal, label, meta = _load_tabular_file(
            p, ext, signal_columns=signal_columns, label_column=label_column, excel_sheet=excel_sheet
        )
    elif ext in {"npy", "npz"}:
        signal, label, meta = _load_numpy_file(p, key=npz_key)
    else:
        raise ValueError(f"Unsupported file extension '{ext}' for: {p}")

    return SignalSample(sample_id=p.stem, signal=signal, fs=fs, label=label, meta=meta)


def load_samples_from_folder(
    folder: str | Path,
    fs: float,
    allowed_exts: Optional[Sequence[str]] = None,
    signal_columns: Optional[Sequence[str]] = None,
    label_column: Optional[str] = None,
    excel_sheet: Optional[str | int] = 0,
    npz_key: Optional[str] = None,
    recursive: bool = False,
) -> List[SignalSample]:
    """Batch-load PD samples in a folder.

    Args:
        folder: Directory containing one-file-per-sample data.
        fs: Sampling frequency (Hz). Default in main is 1e6 for PD pulse capture.
        allowed_exts: Extensions list, default supports csv/txt/xls/xlsx/npy/npz.
        recursive: Whether to recursively search subfolders.
    """
    d = Path(folder)
    if not d.is_dir():
        raise NotADirectoryError(f"Not a directory: {d}")

    allowed = {e.lower().lstrip(".") for e in (allowed_exts or ["csv", "txt", "xls", "xlsx", "npy", "npz"])}
    pattern = "**/*" if recursive else "*"

    files = sorted([p for p in d.glob(pattern) if p.is_file() and p.suffix.lower().lstrip(".") in allowed])
    if not files:
        raise FileNotFoundError(f"No supported sample files found in {d}")

    samples: List[SignalSample] = []
    errors: List[str] = []
    for fp in files:
        try:
            samples.append(
                load_sample(
                    fp,
                    fs=fs,
                    signal_columns=signal_columns,
                    label_column=label_column,
                    excel_sheet=excel_sheet,
                    npz_key=npz_key,
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{fp.name}: {exc}")

    if not samples:
        raise RuntimeError(f"Failed to load all files. Errors: {errors}")

    if errors:
        print("[WARN] Some files were skipped due to parsing errors:")
        for msg in errors:
            print(f"  - {msg}")

    return samples
