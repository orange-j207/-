"""Feature matrix construction and export."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_loader import SignalSample
from feature_extraction import extract_features_multichannel


def build_feature_matrix(
    samples: Sequence[SignalSample],
    include_label: bool = True,
    channel_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Construct sample-feature matrix from loaded SignalSample list."""
    rows: List[dict] = []
    for s in samples:
        feats = extract_features_multichannel(s.signal, fs=s.fs, channel_names=channel_names)
        feats["sample_id"] = s.sample_id
        if include_label and s.label is not None:
            feats["label"] = s.label
        rows.append(feats)

    df = pd.DataFrame(rows)
    base_cols = [c for c in ["sample_id", "label"] if c in df.columns]
    other_cols = sorted([c for c in df.columns if c not in base_cols])
    return df[base_cols + other_cols]


def standardize_feature_matrix(
    feature_df: pd.DataFrame,
    exclude_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Standardize numeric feature columns and keep identifiers untouched."""
    exclude = set(exclude_columns or ["sample_id", "label"])
    numeric_cols = [c for c in feature_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(feature_df[c])]

    scaled_df = feature_df.copy()
    if numeric_cols:
        scaler = StandardScaler()
        scaled_df[numeric_cols] = scaler.fit_transform(feature_df[numeric_cols])
    return scaled_df


def save_feature_matrix(feature_df: pd.DataFrame, output_path: str | Path) -> None:
    """Save feature matrix to CSV or Excel based on extension."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    ext = p.suffix.lower()
    if ext == ".csv":
        feature_df.to_csv(p, index=False)
    elif ext in {".xlsx", ".xls"}:
        feature_df.to_excel(p, index=False)
    else:
        raise ValueError("output_path must end with .csv/.xlsx/.xls")
