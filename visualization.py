"""Visualization helpers for PD signal and extracted features."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_or_show(output_path: Optional[str] = None) -> None:
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.show()


def plot_signal(signal: np.ndarray, fs: float, title: str = "Signal Waveform", output_path: Optional[str] = None) -> None:
    """Plot time-domain waveform."""
    x = np.asarray(signal).squeeze()
    t = np.arange(len(x)) / fs
    plt.figure(figsize=(10, 3))
    plt.plot(t, x, linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(alpha=0.3)
    _save_or_show(output_path)


def plot_spectrum(signal: np.ndarray, fs: float, title: str = "Spectrum", output_path: Optional[str] = None) -> None:
    """Plot one-sided FFT magnitude spectrum."""
    x = np.asarray(signal).squeeze()
    n = len(x)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(x))

    plt.figure(figsize=(10, 3))
    plt.plot(f, mag, linewidth=1)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.grid(alpha=0.3)
    _save_or_show(output_path)


def plot_wavelet_energy_bar(energy_dict: dict, title: str = "Wavelet Energy", output_path: Optional[str] = None) -> None:
    """Plot wavelet energy bars from feature dict keys containing 'energy'."""
    items = [(k, v) for k, v in energy_dict.items() if "energy" in k]
    if not items:
        raise ValueError("No energy entries found in input dictionary")
    keys, vals = zip(*items)

    plt.figure(figsize=(10, 3))
    plt.bar(range(len(keys)), vals)
    plt.xticks(range(len(keys)), keys, rotation=60, ha="right")
    plt.ylabel("Energy")
    plt.title(title)
    plt.tight_layout()
    _save_or_show(output_path)


def plot_feature_correlation(feature_df: pd.DataFrame, title: str = "Feature Correlation", output_path: Optional[str] = None) -> None:
    """Plot correlation heatmap of numeric features."""
    numeric_df = feature_df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    plt.figure(figsize=(8, 6))
    plt.imshow(corr, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    _save_or_show(output_path)


def plot_feature_distribution(
    feature_df: pd.DataFrame,
    feature_name: str,
    bins: int = 30,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
) -> None:
    """Plot histogram of selected feature."""
    if feature_name not in feature_df.columns:
        raise KeyError(f"Feature not found: {feature_name}")

    vals = pd.to_numeric(feature_df[feature_name], errors="coerce").dropna().to_numpy()
    plt.figure(figsize=(6, 4))
    plt.hist(vals, bins=bins, alpha=0.8)
    plt.xlabel(feature_name)
    plt.ylabel("Count")
    plt.title(title or f"Distribution of {feature_name}")
    plt.grid(alpha=0.3)
    _save_or_show(output_path)
