"""Preprocessing utilities for partial discharge signals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import signal
import pywt


@dataclass
class FilterConfig:
    """Digital Butterworth filter settings.

    Default recommendation for PD pulse (fs around 1 MHz):
    - bandpass 10 kHz to 300 kHz to suppress low-frequency drift and high-frequency noise.
    """

    filter_type: str = "bandpass"  # 'lowpass' | 'highpass' | 'bandpass'
    order: int = 4
    lowcut: Optional[float] = 1e4
    highcut: Optional[float] = 3e5


def ensure_2d(signal_array: np.ndarray) -> np.ndarray:
    """Convert 1D signal to shape (n_samples, 1), keep 2D unchanged."""
    x = np.asarray(signal_array, dtype=float)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2:
        return x
    raise ValueError(f"Signal must be 1D or 2D, got shape={x.shape}")


def remove_dc(signal_array: np.ndarray) -> np.ndarray:
    """Remove DC component channel-wise by subtracting mean."""
    x = ensure_2d(signal_array)
    return x - np.mean(x, axis=0, keepdims=True)


def normalize_signal(signal_array: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize signal channel-wise.

    Args:
        method: 'zscore', 'minmax', or 'maxabs'.
    """
    x = ensure_2d(signal_array)
    eps = 1e-12
    if method == "zscore":
        mu = np.mean(x, axis=0, keepdims=True)
        sigma = np.std(x, axis=0, keepdims=True)
        return (x - mu) / (sigma + eps)
    if method == "minmax":
        min_v = np.min(x, axis=0, keepdims=True)
        max_v = np.max(x, axis=0, keepdims=True)
        return (x - min_v) / (max_v - min_v + eps)
    if method == "maxabs":
        max_abs = np.max(np.abs(x), axis=0, keepdims=True)
        return x / (max_abs + eps)
    raise ValueError(f"Unknown normalization method: {method}")


def hampel_outlier_filter(signal_array: np.ndarray, window_size: int = 11, n_sigma: float = 3.0) -> np.ndarray:
    """Simple Hampel filter for impulse outlier removal."""
    x = ensure_2d(signal_array).copy()
    n, ch = x.shape
    k = window_size // 2
    for c in range(ch):
        for i in range(k, n - k):
            window = x[i - k : i + k + 1, c]
            med = np.median(window)
            mad = np.median(np.abs(window - med)) + 1e-12
            threshold = n_sigma * 1.4826 * mad
            if abs(x[i, c] - med) > threshold:
                x[i, c] = med
    return x


def smooth_signal(signal_array: np.ndarray, window_length: int = 9, polyorder: int = 2) -> np.ndarray:
    """Apply Savitzky-Golay smoothing channel-wise."""
    x = ensure_2d(signal_array)
    if window_length % 2 == 0:
        window_length += 1
    if window_length <= polyorder:
        window_length = polyorder + 3 if (polyorder + 3) % 2 == 1 else polyorder + 4
    return signal.savgol_filter(x, window_length=window_length, polyorder=polyorder, axis=0)


def wavelet_denoise(signal_array: np.ndarray, wavelet: str = "db4", level: int = 4) -> np.ndarray:
    """Wavelet denoising using soft-thresholding."""
    x = ensure_2d(signal_array)
    out = np.zeros_like(x)
    for c in range(x.shape[1]):
        coeffs = pywt.wavedec(x[:, c], wavelet=wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745 + 1e-12
        uthresh = sigma * np.sqrt(2 * np.log(len(x[:, c])))
        coeffs_d = [coeffs[0]] + [pywt.threshold(cd, value=uthresh, mode="soft") for cd in coeffs[1:]]
        rec = pywt.waverec(coeffs_d, wavelet=wavelet)
        out[:, c] = rec[: x.shape[0]]
    return out


def butter_filter(signal_array: np.ndarray, fs: float, cfg: FilterConfig) -> np.ndarray:
    """Apply Butterworth filter (low/high/band pass)."""
    x = ensure_2d(signal_array)
    nyq = 0.5 * fs

    if cfg.filter_type == "lowpass":
        if cfg.highcut is None:
            raise ValueError("highcut must be provided for lowpass")
        wn = cfg.highcut / nyq
        btype = "low"
    elif cfg.filter_type == "highpass":
        if cfg.lowcut is None:
            raise ValueError("lowcut must be provided for highpass")
        wn = cfg.lowcut / nyq
        btype = "high"
    elif cfg.filter_type == "bandpass":
        if cfg.lowcut is None or cfg.highcut is None:
            raise ValueError("lowcut and highcut must be provided for bandpass")
        wn = [cfg.lowcut / nyq, cfg.highcut / nyq]
        btype = "band"
    else:
        raise ValueError(f"Unsupported filter type: {cfg.filter_type}")

    b, a = signal.butter(cfg.order, wn, btype=btype)
    return signal.filtfilt(b, a, x, axis=0)


def preprocess_pipeline(
    signal_array: np.ndarray,
    fs: float,
    remove_dc_flag: bool = True,
    normalize_method: Optional[str] = "zscore",
    do_outlier_filter: bool = False,
    do_wavelet_denoise: bool = True,
    filter_cfg: Optional[FilterConfig] = None,
    do_smooth: bool = False,
) -> np.ndarray:
    """Configurable preprocessing pipeline for PD signals."""
    x = ensure_2d(signal_array)

    if remove_dc_flag:
        x = remove_dc(x)
    if do_outlier_filter:
        x = hampel_outlier_filter(x)
    if do_wavelet_denoise:
        x = wavelet_denoise(x, wavelet="db4", level=4)
    if filter_cfg is not None:
        x = butter_filter(x, fs=fs, cfg=filter_cfg)
    if do_smooth:
        x = smooth_signal(x)
    if normalize_method:
        x = normalize_signal(x, method=normalize_method)

    return x.squeeze() if np.asarray(signal_array).ndim == 1 else x
