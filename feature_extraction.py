"""Feature extraction for partial discharge signal analysis.

Includes time-domain, frequency-domain and time-frequency domain features.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pywt
from scipy import signal, stats


def _safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a / (b + eps))


def _to_1d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    raise ValueError("Feature extraction expects 1D signal per call. For multi-channel, iterate channel-wise.")


def time_domain_features(x: np.ndarray) -> Dict[str, float]:
    """Extract classic impulse-sensitive time-domain indicators for PD pulses."""
    s = _to_1d(x)
    abs_s = np.abs(s)
    peak = np.max(abs_s)
    mean = np.mean(s)
    var = np.var(s)
    std = np.std(s)
    rms = np.sqrt(np.mean(s**2))
    p2p = np.ptp(s)
    skew = stats.skew(s, bias=False)
    kurt = stats.kurtosis(s, fisher=False, bias=False)
    mean_abs = np.mean(abs_s)
    sqrt_abs_mean = np.mean(np.sqrt(abs_s))

    return {
        "td_peak": float(peak),
        "td_mean": float(mean),
        "td_var": float(var),
        "td_std": float(std),
        "td_rms": float(rms),
        "td_p2p": float(p2p),
        "td_skew": float(skew),
        "td_kurt": float(kurt),
        "td_impulse_factor": _safe_div(peak, mean_abs),
        "td_margin_factor": _safe_div(peak, sqrt_abs_mean**2),
        "td_shape_factor": _safe_div(rms, mean_abs),
        "td_crest_factor": _safe_div(peak, rms),
        "td_energy": float(np.sum(s**2)),
        "td_zero_cross_rate": float(np.mean(np.diff(np.signbit(s)) != 0)),
    }


def frequency_domain_features(
    x: np.ndarray,
    fs: float,
    band_edges: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Frequency-domain features using one-sided FFT spectrum."""
    s = _to_1d(x)
    n = len(s)
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.fft.rfft(s)
    mag = np.abs(spec)
    power = (mag**2) / n

    total_power = np.sum(power) + 1e-12
    dom_idx = int(np.argmax(mag))
    dom_freq = freq[dom_idx]
    centroid = np.sum(freq * power) / total_power
    peak_mag = np.max(mag)
    spectral_rms = np.sqrt(np.mean(power))

    p_norm = power / total_power
    entropy = -np.sum(p_norm * np.log2(p_norm + 1e-12))

    feats: Dict[str, float] = {
        "fd_main_freq": float(dom_freq),
        "fd_centroid": float(centroid),
        "fd_energy": float(total_power),
        "fd_entropy": float(entropy),
        "fd_peak_mag": float(peak_mag),
        "fd_rms": float(spectral_rms),
        "fd_bandwidth": float(np.sqrt(np.sum(((freq - centroid) ** 2) * p_norm))),
    }

    if band_edges is None:
        # Recommended defaults for fs around 1 MHz; adjust to actual Nyquist.
        nyq = fs / 2
        band_edges = [0, 20e3, 60e3, 120e3, 200e3, min(350e3, nyq)]

    edges = np.asarray(band_edges, dtype=float)
    edges = edges[(edges >= 0) & (edges <= fs / 2)]
    edges = np.unique(edges)
    if len(edges) >= 2:
        for i in range(len(edges) - 1):
            f0, f1 = edges[i], edges[i + 1]
            idx = (freq >= f0) & (freq < f1 if i < len(edges) - 2 else freq <= f1)
            band_pow = np.sum(power[idx])
            feats[f"fd_band_energy_{int(f0)}_{int(f1)}"] = float(band_pow)
            feats[f"fd_band_ratio_{int(f0)}_{int(f1)}"] = float(band_pow / total_power)

    return feats


def wavelet_features(x: np.ndarray, wavelet: str = "db4", level: int = 4) -> Dict[str, float]:
    """DWT energy and entropy features."""
    s = _to_1d(x)
    coeffs = pywt.wavedec(s, wavelet=wavelet, level=level)
    energies = np.array([np.sum(c**2) for c in coeffs], dtype=float)
    total = np.sum(energies) + 1e-12
    ratios = energies / total

    feats = {f"wvt_energy_L{i}": float(e) for i, e in enumerate(energies)}
    feats.update({f"wvt_ratio_L{i}": float(r) for i, r in enumerate(ratios)})
    feats["wvt_entropy"] = float(-np.sum(ratios * np.log2(ratios + 1e-12)))
    return feats


def wavelet_packet_energy_features(
    x: np.ndarray,
    wavelet: str = "db4",
    maxlevel: int = 3,
) -> Dict[str, float]:
    """Wavelet packet terminal-node energy ratios."""
    s = _to_1d(x)
    wp = pywt.WaveletPacket(data=s, wavelet=wavelet, mode="symmetric", maxlevel=maxlevel)
    nodes = wp.get_level(maxlevel, order="freq")
    energies = np.array([np.sum(np.square(n.data)) for n in nodes], dtype=float)
    total = np.sum(energies) + 1e-12
    feats = {}
    for i, en in enumerate(energies):
        feats[f"wpt_energy_n{i}"] = float(en)
        feats[f"wpt_ratio_n{i}"] = float(en / total)
    return feats


def stft_features(
    x: np.ndarray,
    fs: float,
    nperseg: int = 256,
    noverlap: int = 128,
) -> Dict[str, float]:
    """STFT-based compact descriptors."""
    s = _to_1d(x)
    f, t, zxx = signal.stft(s, fs=fs, nperseg=nperseg, noverlap=noverlap)
    sxx = np.abs(zxx) ** 2
    tf_energy = np.sum(sxx)

    mean_freq_track = np.sum(f[:, None] * sxx, axis=0) / (np.sum(sxx, axis=0) + 1e-12)
    dom_freq_track = f[np.argmax(sxx, axis=0)]

    return {
        "tf_stft_energy": float(tf_energy),
        "tf_stft_mean_freq_mean": float(np.mean(mean_freq_track)),
        "tf_stft_mean_freq_std": float(np.std(mean_freq_track)),
        "tf_stft_dom_freq_mean": float(np.mean(dom_freq_track)),
        "tf_stft_dom_freq_std": float(np.std(dom_freq_track)),
        "tf_stft_entropy": float(
            -np.sum((sxx / (np.sum(sxx) + 1e-12)) * np.log2(sxx / (np.sum(sxx) + 1e-12) + 1e-12))
        ),
    }


def extract_all_features(
    x: np.ndarray,
    fs: float,
    band_edges: Optional[Sequence[float]] = None,
    wavelet: str = "db4",
    wavelet_level: int = 4,
    wpt_level: int = 3,
) -> Dict[str, float]:
    """Extract and merge all default feature groups for one signal."""
    feats = {}
    feats.update(time_domain_features(x))
    feats.update(frequency_domain_features(x, fs=fs, band_edges=band_edges))
    feats.update(wavelet_features(x, wavelet=wavelet, level=wavelet_level))
    feats.update(wavelet_packet_energy_features(x, wavelet=wavelet, maxlevel=wpt_level))
    feats.update(stft_features(x, fs=fs))
    return feats


def extract_features_multichannel(
    x: np.ndarray,
    fs: float,
    channel_names: Optional[Sequence[str]] = None,
    **kwargs: object,
) -> Dict[str, float]:
    """Extract features from single/multi-channel signal and prefix per channel."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return extract_all_features(arr, fs=fs, **kwargs)

    if arr.ndim != 2:
        raise ValueError(f"Expected 1D/2D signal, got shape={arr.shape}")

    n_channels = arr.shape[1]
    names = list(channel_names) if channel_names and len(channel_names) == n_channels else [f"ch{i}" for i in range(n_channels)]

    merged: Dict[str, float] = {}
    for i in range(n_channels):
        ch_feats = extract_all_features(arr[:, i], fs=fs, **kwargs)
        merged.update({f"{names[i]}_{k}": v for k, v in ch_feats.items()})
    return merged
