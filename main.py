"""Example pipeline for converter transformer internal PD data processing."""
from __future__ import annotations

from pathlib import Path

from data_loader import load_samples_from_folder
from feature_extraction import extract_features_multichannel
from feature_matrix import build_feature_matrix, save_feature_matrix, standardize_feature_matrix
from preprocess import FilterConfig, preprocess_pipeline
from visualization import (
    plot_feature_correlation,
    plot_feature_distribution,
    plot_signal,
    plot_spectrum,
    plot_wavelet_energy_bar,
)


def run_demo(
    data_dir: str = "./data",
    output_dir: str = "./outputs",
    fs: float = 1_000_000.0,
) -> None:
    """Run end-to-end workflow with recommended defaults.

    Default values are suitable for many PD pulse signals (single sample per file).
    You can adjust fs/filter/wavelet settings based on sensor bandwidth and experiment setup.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Load data (single/multiple formats mixed in one folder)
    samples = load_samples_from_folder(data_dir, fs=fs, recursive=False)

    # 2) Preprocess each sample
    filter_cfg = FilterConfig(filter_type="bandpass", order=4, lowcut=10_000, highcut=300_000)
    for s in samples:
        s.signal = preprocess_pipeline(
            s.signal,
            fs=s.fs,
            remove_dc_flag=True,
            normalize_method="zscore",
            do_outlier_filter=False,
            do_wavelet_denoise=True,
            filter_cfg=filter_cfg,
            do_smooth=False,
        )

    # 3) Build feature matrix
    feature_df = build_feature_matrix(samples, include_label=True)
    feature_df_std = standardize_feature_matrix(feature_df)

    # 4) Save results
    save_feature_matrix(feature_df, out / "pd_features_raw.csv")
    save_feature_matrix(feature_df_std, out / "pd_features_standardized.csv")

    # 5) Quick quality-check visualization on first sample
    s0 = samples[0]
    sig0 = s0.signal[:, 0] if getattr(s0.signal, "ndim", 1) == 2 else s0.signal

    plot_signal(sig0, fs=s0.fs, title=f"Sample {s0.sample_id} - Preprocessed Signal", output_path=str(out / "signal.png"))
    plot_spectrum(sig0, fs=s0.fs, title=f"Sample {s0.sample_id} - Spectrum", output_path=str(out / "spectrum.png"))

    # wavelet energy bar demo (from single-sample extracted feature subset)
    feats0 = extract_features_multichannel(sig0, fs=s0.fs)
    plot_wavelet_energy_bar(feats0, title="Wavelet/WPT Energy Features", output_path=str(out / "wavelet_energy.png"))

    plot_feature_correlation(feature_df_std, output_path=str(out / "feature_corr.png"))

    # Pick one robust impulse-sensitive feature for distribution display
    feature_name = "td_crest_factor" if "td_crest_factor" in feature_df_std.columns else feature_df_std.select_dtypes("number").columns[0]
    plot_feature_distribution(feature_df_std, feature_name=feature_name, output_path=str(out / "feature_dist.png"))

    print("[INFO] Pipeline completed.")
    print(f"[INFO] Raw feature matrix saved to: {out / 'pd_features_raw.csv'}")
    print(f"[INFO] Standardized feature matrix saved to: {out / 'pd_features_standardized.csv'}")


if __name__ == "__main__":
    # TODO: Replace with actual sampling frequency in your setup (Hz).
    run_demo(data_dir="./data", output_dir="./outputs", fs=1_000_000.0)
