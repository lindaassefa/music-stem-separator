"""Conventional DSP measurements for stems Demucs has already separated.

Nothing in this module trains or runs a neural network. Spectral features come
from librosa, and the sums and roots come from NumPy.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

from stem_ai.export import STEM_NAMES

# Standard music-analysis window. Shorter stems use the largest power of two
# that fits so the STFT is defined without padding past the clip.
STANDARD_N_FFT = 2048
STANDARD_HOP = 512
# Below this RMS the spectrum has no energy. Centroid and bandwidth are then
# undefined; they are reported as 0 so silence stays numeric.
SILENT_RMS = 1e-8
MAX_WAVEFORM_POINTS = 12_000
MAX_SPECTROGRAM_FRAMES = 1_500


class AnalysisError(ValueError):
    """A measurement or plot could not be computed from the stem samples."""


@dataclass(frozen=True)
class AudioFeatures:
    """Frame-averaged DSP features of one mono waveform."""

    rms: float
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    zero_crossing_rate: float


@dataclass(frozen=True)
class ReconstructionError:
    """How far the sum of the four stems is from the mixture Demucs received."""

    rmse: float
    mixture_rms: float
    relative_rmse: float | None
    peak_absolute_error: float


@dataclass(frozen=True)
class StemMeasurement:
    stem: str
    rms: float
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    zero_crossing_rate: float
    waveform_plot: Path
    spectrogram_plot: Path


def analyze_audio(samples: np.ndarray, sample_rate: int) -> AudioFeatures:
    """Measure one stem. Stereo input is averaged to mono before any feature."""

    if sample_rate <= 0:
        raise AnalysisError("Sample rate must be positive.")
    mono = to_mono(samples)
    n_fft, hop = analysis_window(mono.size)
    rms = float(np.sqrt(np.mean(np.square(mono))))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_fft=.*too large")
        centroid = librosa.feature.spectral_centroid(
            y=mono, sr=sample_rate, n_fft=n_fft, hop_length=hop
        )
        bandwidth = librosa.feature.spectral_bandwidth(
            y=mono, sr=sample_rate, n_fft=n_fft, hop_length=hop
        )
        crossings = librosa.feature.zero_crossing_rate(
            mono, frame_length=n_fft, hop_length=hop
        )
    return AudioFeatures(
        rms=rms,
        spectral_centroid_hz=_finite_mean(centroid, rms=rms, name="Spectral centroid"),
        spectral_bandwidth_hz=_finite_mean(bandwidth, rms=rms, name="Spectral bandwidth"),
        zero_crossing_rate=_finite_mean(crossings, rms=rms, name="Zero-crossing rate"),
    )


def reconstruction_error(
    mixture: np.ndarray,
    stems: dict[str, np.ndarray],
) -> ReconstructionError:
    """Compare a mixture with the sum of vocals, drums, bass, and other.

    Shapes are sample-for-sample. A zero error is not assumed. ``relative_rmse``
    is RMSE divided by the mixture RMS, and is None when the mixture is silent
    because that ratio is undefined.
    """

    reference = np.asarray(mixture, dtype=np.float64)
    if reference.ndim != 2 or reference.size == 0:
        raise AnalysisError("The mixture must have shape (frames, channels).")
    if not np.isfinite(reference).all():
        raise AnalysisError("The mixture contains non-finite samples.")

    total = np.zeros_like(reference)
    for name in STEM_NAMES:
        if name not in stems:
            raise AnalysisError(f"Missing stem: {name}.")
        part = np.asarray(stems[name], dtype=np.float64)
        if part.shape != reference.shape:
            raise AnalysisError(
                f"{name} shape {part.shape} does not match the mixture shape {reference.shape}."
            )
        if not np.isfinite(part).all():
            raise AnalysisError(f"{name} contains non-finite samples.")
        total += part

    residual = total - reference
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    mixture_rms = float(np.sqrt(np.mean(np.square(reference))))
    peak = float(np.max(np.abs(residual)))
    relative = None if mixture_rms < SILENT_RMS else float(rmse / mixture_rms)
    return ReconstructionError(
        rmse=rmse,
        mixture_rms=mixture_rms,
        relative_rmse=relative,
        peak_absolute_error=peak,
    )


def inference_duration_ratio(inference_seconds: float, duration_seconds: float) -> float:
    """Return inference time divided by the uploaded audio duration."""

    if not np.isfinite(inference_seconds) or not np.isfinite(duration_seconds):
        raise AnalysisError("Inference time and duration must be finite.")
    if inference_seconds < 0:
        raise AnalysisError("Inference time cannot be negative.")
    if duration_seconds <= 0:
        raise AnalysisError("Audio duration must be positive.")
    return float(inference_seconds / duration_seconds)


def measure_stems(
    stems: dict[str, np.ndarray],
    sample_rate: int,
    output_dir: Path,
) -> tuple[StemMeasurement, ...]:
    """Compute the four DSP features and write both plots for every stem."""

    output_dir.mkdir(parents=True, exist_ok=True)
    measured: list[StemMeasurement] = []
    for name in STEM_NAMES:
        if name not in stems:
            raise AnalysisError(f"Missing stem: {name}.")
        features = analyze_audio(stems[name], sample_rate)
        waveform_plot, spectrogram_plot = render_stem_figures(
            name, stems[name], sample_rate, output_dir
        )
        measured.append(
            StemMeasurement(
                stem=name,
                rms=features.rms,
                spectral_centroid_hz=features.spectral_centroid_hz,
                spectral_bandwidth_hz=features.spectral_bandwidth_hz,
                zero_crossing_rate=features.zero_crossing_rate,
                waveform_plot=waveform_plot,
                spectrogram_plot=spectrogram_plot,
            )
        )
    return tuple(measured)


def render_stem_figures(
    name: str,
    samples: np.ndarray,
    sample_rate: int,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write a waveform PNG and a log-frequency spectrogram PNG."""

    mono = to_mono(samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    waveform_path = output_dir / f"{name}_waveform.png"
    spectrogram_path = output_dir / f"{name}_spectrogram.png"
    _plot_waveform(name, mono, sample_rate, waveform_path)
    _plot_spectrogram(name, mono, sample_rate, spectrogram_path)
    return waveform_path, spectrogram_path


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Average channels. Input is (frames,) or (frames, channels)."""

    audio = np.asarray(samples, dtype=np.float64)
    if audio.ndim == 1:
        mono = audio
    elif audio.ndim == 2:
        if audio.shape[0] == 0 or audio.shape[1] == 0:
            raise AnalysisError("Audio is empty.")
        mono = np.mean(audio, axis=1)
    else:
        raise AnalysisError("Audio must be shaped (frames,) or (frames, channels).")
    if mono.size == 0:
        raise AnalysisError("Audio is empty.")
    if not np.isfinite(mono).all():
        raise AnalysisError("Audio contains non-finite samples.")
    return np.ascontiguousarray(mono)


def analysis_window(n_samples: int) -> tuple[int, int]:
    """Return (n_fft, hop_length) used for the reported features."""

    if n_samples >= STANDARD_N_FFT:
        return STANDARD_N_FFT, STANDARD_HOP
    if n_samples <= 1:
        return 1, 1
    n_fft = 1 << int(np.floor(np.log2(n_samples)))
    hop = max(1, n_fft // 4)
    return n_fft, hop


def _finite_mean(values: np.ndarray, rms: float, name: str) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        if rms < SILENT_RMS:
            return 0.0
        raise AnalysisError(f"{name} was non-finite.")
    if finite.size != array.size and rms >= SILENT_RMS:
        raise AnalysisError(f"{name} was non-finite.")
    return float(np.mean(finite))


def _plot_waveform(name: str, mono: np.ndarray, sample_rate: int, path: Path) -> None:
    stride = int(np.ceil(mono.size / MAX_WAVEFORM_POINTS)) if mono.size > MAX_WAVEFORM_POINTS else 1
    plotted = mono[::stride]
    times = np.arange(0, mono.size, stride, dtype=np.float64) / sample_rate
    note = (
        "every sample"
        if stride == 1
        else f"1 of every {stride} samples; RMS uses every sample"
    )
    figure, axis = plt.subplots(figsize=(8, 2.8))
    try:
        axis.plot(times, plotted, color="#1f2933", linewidth=0.8)
        axis.set_title(f"{name} waveform ({note})")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Amplitude")
        axis.set_xlim(0, max(times[-1] if times.size else 0, 0))
        figure.tight_layout()
        figure.savefig(path, dpi=100)
    finally:
        plt.close(figure)


def _plot_spectrogram(name: str, mono: np.ndarray, sample_rate: int, path: Path) -> None:
    n_fft, feature_hop = analysis_window(mono.size)
    hop = feature_hop
    frame_count = 1 + max(mono.size - n_fft, 0) // max(hop, 1)
    if frame_count > MAX_SPECTROGRAM_FRAMES:
        hop = max(feature_hop, int(np.ceil(mono.size / MAX_SPECTROGRAM_FRAMES)))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_fft=.*too large")
        spectrum = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=hop))
    peak = float(np.max(spectrum)) if spectrum.size else 0.0
    decibels = librosa.amplitude_to_db(spectrum, ref=peak if peak > 0 else 1.0)
    figure, axis = plt.subplots(figsize=(8, 3.2))
    try:
        image = librosa.display.specshow(
            decibels,
            sr=sample_rate,
            hop_length=hop,
            n_fft=n_fft,
            x_axis="time",
            y_axis="log",
            ax=axis,
            cmap="magma",
        )
        same_hop = hop == feature_hop
        hop_note = (
            f"hop {hop}"
            if same_hop
            else f"display hop {hop}; feature hop {feature_hop}"
        )
        axis.set_title(f"{name} spectrogram, dB vs this image peak ({hop_note})")
        figure.colorbar(image, ax=axis, format="%+2.0f dB")
        figure.tight_layout()
        figure.savefig(path, dpi=100)
    finally:
        plt.close(figure)
