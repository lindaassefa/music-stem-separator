"""DSP measurement tests. These do not load Demucs."""

from __future__ import annotations

import numpy as np
import pytest

from stem_ai.analysis import (
    AnalysisError,
    analyze_audio,
    inference_duration_ratio,
    measure_stems,
    reconstruction_error,
)
from stem_ai.export import STEM_NAMES


def _tone(seconds: float = 1.0, frequency: float = 440.0, amplitude: float = 0.5, sample_rate: int = 44100):
    time = np.arange(int(seconds * sample_rate), dtype=np.float64) / sample_rate
    return amplitude * np.sin(2 * np.pi * frequency * time)


def test_sine_rms_matches_the_continuous_formula():
    features = analyze_audio(_tone(amplitude=0.5), 44100)

    assert features.rms == pytest.approx(0.5 / np.sqrt(2), rel=1e-6)


def test_sine_centroid_is_near_its_frequency():
    features = analyze_audio(_tone(frequency=440.0), 44100)

    assert features.spectral_centroid_hz == pytest.approx(440.0, abs=20.0)


def test_sine_zero_crossing_rate_is_near_two_cycles_per_sample():
    sample_rate = 44100
    frequency = 440.0
    features = analyze_audio(_tone(frequency=frequency), sample_rate)

    assert features.zero_crossing_rate == pytest.approx(2 * frequency / sample_rate, abs=0.002)


def test_noise_has_a_wider_bandwidth_than_a_pure_tone():
    sample_rate = 44100
    tone = analyze_audio(_tone(), sample_rate)
    noise = analyze_audio(np.random.default_rng(0).normal(0.0, 0.2, sample_rate), sample_rate)

    assert noise.spectral_bandwidth_hz > tone.spectral_bandwidth_hz
    assert np.isfinite(tone.spectral_bandwidth_hz)
    assert np.isfinite(noise.spectral_bandwidth_hz)


def test_silence_is_finite_and_has_no_energy():
    features = analyze_audio(np.zeros(44100), 44100)

    assert features.rms == 0.0
    assert features.zero_crossing_rate == 0.0
    assert features.spectral_centroid_hz == 0.0
    assert features.spectral_bandwidth_hz == 0.0
    assert np.isfinite(
        [
            features.rms,
            features.spectral_centroid_hz,
            features.spectral_bandwidth_hz,
            features.zero_crossing_rate,
        ]
    ).all()


def test_short_clip_stays_finite():
    features = analyze_audio(_tone(seconds=32 / 44100), 44100)

    assert features.rms > 0
    assert np.isfinite(
        [
            features.rms,
            features.spectral_centroid_hz,
            features.spectral_bandwidth_hz,
            features.zero_crossing_rate,
        ]
    ).all()


def test_large_constant_does_not_overflow():
    features = analyze_audio(np.full(4096, 1e8), 44100)

    assert features.rms == pytest.approx(1e8)
    assert np.isfinite(features.spectral_centroid_hz)
    assert np.isfinite(features.spectral_bandwidth_hz)
    assert features.zero_crossing_rate == pytest.approx(0.0, abs=1e-6)


def test_stereo_average_matches_the_mono_signal():
    mono = _tone()
    stereo = np.stack([mono, mono], axis=1)

    assert analyze_audio(stereo, 44100) == analyze_audio(mono, 44100)


def test_non_finite_samples_are_rejected():
    samples = _tone()
    samples[10] = np.nan

    with pytest.raises(AnalysisError, match="non-finite"):
        analyze_audio(samples, 44100)


def test_exact_stem_sum_has_zero_reconstruction_error():
    vocals = np.full((100, 2), 0.2)
    drums = np.full((100, 2), -0.05)
    bass = np.zeros((100, 2))
    other = np.full((100, 2), 0.1)
    mixture = vocals + drums + bass + other

    error = reconstruction_error(
        mixture,
        {"vocals": vocals, "drums": drums, "bass": bass, "other": other},
    )

    assert error.rmse == pytest.approx(0.0, abs=1e-12)
    assert error.peak_absolute_error == pytest.approx(0.0, abs=1e-12)
    assert error.relative_rmse == pytest.approx(0.0, abs=1e-12)


def test_constant_residual_is_reported_exactly():
    mixture = np.ones((50, 1))
    stems = {name: np.zeros((50, 1)) for name in STEM_NAMES}
    stems["vocals"] = np.full((50, 1), 0.8)

    error = reconstruction_error(mixture, stems)

    assert error.rmse == pytest.approx(0.2)
    assert error.peak_absolute_error == pytest.approx(0.2)
    assert error.relative_rmse == pytest.approx(0.2)


def test_silent_mixture_leaves_the_relative_error_undefined():
    zeros = {name: np.zeros((16, 1)) for name in STEM_NAMES}

    error = reconstruction_error(np.zeros((16, 1)), zeros)

    assert error.rmse == 0.0
    assert error.relative_rmse is None


def test_shape_mismatch_is_rejected():
    stems = {name: np.zeros((10, 1)) for name in STEM_NAMES}

    with pytest.raises(AnalysisError, match="shape"):
        reconstruction_error(np.zeros((8, 1)), stems)


def test_inference_ratio_uses_the_two_measured_times():
    assert inference_duration_ratio(3.25, 12.5) == pytest.approx(0.26)

    with pytest.raises(AnalysisError):
        inference_duration_ratio(1.0, 0.0)


def test_plots_are_png_files(tmp_path):
    stems = {name: _tone(seconds=0.2)[:, None] for name in STEM_NAMES}

    measured = measure_stems(stems, 44100, tmp_path)

    assert [item.stem for item in measured] == list(STEM_NAMES)
    for item in measured:
        for path in (item.waveform_plot, item.spectrogram_plot):
            header = path.read_bytes()[:8]
            assert header.startswith(b"\x89PNG")
            assert path.stat().st_size > 1000
