"""Tests for stem arrays and the WAV files written from them."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from stem_ai.export import (
    STEM_NAMES,
    OutputIntegrityError,
    validate_stem_arrays,
    verify_written_wavs,
    write_stem_wavs,
)


def _stems(frames: int = 1000, channels: int = 2, amplitude: float = 0.4) -> dict[str, np.ndarray]:
    stems = {}
    for index, name in enumerate(STEM_NAMES):
        time = np.arange(frames, dtype=np.float32)
        tone = amplitude * np.sin(2 * np.pi * (220 + 40 * index) * time / 44100)
        stems[name] = np.repeat(tone[:, None], channels, axis=1).astype(np.float32)
    return stems


def test_matching_stems_pass():
    validate_stem_arrays(_stems(), sample_rate=44100, expected_frames=1000)


def test_missing_stem_fails():
    stems = _stems()
    del stems["bass"]

    with pytest.raises(OutputIntegrityError, match="Missing stems: bass"):
        validate_stem_arrays(stems, sample_rate=44100)


def test_non_finite_stem_fails():
    stems = _stems()
    stems["drums"][0, 0] = np.nan

    with pytest.raises(OutputIntegrityError, match="non-finite"):
        validate_stem_arrays(stems, sample_rate=44100)


def test_different_lengths_fail():
    stems = _stems()
    stems["other"] = stems["other"][:-10]

    with pytest.raises(OutputIntegrityError, match="same length"):
        validate_stem_arrays(stems, sample_rate=44100)


def test_unexpected_length_fails():
    with pytest.raises(OutputIntegrityError, match="expected length"):
        validate_stem_arrays(_stems(frames=1000), sample_rate=44100, expected_frames=900)


def test_write_and_read_back_round_trip(tmp_path):
    stems = _stems(frames=2048)
    written = write_stem_wavs(stems, 44100, tmp_path / "run")

    verify_written_wavs(written, 44100, 2048)
    for name in STEM_NAMES:
        info = sf.info(str(written[name]))
        data, sample_rate = sf.read(str(written[name]), always_2d=True, dtype="float32")
        assert info.subtype == "PCM_16"
        assert sample_rate == 44100
        assert data.shape == (2048, 2)
        assert np.max(np.abs(data)) > 0.1
        assert np.max(np.abs(data)) <= 1.0


def test_peaks_above_full_scale_are_clipped_into_the_wav(tmp_path):
    stems = _stems(amplitude=2.0)
    written = write_stem_wavs(stems, 44100, tmp_path / "clipped")

    verify_written_wavs(written, 44100, 1000)
    data, _sample_rate = sf.read(str(written["vocals"]), always_2d=True, dtype="float32")
    assert float(np.max(np.abs(data))) <= 1.0


def test_unreadable_wav_fails_verification(tmp_path):
    written = write_stem_wavs(_stems(), 44100, tmp_path / "broken")
    written["vocals"].write_bytes(b"not a wav anymore")

    with pytest.raises(OutputIntegrityError, match="Could not read the vocals WAV"):
        verify_written_wavs(written, 44100, 1000)
