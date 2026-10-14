"""Checks that bad uploads stop before model loading, and that reports echo measurements."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from stem_ai.separation import SeparationResult, format_report, separate
from stem_ai.validation import AudioValidationError


def test_invalid_upload_does_not_load_the_model(tmp_path, monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("The pretrained model should not load for an invalid upload.")

    monkeypatch.setattr("stem_ai.separation.load_htdemucs", fail_if_called)
    path = tmp_path / "notes.txt"
    path.write_text("hello")

    with pytest.raises(AudioValidationError):
        separate(path, output_dir=tmp_path / "out")


def test_silence_does_not_load_the_model(tmp_path, monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("The pretrained model should not load for silence.")

    monkeypatch.setattr("stem_ai.separation.load_htdemucs", fail_if_called)
    path = tmp_path / "silence.wav"
    sf.write(str(path), np.zeros((44100, 2), dtype=np.float32), 44100)

    with pytest.raises(AudioValidationError, match="silent"):
        separate(path, output_dir=tmp_path / "out")


def test_report_prints_the_measurements_it_is_given(tmp_path):
    result = SeparationResult(
        vocals=tmp_path / "vocals.wav",
        drums=tmp_path / "drums.wav",
        bass=tmp_path / "bass.wav",
        other=tmp_path / "other.wav",
        inference_seconds=3.25,
        processing_seconds=4.5,
        audio_duration_seconds=12.5,
        device="mps",
        sample_rate=44100,
        reconstruction_rmse=0.012345,
        reconstruction_relative_rmse=0.25,
        reconstruction_peak_absolute_error=0.04,
    )

    text = format_report(result)

    assert "htdemucs" in text
    assert "does not train Demucs" in text
    assert "Device: mps" in text
    assert "Audio duration: 12.50 s" in text
    assert "Inference time: 3.25 s" in text
    assert "Processing time: 4.50 s" in text
    assert "Inference time / audio duration: 0.260" in text
    assert "Reconstruction RMSE: 1.234500e-02" in text
    assert "Reconstruction RMSE / mixture RMS: 2.500000e-01" in text
    assert "Reconstruction peak absolute error: 4.000000e-02" in text
    assert "not a source-separation accuracy metric" in text
    assert "DSP measurements" in text
    assert "Output sample rate: 44100 Hz" in text
    assert isinstance(result.paths()["bass"], Path)
