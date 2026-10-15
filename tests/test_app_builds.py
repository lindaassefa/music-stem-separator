"""The interface should import and build without downloading model weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

import app
from stem_ai.analysis import StemMeasurement
from stem_ai.export import STEM_NAMES
from stem_ai.separation import SeparationResult


def test_separating_message_does_not_claim_a_finished_time():
    message = app.show_separating()

    assert message == "Preparing audio"
    assert "seconds" not in message


def test_dashboard_names_the_product_and_the_pretrained_model():
    demo = app.build_demo()

    assert demo.title == "Music Stem Separator"
    assert "MUSIC STEM SEPARATOR" in app.HEADER
    assert "Pull a song apart." in app.HEADER
    assert "Separate vocals, drums, bass, and everything else." in app.HEADER
    readme = Path(__file__).resolve().parents[1].joinpath("README.md").read_text()
    assert "Hybrid Transformer Demucs" not in app.FOOTER
    assert "Hybrid Transformer Demucs" in readme
    assert "github.com/lindaassefa/music-stem-separator" in app.FOOTER
    texts = []
    for block in demo.blocks.values():
        for attr in ("label", "value"):
            item = getattr(block, attr, None)
            if isinstance(item, str):
                texts.append(item)
    blob = "\n".join(texts).lower()
    assert "Separate Audio" in texts
    for banned in (
        "benchmark",
        "rmse",
        "spectral",
        "inference",
        "real-time",
        "centroid",
        "zero-crossing",
    ):
        assert banned not in blob


def test_interface_builds_without_loading_the_model(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("Building the interface should not load htdemucs.")

    monkeypatch.setattr("stem_ai.separation.load_htdemucs", fail_if_called)

    demo = app.build_demo()

    assert demo is not None
    assert len(STEM_NAMES) == 4


def test_missing_upload_returns_a_message_and_no_audio():
    result = app.run_separation(None)

    assert result[0] is None
    assert result[1] is None
    assert result[2] is None
    assert result[3] is None
    assert "Upload a WAV or MP3" in result[4]
    assert len(result) == 5


def test_dashboard_receives_measured_stem_values(tmp_path, monkeypatch):
    seen = {}

    def fake_separate(_path, output_dir=None, *, analyze=True, loaded=None):
        seen["analyze"] = analyze
        seen["loaded"] = loaded
        measurements = tuple(
            StemMeasurement(
                stem=name,
                rms=0.1 * (index + 1),
                spectral_centroid_hz=1000.0 + index,
                spectral_bandwidth_hz=500.0 + index,
                zero_crossing_rate=0.01 * (index + 1),
                waveform_plot=tmp_path / f"{name}_waveform.png",
                spectrogram_plot=tmp_path / f"{name}_spectrogram.png",
            )
            for index, name in enumerate(STEM_NAMES)
        )
        return SeparationResult(
            vocals=tmp_path / "vocals.wav",
            drums=tmp_path / "drums.wav",
            bass=tmp_path / "bass.wav",
            other=tmp_path / "other.wav",
            inference_seconds=2.0,
            processing_seconds=2.5,
            audio_duration_seconds=4.0,
            device="mps",
            sample_rate=44100,
            reconstruction_rmse=0.125,
            reconstruction_relative_rmse=0.5,
            reconstruction_peak_absolute_error=0.25,
            measurements=measurements,
        )

    monkeypatch.setattr(app, "separate", fake_separate)
    upload = tmp_path / "song.wav"
    tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(8000) / 8000)).astype(np.float32)
    sf.write(upload, tone, 8000)
    vocals, drums, bass, other, status = app.run_separation(upload)

    assert vocals.endswith("vocals.wav")
    assert drums.endswith("drums.wav")
    assert bass.endswith("bass.wav")
    assert other.endswith("other.wav")
    assert status == ""
    assert "rmse" not in status.lower()
    assert seen["analyze"] is False
    assert seen["loaded"] is not None
