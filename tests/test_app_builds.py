"""The interface should import and build without downloading model weights."""

from __future__ import annotations

import app
from stem_ai.analysis import StemMeasurement
from stem_ai.export import STEM_NAMES
from stem_ai.separation import SeparationResult


def test_separating_message_does_not_claim_a_finished_time():
    message = app.show_separating()

    assert "pretrained htdemucs" in message
    assert "seconds" not in message


def test_dashboard_names_the_product_and_the_pretrained_model():
    demo = app.build_demo()

    assert demo.title == "Music Stem Separator"
    assert "Music Stem Separator" in app.HEADER
    assert "Neural Audio Intelligence" in app.HEADER
    assert "htdemucs" in app.FOOTER
    assert "not a source-separation accuracy metric" in app.ANALYSIS_NOTE


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
    assert result[5] is None
    assert all(item is None for item in result[6:])
    assert len(result) == 14


def test_dashboard_receives_measured_stem_values(tmp_path, monkeypatch):
    def fake_separate(_path, output_dir=None):
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
    vocals, _drums, _bass, _other, report, table, *figures = app.run_separation("song.wav")

    assert vocals.endswith("vocals.wav")
    assert "Inference time / audio duration: 0.500" in report
    assert "Reconstruction RMSE: 1.250000e-01" in report
    assert table[0][0] == "Vocals"
    assert table[0][1] == 0.1
    assert figures[0].endswith("vocals_waveform.png")
    assert figures[1].endswith("vocals_spectrogram.png")
    assert len(figures) == 8
