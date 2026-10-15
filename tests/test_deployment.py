"""Public-space limits. These tests do not download Demucs or reserve a GPU."""

from __future__ import annotations

import os
import time

import numpy as np
import soundfile as sf

from stem_ai.benchmark_metrics import format_saved_benchmark, load_saved_benchmark
from stem_ai.deployment import bind_zero_gpu, cleanup_old_outputs, on_huggingface_space
from stem_ai.validation import AudioValidationError, load_audio, max_duration_seconds
import app


def _tone(path, seconds: float, sample_rate: int = 8000) -> None:
    frames = int(seconds * sample_rate)
    tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(frames) / sample_rate)).astype(np.float32)
    sf.write(path, tone, sample_rate)


def test_space_detection_follows_space_id(monkeypatch):
    monkeypatch.delenv("SPACE_ID", raising=False)
    assert on_huggingface_space() is False
    monkeypatch.setenv("SPACE_ID", "lindaassefa/music-stem-separator")
    assert on_huggingface_space() is True


def test_local_duration_limit_stays_at_ten_minutes(monkeypatch):
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.delenv("STEM_SHARE", raising=False)
    monkeypatch.delenv("STEM_MAX_DURATION_SEC", raising=False)
    assert max_duration_seconds() == 600


def test_public_space_uses_the_ten_minute_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACE_ID", "lindaassefa/music-stem-separator")
    monkeypatch.delenv("STEM_SHARE", raising=False)
    monkeypatch.delenv("STEM_MAX_DURATION_SEC", raising=False)
    assert max_duration_seconds() == 600

    accepted = tmp_path / "short.wav"
    _tone(accepted, 31)
    loaded = load_audio(accepted)
    assert loaded.duration_seconds == 31

    path = tmp_path / "too_long.wav"
    _tone(path, 601)
    try:
        load_audio(path)
    except AudioValidationError as exc:
        assert "600 second" in str(exc)
    else:
        raise AssertionError("Audio longer than 10 minutes should be rejected.")


def test_public_space_rejects_an_oversized_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACE_ID", "lindaassefa/music-stem-separator")
    monkeypatch.delenv("STEM_SHARE", raising=False)
    path = tmp_path / "big.wav"
    _tone(path, 1)
    # Shrink the allowed size for this test only.
    monkeypatch.setattr("stem_ai.validation.PUBLIC_MAX_FILE_BYTES", 10)
    try:
        load_audio(path)
    except AudioValidationError as exc:
        assert "larger than" in str(exc)
    else:
        raise AssertionError("An oversized public upload should be rejected.")


def test_cleanup_removes_only_old_run_folders(tmp_path):
    old = tmp_path / "old-run"
    fresh = tmp_path / "fresh-run"
    loose = tmp_path / "note.txt"
    old.mkdir()
    fresh.mkdir()
    loose.write_text("keep")
    old_time = time.time() - 3600
    os.utime(old, (old_time, old_time))

    removed = cleanup_old_outputs(tmp_path, max_age_seconds=60)

    assert removed == 1
    assert not old.exists()
    assert fresh.is_dir()
    assert loose.read_text() == "keep"


def test_missing_spaces_package_does_not_change_the_call():
    def add_one(value):
        return value + 1

    wrapped = bind_zero_gpu(add_one, duration=90)
    assert wrapped(2) == 3


def test_share_link_uses_the_ten_minute_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("STEM_SHARE", "1")
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.delenv("STEM_MAX_DURATION_SEC", raising=False)
    assert max_duration_seconds() == 600

    allowed = tmp_path / "allowed.wav"
    _tone(allowed, 16)
    loaded = load_audio(allowed)
    assert loaded.duration_seconds == 16

    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(upload_root))
    too_long = upload_root / "too_long.wav"
    _tone(too_long, 601)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("Audio longer than 10 minutes must be rejected before the model runs.")

    monkeypatch.setattr(app, "separate_checked", fail_if_called)
    _vocals, _drums, _bass, _other, report, *_rest = app.run_separation(too_long)
    assert "600 second" in report

    outside = tmp_path / "private.wav"
    _tone(outside, 2)
    vocals, _drums, _bass, _other, report, *_rest = app.run_separation(outside)
    assert vocals is None
    assert report == "Upload a WAV or MP3 file."


def test_local_sixteen_second_clip_stays_allowed(tmp_path, monkeypatch):
    monkeypatch.delenv("STEM_SHARE", raising=False)
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.delenv("STEM_MAX_DURATION_SEC", raising=False)
    path = tmp_path / "local.wav"
    _tone(path, 16)
    loaded = load_audio(path)
    assert loaded.duration_seconds == 16


def test_share_serving_blocks_private_directories(tmp_path, monkeypatch):
    monkeypatch.setenv("STEM_SHARE", "1")
    monkeypatch.delenv("SPACE_ID", raising=False)
    policy = app.serving_policy(tmp_path)
    blocked = policy["blocked_paths"]

    assert policy["share"] is True
    assert policy["allowed_paths"] == [str(tmp_path.resolve())]
    assert policy["max_file_size"] == "250mb"
    assert policy["max_size"] == 1
    assert policy["run_history"] is False
    assert any(path.endswith(f"{os.sep}.ssh") for path in blocked)
    assert any(path.endswith(f"{os.sep}huggingface") and ".cache" in path for path in blocked)
    assert any(path.endswith(f"{os.sep}outputs") for path in blocked)
    assert any(path.endswith(f"{os.sep}.git") for path in blocked)
    assert str(tmp_path.resolve()) not in blocked


def test_local_serving_does_not_open_a_share_link(tmp_path, monkeypatch):
    monkeypatch.delenv("STEM_SHARE", raising=False)
    monkeypatch.delenv("SPACE_ID", raising=False)
    policy = app.serving_policy(tmp_path)

    assert policy["share"] is False
    assert policy["max_file_size"] is None
    assert policy["max_size"] is None
    assert policy["blocked_paths"] == []


def test_public_share_omits_the_saved_benchmark_file(monkeypatch):
    monkeypatch.setenv("STEM_SHARE", "1")
    text, chart = app.saved_benchmark_for_display()

    assert chart is None
    assert "does not publish the saved Mac benchmark" in text
    assert "local machine" in text
    assert "/" not in text


def test_benchmark_text_is_labeled_as_a_mac_measurement():
    text = format_saved_benchmark(
        {
            "audio_duration_seconds": 10,
            "model_load_seconds": 0.2,
            "inference_settings": {"shifts": 0, "overlap": 0.25, "split": True, "segment": None},
            "devices": {},
        }
    )
    empty, chart = load_saved_benchmark("/this/folder/does/not/exist")

    assert "local Mac" in text
    assert "not Hugging Face server benchmarks" in text
    assert "not Hugging Face server measurements" in empty
    assert chart is None
    assert "local Mac" in app.BENCHMARK_NOTE
    assert "not a measurement of this server" in app.BENCHMARK_NOTE
