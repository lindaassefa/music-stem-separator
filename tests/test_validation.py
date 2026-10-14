"""Tests for upload checks. These do not load Demucs."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from stem_ai.validation import (
    MIN_DURATION_SECONDS,
    SILENCE_PEAK_THRESHOLD,
    AudioValidationError,
    load_audio,
)


def _sine(
    duration: float,
    sample_rate: int = 44100,
    amplitude: float = 0.2,
    channels: int = 2,
) -> np.ndarray:
    frames = int(round(duration * sample_rate))
    time = np.arange(frames, dtype=np.float32) / sample_rate
    mono = (amplitude * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    if channels == 1:
        return mono[:, None]
    return np.repeat(mono[:, None], channels, axis=1)


def _write_wav(path, samples: np.ndarray, sample_rate: int = 44100) -> None:
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")


def test_stereo_wav_loads(tmp_path):
    path = tmp_path / "song.wav"
    _write_wav(path, _sine(1.0))

    loaded = load_audio(path)

    assert loaded.source_format == "wav"
    assert loaded.sample_rate == 44100
    assert loaded.channels == 2
    assert loaded.duration_seconds == pytest.approx(1.0, abs=0.02)
    assert loaded.samples.shape[0] > 0


def test_mono_wav_and_uppercase_extension_load(tmp_path):
    path = tmp_path / "voice.WAV"
    _write_wav(path, _sine(0.8, channels=1), sample_rate=48000)

    loaded = load_audio(path)

    assert loaded.channels == 1
    assert loaded.sample_rate == 48000
    assert loaded.source_format == "wav"


def test_minimum_duration_is_accepted(tmp_path):
    path = tmp_path / "short.wav"
    _write_wav(path, _sine(MIN_DURATION_SECONDS))

    loaded = load_audio(path)

    assert loaded.duration_seconds == pytest.approx(MIN_DURATION_SECONDS, abs=0.02)


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.wav"
    path.write_bytes(b"")

    with pytest.raises(AudioValidationError, match="empty"):
        load_audio(path)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(AudioValidationError, match="could not be found"):
        load_audio(tmp_path / "missing.wav")


def test_directory_is_rejected(tmp_path):
    with pytest.raises(AudioValidationError, match="WAV or MP3"):
        load_audio(tmp_path)


def test_text_file_is_rejected(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not audio")

    with pytest.raises(AudioValidationError, match="WAV or MP3"):
        load_audio(path)


def test_flac_extension_is_rejected(tmp_path):
    path = tmp_path / "song.flac"
    path.write_bytes(b"fLaC" + b"\x00" * 32)

    with pytest.raises(AudioValidationError, match="WAV or MP3"):
        load_audio(path)


def test_wav_extension_with_mp3_header_is_rejected(tmp_path):
    path = tmp_path / "mislabeled.wav"
    path.write_bytes(b"ID3" + b"\x00" * 64)

    with pytest.raises(AudioValidationError, match="not WAV"):
        load_audio(path)


def test_mp3_extension_with_wav_bytes_is_rejected(tmp_path):
    wav_path = tmp_path / "real.wav"
    _write_wav(wav_path, _sine(0.6))
    mp3_path = tmp_path / "mislabeled.mp3"
    mp3_path.write_bytes(wav_path.read_bytes())

    with pytest.raises(AudioValidationError, match="not MP3"):
        load_audio(mp3_path)


def test_truncated_wav_is_rejected(tmp_path):
    path = tmp_path / "truncated.wav"
    path.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")

    with pytest.raises(AudioValidationError, match="could not be read"):
        load_audio(path)


def test_real_mp3_decodes(tmp_path):
    lameenc = pytest.importorskip("lameenc")
    sample_rate = 44100
    frames = sample_rate
    time = np.arange(frames, dtype=np.float32) / sample_rate
    mono = (0.2 * np.sin(2 * np.pi * 440 * time) * 32767).astype(np.int16)
    stereo = np.repeat(mono[:, None], 2, axis=1)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(128)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(2)
    encoder.set_quality(2)
    payload = encoder.encode(stereo.tobytes()) + encoder.flush()
    path = tmp_path / "tone.mp3"
    path.write_bytes(payload)

    loaded = load_audio(path)

    assert loaded.source_format == "mp3"
    assert loaded.sample_rate == sample_rate
    assert loaded.channels == 2
    assert loaded.duration_seconds >= MIN_DURATION_SECONDS
    assert float(np.max(np.abs(loaded.samples))) >= SILENCE_PEAK_THRESHOLD


def test_corrupt_mp3_is_rejected(tmp_path):
    path = tmp_path / "bad.mp3"
    path.write_bytes(b"ID3" + b"\x03\x00" + b"\x00" * 64)

    with pytest.raises(AudioValidationError, match="could not be decoded"):
        load_audio(path)


def test_silence_is_rejected(tmp_path):
    path = tmp_path / "silence.wav"
    _write_wav(path, np.zeros((44100, 2), dtype=np.float32))

    with pytest.raises(AudioValidationError, match="silent"):
        load_audio(path)


def test_near_silence_is_rejected(tmp_path):
    samples = np.full((44100, 1), SILENCE_PEAK_THRESHOLD / 2, dtype=np.float32)
    path = tmp_path / "quiet.wav"
    _write_wav(path, samples)

    with pytest.raises(AudioValidationError, match="silent"):
        load_audio(path)


def test_too_short_audio_is_rejected(tmp_path):
    path = tmp_path / "blip.wav"
    _write_wav(path, _sine(0.2))

    with pytest.raises(AudioValidationError, match="at least"):
        load_audio(path)


def test_configured_duration_limit_is_enforced(tmp_path, monkeypatch):
    monkeypatch.setenv("STEM_MAX_DURATION_SEC", "1")
    path = tmp_path / "long.wav"
    _write_wav(path, _sine(1.5))

    with pytest.raises(AudioValidationError, match="above the 1 second limit"):
        load_audio(path)


def test_extensionless_wav_bytes_are_accepted(tmp_path):
    wav_path = tmp_path / "song.wav"
    _write_wav(wav_path, _sine(0.7))
    bare_path = tmp_path / "upload"
    bare_path.write_bytes(wav_path.read_bytes())

    loaded = load_audio(bare_path)

    assert loaded.source_format == "wav"
