"""Decide whether an uploaded file is a song Demucs can separate."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from stem_ai.deployment import (
    PUBLIC_MAX_DURATION_SECONDS,
    PUBLIC_MAX_FILE_BYTES,
    SHARE_MAX_DURATION_SECONDS,
    SHARE_MAX_FILE_BYTES,
    on_huggingface_space,
    public_share_enabled,
)

import numpy as np
import soundfile as sf

# Peak amplitude below this is about -60 dBFS. A whole file that never
# exceeds it has no musical signal worth sending to the model.
SILENCE_PEAK_THRESHOLD = 1e-3
MIN_DURATION_SECONDS = 0.5
DEFAULT_MAX_DURATION_SECONDS = 600.0
MAX_FILE_BYTES = 250 * 1024 * 1024
MAX_CHANNELS = 8
MIN_SAMPLE_RATE = 8_000
MAX_SAMPLE_RATE = 192_000


class AudioValidationError(ValueError):
    """The upload cannot be separated. The message is safe to show in the UI."""


@dataclass(frozen=True)
class LoadedAudio:
    """Decoded audio in float32, shaped (frames, channels), before resampling."""

    samples: np.ndarray
    sample_rate: int
    duration_seconds: float
    source_format: str

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])


def _max_file_bytes() -> int:
    if public_share_enabled():
        return SHARE_MAX_FILE_BYTES
    if on_huggingface_space():
        return PUBLIC_MAX_FILE_BYTES
    return MAX_FILE_BYTES


def limits_text() -> str:
    """The duration and size cap currently in force, for the upload note."""

    seconds = max_duration_seconds()
    if seconds >= 60 and abs(seconds - round(seconds / 60) * 60) < 0.05:
        duration = f"{int(round(seconds / 60))} minutes"
    else:
        duration = f"{seconds:.0f} seconds"
    megabytes = max(1, _max_file_bytes() // (1024 * 1024))
    return (
        f"WAV or MP3. Up to {duration} and {megabytes} MB. "
        "One song is separated at a time."
    )


def max_duration_seconds() -> float:
    raw = os.environ.get("STEM_MAX_DURATION_SEC", str(DEFAULT_MAX_DURATION_SECONDS))
    try:
        value = float(raw)
    except ValueError as exc:
        raise AudioValidationError("STEM_MAX_DURATION_SEC must be a number of seconds.") from exc
    if value <= MIN_DURATION_SECONDS:
        raise AudioValidationError(
            "STEM_MAX_DURATION_SEC must be longer than the minimum audio length."
        )
    if public_share_enabled():
        return min(value, SHARE_MAX_DURATION_SECONDS)
    if on_huggingface_space():
        return min(value, PUBLIC_MAX_DURATION_SECONDS)
    return value


def load_audio(path: str | Path) -> LoadedAudio:
    """Load a WAV or MP3 file, or raise AudioValidationError."""

    source = Path(path)
    if not source.exists():
        raise AudioValidationError("The uploaded file could not be found.")
    if not source.is_file():
        raise AudioValidationError("Upload a WAV or MP3 file, not a folder.")

    size = source.stat().st_size
    if size == 0:
        raise AudioValidationError("The uploaded file is empty.")
    limit = _max_file_bytes()
    if size > limit:
        raise AudioValidationError(
            f"The uploaded file is larger than {max(1, limit // (1024 * 1024))} MB."
        )

    audio_format = _confirmed_format(source)
    if audio_format == "wav":
        samples, sample_rate = _read_wav(source)
    else:
        samples, sample_rate = _read_mp3(source)

    return _validate_samples(samples, sample_rate, audio_format)


def _confirmed_format(path: Path) -> str:
    suffix = path.suffix.lower()
    sniffed = _sniff_format(path)
    if suffix == ".wav":
        if sniffed != "wav":
            raise AudioValidationError("This file ends in .wav but its contents are not WAV.")
        return "wav"
    if suffix == ".mp3":
        if sniffed != "mp3":
            raise AudioValidationError("This file ends in .mp3 but its contents are not MP3.")
        return "mp3"
    if suffix == "" and sniffed in {"wav", "mp3"}:
        return sniffed
    raise AudioValidationError("Upload a WAV or MP3 file.")


def _sniff_format(path: Path) -> str | None:
    with path.open("rb") as handle:
        header = handle.read(16)
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "wav"
    if header.startswith(b"ID3"):
        return "mp3"
    # MP3 frame sync: 11 set bits at the start of a frame.
    if len(header) >= 2 and header[0] == 0xFF and header[1] & 0xE0 == 0xE0:
        return "mp3"
    return None


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    try:
        samples, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    except Exception as exc:
        raise AudioValidationError(f"The WAV file could not be read: {exc}") from exc
    return np.asarray(samples, dtype=np.float32), int(sample_rate)


def _read_mp3(path: Path) -> tuple[np.ndarray, int]:
    try:
        import miniaudio
    except ImportError as exc:
        raise AudioValidationError(
            "MP3 decoding requires the miniaudio package. Reinstall the project dependencies."
        ) from exc

    try:
        decoded = miniaudio.decode_file(
            str(path),
            output_format=miniaudio.SampleFormat.FLOAT32,
        )
    except Exception as exc:
        raise AudioValidationError(f"The MP3 file could not be decoded: {exc}") from exc

    channels = int(decoded.nchannels)
    sample_rate = int(decoded.sample_rate)
    interleaved = np.asarray(decoded.samples, dtype=np.float32)
    if channels <= 0 or interleaved.size == 0 or interleaved.size % channels != 0:
        raise AudioValidationError("The MP3 file did not contain complete audio frames.")
    return interleaved.reshape(-1, channels), sample_rate


def _validate_samples(samples: np.ndarray, sample_rate: int, audio_format: str) -> LoadedAudio:
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise AudioValidationError("The audio file has no samples.")
    if samples.shape[1] > MAX_CHANNELS:
        raise AudioValidationError(
            f"Audio with more than {MAX_CHANNELS} channels is not supported."
        )
    if sample_rate < MIN_SAMPLE_RATE or sample_rate > MAX_SAMPLE_RATE:
        raise AudioValidationError(
            f"Sample rate must be between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE} Hz."
        )
    if not np.isfinite(samples).all():
        raise AudioValidationError("The audio contains non-finite samples.")

    duration = float(samples.shape[0] / sample_rate)
    if duration < MIN_DURATION_SECONDS:
        raise AudioValidationError(
            f"Audio must be at least {MIN_DURATION_SECONDS:.1f} seconds long."
        )
    limit = max_duration_seconds()
    if duration > limit:
        raise AudioValidationError(
            f"Audio is {duration:.1f} seconds long, above the {limit:.0f} second limit."
        )

    peak = float(np.max(np.abs(samples)))
    if peak < SILENCE_PEAK_THRESHOLD:
        raise AudioValidationError("This file is silent, so there is nothing to separate.")

    return LoadedAudio(
        samples=np.ascontiguousarray(samples, dtype=np.float32),
        sample_rate=sample_rate,
        duration_seconds=duration,
        source_format=audio_format,
    )
