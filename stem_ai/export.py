"""Write Demucs stems to WAV files and check that those files are usable."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

# Playback order in the interface. htdemucs emits these four names.
STEM_NAMES = ("vocals", "drums", "bass", "other")


class OutputIntegrityError(ValueError):
    """The model output or a written WAV does not match the stem contract."""


def tensor_to_frames(tensor) -> np.ndarray:
    """Convert a Demucs stem, shaped (channels, frames), to (frames, channels)."""

    array = tensor.detach().to("cpu").float().numpy()
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise OutputIntegrityError("A stem tensor did not have channel and time axes.")
    # Demucs stores channels first. WAV writers and the integrity checks use
    # frames first, which is also the layout soundfile expects.
    return np.ascontiguousarray(np.transpose(array), dtype=np.float32)


def validate_stem_arrays(
    stems: dict[str, np.ndarray],
    sample_rate: int,
    expected_frames: int | None = None,
) -> None:
    """Require four finite stems that share a shape and sample rate."""

    missing = [name for name in STEM_NAMES if name not in stems]
    if missing:
        raise OutputIntegrityError(f"Missing stems: {', '.join(missing)}.")
    extra = sorted(set(stems) - set(STEM_NAMES))
    if extra:
        raise OutputIntegrityError(f"Unexpected stems: {', '.join(extra)}.")
    if sample_rate <= 0:
        raise OutputIntegrityError("Sample rate must be positive.")

    shapes: set[tuple[int, int]] = set()
    for name in STEM_NAMES:
        audio = np.asarray(stems[name])
        if audio.ndim != 2 or audio.shape[0] == 0 or audio.shape[1] == 0:
            raise OutputIntegrityError(f"{name} is empty or has the wrong shape.")
        if not np.isfinite(audio).all():
            raise OutputIntegrityError(f"{name} contains non-finite samples.")
        shapes.add((int(audio.shape[0]), int(audio.shape[1])))

    if len(shapes) != 1:
        raise OutputIntegrityError("The four stems do not have the same length and channel count.")
    frames = next(iter(shapes))[0]
    if expected_frames is not None and frames != expected_frames:
        raise OutputIntegrityError(
            f"Stem length {frames} does not match the expected length {expected_frames}."
        )


def write_stem_wavs(
    stems: dict[str, np.ndarray],
    sample_rate: int,
    output_dir: Path,
) -> dict[str, Path]:
    """Validate stems, clip them into the WAV range, and write 16-bit files."""

    validate_stem_arrays(stems, sample_rate)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name in STEM_NAMES:
        audio = np.clip(np.asarray(stems[name], dtype=np.float32), -1.0, 1.0)
        path = output_dir / f"{name}.wav"
        sf.write(str(path), audio, sample_rate, subtype="PCM_16", format="WAV")
        written[name] = path
    return written


def verify_written_wavs(
    paths: dict[str, Path],
    sample_rate: int,
    frames: int,
) -> None:
    """Read the WAV files back and confirm they match the tensors that were saved."""

    for name in STEM_NAMES:
        path = paths.get(name)
        if path is None or not path.is_file() or path.stat().st_size == 0:
            raise OutputIntegrityError(f"The {name} WAV was not written.")
        try:
            info = sf.info(str(path))
            data, read_rate = sf.read(str(path), always_2d=True, dtype="float32")
        except Exception as exc:
            raise OutputIntegrityError(f"Could not read the {name} WAV: {exc}") from exc
        if info.subtype != "PCM_16":
            raise OutputIntegrityError(f"The {name} WAV is not 16-bit PCM.")
        if int(read_rate) != sample_rate or int(info.samplerate) != sample_rate:
            raise OutputIntegrityError(f"The {name} WAV has an unexpected sample rate.")
        if int(info.frames) != frames or data.shape[0] != frames:
            raise OutputIntegrityError(f"The {name} WAV has an unexpected length.")
        if data.shape[1] <= 0 or not np.isfinite(data).all():
            raise OutputIntegrityError(f"The {name} WAV contains invalid samples.")
        if float(np.max(np.abs(data))) > 1.0 + 1e-3:
            raise OutputIntegrityError(f"The {name} WAV is outside the playable range.")
