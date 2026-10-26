"""Run pretrained Hybrid Transformer Demucs on one uploaded song."""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

# Missing MPS operators fall back to CPU instead of aborting the forward pass.
# Demucs also does this itself for complex spectrogram math. Respect a value
# the user has already exported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from stem_ai.analysis import (
    AnalysisError,
    StemMeasurement,
    inference_duration_ratio,
    measure_stems,
    reconstruction_error,
)
from stem_ai.device import DeviceError, select_device
from stem_ai.export import (
    STEM_NAMES,
    OutputIntegrityError,
    tensor_to_frames,
    validate_stem_arrays,
    verify_written_wavs,
    write_stem_wavs,
)
from stem_ai.validation import LoadedAudio, load_audio

logger = logging.getLogger(__name__)

MODEL_NAME = "htdemucs"
# The authors limit Hybrid Transformer Demucs chunks to 7.8 seconds.
MAX_SEGMENT_SECONDS = 7.8
# Resampling can change the length by a few samples. A quarter second is enough
# slack to catch a truncated or padded stem without flagging normal rate conversion.
DURATION_TOLERANCE_SECONDS = 0.25

_LOCK = threading.Lock()
_MODEL: dict[str, object] = {}


class SeparationError(RuntimeError):
    """Pretrained inference failed, or its output could not be saved."""


@dataclass(frozen=True)
class SeparationResult:
    vocals: Path
    drums: Path
    bass: Path
    other: Path
    inference_seconds: float
    processing_seconds: float
    audio_duration_seconds: float
    device: str
    sample_rate: int
    reconstruction_rmse: float
    reconstruction_relative_rmse: float | None
    reconstruction_peak_absolute_error: float
    measurements: tuple[StemMeasurement, ...] = ()
    analysis_error: str | None = None
    model_name: str = MODEL_NAME

    def paths(self) -> dict[str, Path]:
        return {
            "vocals": self.vocals,
            "drums": self.drums,
            "bass": self.bass,
            "other": self.other,
        }


def format_report(result: SeparationResult) -> str:
    """Render the measurements taken for this run. Values come from the caller."""

    ratio = inference_duration_ratio(result.inference_seconds, result.audio_duration_seconds)
    lines = [
        "Model: htdemucs (pretrained weights; this project does not train Demucs)",
        f"Device: {result.device}",
        f"Audio duration: {result.audio_duration_seconds:.2f} s",
        f"Inference time: {result.inference_seconds:.2f} s",
        f"Processing time: {result.processing_seconds:.2f} s",
        f"Inference time / audio duration: {ratio:.3f}",
        f"Output sample rate: {result.sample_rate} Hz",
        f"Reconstruction RMSE: {result.reconstruction_rmse:.6e}",
        "Reconstruction RMSE / mixture RMS: "
        + _format_relative(result.reconstruction_relative_rmse),
        f"Reconstruction peak absolute error: {result.reconstruction_peak_absolute_error:.6e}",
        "Reconstruction RMSE is the residual between the mixture and the summed stems. It is not a source-separation accuracy metric.",
        "Stems: vocals, drums, bass, other",
        "Stem RMS, centroid, bandwidth, and zero-crossing rate are DSP measurements, not model outputs.",
    ]
    if result.analysis_error:
        lines.append(f"Analysis failed: {result.analysis_error}")
    return "\n".join(lines)


def separate(
    path: str | Path,
    output_dir: str | Path | None = None,
    *,
    analyze: bool = True,
    loaded: LoadedAudio | None = None,
    model_name: str = MODEL_NAME,
) -> SeparationResult:
    """Separate one song with pretrained htdemucs.

    Validation happens before the model is loaded, so a bad upload does not
    download weights. The recorded inference time covers only the successful
    ``apply_model`` call, not weight loading and not writing the WAV files.

    Demucs 4.0.1 does not ship ``demucs.api``. This function uses the same
    calls as the Demucs command line: ``get_model("htdemucs")`` and
    ``apply_model``. WAV is decoded with soundfile and MP3 with miniaudio,
    so ffmpeg is not required.
    """

    if model_name not in ("htdemucs", "htdemucs_ft"):
        raise SeparationError(f"Unsupported model: {model_name}")

    decode_started = time.perf_counter()
    if loaded is None:
        loaded = load_audio(path)
    decode_seconds = time.perf_counter() - decode_started
    destination = Path(output_dir) if output_dir is not None else Path(
        os.environ.get("STEM_OUTPUT_DIR", "outputs")
    )
    run_dir = destination / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    processing_started = time.perf_counter()

    with _LOCK:
        requested = select_device()
        try:
            model, separated, inference_seconds, waveform, stage_seconds = _separate_loaded(
                loaded, requested, model_name
            )
            device = requested
        except Exception as exc:
            if requested == "cpu" or isinstance(exc, (SeparationError, OutputIntegrityError, DeviceError)):
                if isinstance(exc, (SeparationError, OutputIntegrityError, DeviceError)):
                    raise
                raise SeparationError(
                    f"Demucs inference failed on cpu: {_short_error(exc)}"
                ) from exc
            logger.warning(
                "Inference on %s failed. Retrying the same audio on cpu.",
                requested,
                exc_info=True,
            )
            try:
                model, separated, inference_seconds, waveform, stage_seconds = _separate_loaded(
                    loaded, "cpu", model_name
                )
            except Exception as cpu_exc:
                raise SeparationError(
                    "Demucs inference failed on "
                    f"{requested} ({_short_error(exc)}) and on cpu ({_short_error(cpu_exc)})."
                ) from cpu_exc
            device = f"cpu after {requested} failed"

        frames = _frames_from_model(separated)
        expected_frames = int(frames["vocals"].shape[0])
        sample_rate = int(model.samplerate)
        validate_stem_arrays(frames, sample_rate, expected_frames)
        _check_duration(loaded.duration_seconds, expected_frames, sample_rate)
        mixture = tensor_to_frames(waveform)
        check_started = time.perf_counter()
        residual = reconstruction_error(mixture, frames)
        stage_seconds["reconstruction_seconds"] = time.perf_counter() - check_started
        write_started = time.perf_counter()
        try:
            written = write_stem_wavs(frames, sample_rate, run_dir)
            verify_written_wavs(written, sample_rate, expected_frames)
        except OutputIntegrityError:
            raise
        except OSError as exc:
            raise SeparationError(f"Could not write stem WAV files: {exc}") from exc
        stage_seconds["wav_write_seconds"] = time.perf_counter() - write_started
        analysis_started = time.perf_counter()
        if analyze:
            try:
                measurements = measure_stems(frames, sample_rate, run_dir)
                analysis_error = None
            except AnalysisError as exc:
                logger.exception("Stem measurement failed after separation")
                measurements = ()
                analysis_error = str(exc)
        else:
            measurements = ()
            analysis_error = None
        stage_seconds["analysis_seconds"] = time.perf_counter() - analysis_started
        stage_seconds["decode_seconds"] = decode_seconds
        logger.info(
            "separation stages device=%s audio=%.2fs %s",
            device,
            loaded.duration_seconds,
            " ".join(f"{name}={seconds:.3f}s" for name, seconds in stage_seconds.items()),
        )

    return SeparationResult(
        vocals=written["vocals"],
        drums=written["drums"],
        bass=written["bass"],
        other=written["other"],
        inference_seconds=inference_seconds,
        processing_seconds=time.perf_counter() - processing_started,
        audio_duration_seconds=loaded.duration_seconds,
        device=device,
        sample_rate=sample_rate,
        reconstruction_rmse=residual.rmse,
        reconstruction_relative_rmse=residual.relative_rmse,
        reconstruction_peak_absolute_error=residual.peak_absolute_error,
        measurements=measurements,
        analysis_error=analysis_error,
        model_name=model_name,
    )


def _separate_loaded(
    loaded: LoadedAudio,
    device: str,
    model_name: str = MODEL_NAME,
):
    """Move htdemucs onto `device`, then time only the apply_model call."""

    load_started = time.perf_counter()
    model = load_htdemucs(model_name)
    model_load_seconds = time.perf_counter() - load_started
    # Weight loading happens in load_htdemucs. Moving the module is also kept
    # outside the timer so the reported time is the separation call itself.
    # A second request leaves the weights where they already are.
    move_started = time.perf_counter()
    if _parameter_device(model) != device:
        model.to(device)
    device_move_seconds = time.perf_counter() - move_started
    prepare_started = time.perf_counter()
    waveform = _prepare_mixture(loaded, model)
    resample_seconds = time.perf_counter() - prepare_started
    separated, inference_seconds = _infer(model, waveform, device)
    return model, separated, inference_seconds, waveform, {
        "model_load_seconds": model_load_seconds,
        "device_move_seconds": device_move_seconds,
        "resample_seconds": resample_seconds,
        "inference_seconds": inference_seconds,
    }


def _parameter_device(model) -> str:
    """Device type of the loaded weights, such as cpu, mps, or cuda."""

    try:
        return next(model.parameters()).device.type
    except StopIteration:
        return ""


def load_htdemucs(model_name: str = MODEL_NAME):
    """Load and cache the requested pretrained Demucs model."""

    if model_name not in ("htdemucs", "htdemucs_ft"):
        raise SeparationError(f"Unsupported model: {model_name}")

    cached = _MODEL.get(model_name)
    if cached is not None:
        return cached

    try:
        from demucs.pretrained import get_model
    except ImportError as exc:
        raise SeparationError(
            "Demucs is not installed. Run scripts/setup.sh, then start the app again."
        ) from exc

    try:
        model = get_model(model_name)
    except Exception as exc:
        raise SeparationError(
            f"Could not load {model_name}: {_short_error(exc)}"
        ) from exc

    model.eval()
    _MODEL[model_name] = model
    return model


def _prepare_mixture(audio: LoadedAudio, model):
    """Match the waveform to the model's channel count and sample rate."""

    import torch
    from demucs.audio import convert_audio

    waveform = torch.from_numpy(np_audio(audio)).float()
    converted = convert_audio(
        waveform,
        audio.sample_rate,
        model.samplerate,
        model.audio_channels,
    )
    # Clone so a mono file expanded into a stereo view is safe to normalize,
    # and so a failed device attempt can be rebuilt from the original samples.
    return converted.contiguous().float().clone()


def np_audio(audio: LoadedAudio):
    import numpy as np

    # soundfile layout is (frames, channels). Demucs wants (channels, frames).
    return np.ascontiguousarray(audio.samples.T)


def apply_htdemucs(
    model,
    waveform,
    device: str,
    *,
    shifts: int,
    segment: float | None,
    apply_model_seconds: list | None = None,
):
    """One pretrained forward pass. The caller chooses shifts and segment.

    The app uses ``_shifts()`` and ``_segment()``. The benchmark passes the
    same explicit values for every device. Weight loading stays with the caller.
    When ``apply_model_seconds`` is a list, the app appends the ``apply_model``
    duration only. Loudness normalization and restoring the level stay outside
    that measurement.
    """

    import torch
    from demucs.apply import apply_model

    mixture = waveform.detach().float().clone()
    reference = mixture.mean(0)
    # The published CLI divides by the mixture standard deviation. The floor
    # avoids a divide-by-zero if a file passes the silence check but is constant.
    standard_deviation = reference.std().clamp(min=1e-8)
    centered = (mixture - reference.mean()) / standard_deviation
    started = time.perf_counter()
    with torch.no_grad():
        estimates = apply_model(
            model,
            centered[None],
            device=device,
            shifts=shifts,
            split=True,
            overlap=0.0,
            progress=False,
            num_workers=0,
            segment=segment,
        )[0]
    if apply_model_seconds is not None:
        apply_model_seconds.append(time.perf_counter() - started)
    estimates = estimates * standard_deviation + reference.mean()
    return {name: estimates[index] for index, name in enumerate(model.sources)}


def _infer(model, waveform, device: str):
    """Time apply_model only. Normalization stays outside that clock."""

    measured: list[float] = []
    separated = apply_htdemucs(
        model,
        waveform,
        device,
        shifts=_shifts(),
        segment=_segment(),
        apply_model_seconds=measured,
    )
    return separated, measured[0]


def _frames_from_model(separated: dict) -> dict[str, object]:
    try:
        return {name: tensor_to_frames(separated[name]) for name in STEM_NAMES}
    except KeyError as exc:
        found = ", ".join(sorted(str(name) for name in separated))
        raise OutputIntegrityError(
            f"htdemucs did not return the expected stems. It returned: {found}."
        ) from exc


def _check_duration(input_seconds: float, frames: int, sample_rate: int) -> None:
    output_seconds = frames / sample_rate
    if abs(output_seconds - input_seconds) > DURATION_TOLERANCE_SECONDS:
        raise OutputIntegrityError(
            f"Separated audio is {output_seconds:.2f} s, but the upload is {input_seconds:.2f} s."
        )


def _shifts() -> int:
    raw = os.environ.get("STEM_SHIFTS", "1")
    try:
        value = int(raw)
    except ValueError as exc:
        raise SeparationError("STEM_SHIFTS must be an integer.") from exc
    if value < 0:
        raise SeparationError("STEM_SHIFTS must be zero or greater.")
    return value


def _segment() -> float | None:
    raw = os.environ.get("STEM_SEGMENT", "").strip()
    if raw == "":
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise SeparationError("STEM_SEGMENT must be a number of seconds.") from exc
    if value <= 0 or value > MAX_SEGMENT_SECONDS:
        raise SeparationError(
            f"STEM_SEGMENT must be greater than 0 and at most {MAX_SEGMENT_SECONDS} seconds."
        )
    return value


def _format_relative(value: float | None) -> str:
    if value is None:
        return "undefined (mixture RMS is 0)"
    return f"{value:.6e}"


def _short_error(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    summary = text[0] if text else exc.__class__.__name__
    if len(summary) > 300:
        summary = summary[:300] + "..."
    return f"{exc.__class__.__name__}: {summary}"
