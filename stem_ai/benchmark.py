"""Benchmark pretrained htdemucs on the processors available on this machine.

Model loading and the move onto a device are timed separately from the forward
pass. The benchmark forces shifts=0 so CPU and MPS see the same deterministic
settings. The Gradio app is unchanged and still uses its own shift setting.

Run from the repository root:

    python -m stem_ai.benchmark /path/to/song.wav --duration 10
"""

from __future__ import annotations

import argparse
import os
import platform
import resource
import subprocess
import time
from pathlib import Path

import numpy as np

from stem_ai.benchmark_metrics import (
    stem_agreement,
    latency_summary,
    write_csv,
    write_json,
    write_latency_chart,
)
from stem_ai.device import probe_accelerators
from stem_ai.export import tensor_to_frames
from stem_ai.separation import MODEL_NAME, _MODEL, apply_htdemucs, load_htdemucs, _prepare_mixture
from stem_ai.validation import SILENCE_PEAK_THRESHOLD, AudioValidationError, LoadedAudio, load_audio

BENCHMARK_SHIFTS = 0
BENCHMARK_SEGMENT = None
PROCESS_RSS_NOTE = (
    "process_rss_bytes_after_run is the current resident set of this whole "
    "process, sampled with ps after the run. It is not GPU memory and it is "
    "not isolated to one device. process_lifetime_peak_rss_bytes is the "
    "process high-water mark and is also not assigned to one device."
)
MPS_MEMORY_NOTE = (
    "mps_current_allocated_bytes and mps_driver_allocated_bytes are PyTorch "
    "MPS allocator counters sampled after synchronize. They are not total GPU "
    "memory and they are not a hardware peak. They are null on CPU."
)


class BenchmarkError(RuntimeError):
    """The requested clip or device comparison could not be completed."""


def load_clip(path: str | Path, duration_seconds: float) -> LoadedAudio:
    """Load the start of an existing file. Do not pad a short file."""

    if duration_seconds <= 0 or not np.isfinite(duration_seconds):
        raise BenchmarkError("Duration must be a positive number of seconds.")
    try:
        loaded = load_audio(path)
    except AudioValidationError as exc:
        raise BenchmarkError(str(exc)) from exc
    frames = int(round(duration_seconds * loaded.sample_rate))
    if frames > loaded.samples.shape[0]:
        raise BenchmarkError(
            f"Requested {duration_seconds:.2f} s, but the file is "
            f"{loaded.duration_seconds:.2f} s. The clip was not padded."
        )
    samples = np.ascontiguousarray(loaded.samples[:frames])
    peak = float(np.max(np.abs(samples)))
    if peak < SILENCE_PEAK_THRESHOLD:
        raise BenchmarkError("The requested clip is silent, so it was not benchmarked.")
    return LoadedAudio(
        samples=samples,
        sample_rate=loaded.sample_rate,
        duration_seconds=float(frames / loaded.sample_rate),
        source_format=loaded.source_format,
    )


def select_benchmark_devices(skip_cpu: bool, mps_available: bool) -> list[str]:
    """MPS when this PyTorch build can use it, then CPU unless skipped."""

    devices = []
    if mps_available:
        devices.append("mps")
    if not skip_cpu:
        devices.append("cpu")
    if not devices:
        raise BenchmarkError("No devices left to benchmark. CPU was skipped and MPS is unavailable.")
    return devices


def synchronize_device(device: str) -> str | None:
    """Wait for queued work. CPU has no device queue, so it returns None."""

    import torch

    mps = getattr(torch.backends, "mps", None)
    if (
        device == "mps"
        and mps is not None
        and mps.is_available()
        and hasattr(torch.mps, "synchronize")
    ):
        torch.mps.synchronize()
        return "torch.mps.synchronize"
    cuda = getattr(torch.backends, "cuda", None)
    if device == "cuda" and cuda is not None and cuda.is_available() and hasattr(torch.cuda, "synchronize"):
        torch.cuda.synchronize()
        return "torch.cuda.synchronize"
    return None


def process_rss_bytes() -> int | None:
    """Current process resident size in bytes, or None if ps cannot be read.

    On macOS and Linux, `ps -o rss=` is kilobytes. The value stored is bytes.
    """

    try:
        raw = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
        ).strip()
        return int(raw) * 1024
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def process_lifetime_peak_rss_bytes() -> dict:
    """High-water resident size for this process, with the OS unit made explicit."""

    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    system = platform.system()
    if system == "Darwin":
        return {
            "bytes": raw,
            "source": "resource.getrusage.ru_maxrss",
            "unit_note": "macOS reports ru_maxrss in bytes.",
        }
    if system == "Linux":
        return {
            "bytes": raw * 1024,
            "source": "resource.getrusage.ru_maxrss",
            "unit_note": "Linux reports ru_maxrss in kilobytes. This value was multiplied by 1024.",
        }
    return {
        "bytes": None,
        "raw_ru_maxrss": raw,
        "source": "resource.getrusage.ru_maxrss",
        "unit_note": "The unit of ru_maxrss is not known for this OS, so it was not converted.",
    }


def mps_allocator_bytes(device: str) -> dict[str, int | None]:
    """Sample PyTorch's MPS counters. Unavailable calls stay None."""

    empty = {"mps_current_allocated_bytes": None, "mps_driver_allocated_bytes": None}
    if device != "mps":
        return empty
    import torch

    if not hasattr(torch.mps, "current_allocated_memory"):
        return empty
    try:
        allocated = int(torch.mps.current_allocated_memory())
        driver = (
            int(torch.mps.driver_allocated_memory())
            if hasattr(torch.mps, "driver_allocated_memory")
            else None
        )
    except Exception:
        return empty
    return {
        "mps_current_allocated_bytes": allocated,
        "mps_driver_allocated_bytes": driver,
    }


def run_benchmark(
    path: str | Path,
    duration_seconds: float,
    *,
    runs: int = 3,
    skip_cpu: bool = False,
    output_dir: str | Path = "outputs/benchmarks",
) -> dict:
    """Warm up once and time `runs` forward passes on each selected device."""

    if runs < 1:
        raise BenchmarkError("The number of timed runs must be at least 1.")
    clip = load_clip(path, duration_seconds)
    _cuda, mps_available = probe_accelerators()
    devices = select_benchmark_devices(skip_cpu, mps_available)
    already_loaded = MODEL_NAME in _MODEL
    load_started = time.perf_counter()
    model = load_htdemucs()
    model_load_seconds = time.perf_counter() - load_started
    waveform = _prepare_mixture(clip, model)

    payload = {
        "model": MODEL_NAME,
        "model_note": "Published htdemucs weights. This benchmark does not train Demucs.",
        "audio_path": str(Path(path).resolve()),
        "requested_duration_seconds": duration_seconds,
        "audio_duration_seconds": clip.duration_seconds,
        "sample_rate_hz": clip.sample_rate,
        "source_format": clip.source_format,
        "inference_settings": {
            "shifts": BENCHMARK_SHIFTS,
            "segment": BENCHMARK_SEGMENT,
            "overlap": 0.25,
            "split": True,
            "num_workers": 0,
            "note": (
                "shifts is 0 so every device and every timed run uses one "
                "unshifted forward pass. The app default can still be 1."
            ),
        },
        "model_load_seconds": model_load_seconds,
        "model_was_already_in_memory": already_loaded,
        "timing_note": (
            "Each timed run calls torch.mps.synchronize before the clock starts "
            "and again before it stops, when that function exists. The clock "
            "covers one apply_htdemucs call, including loudness normalization "
            "and restoring the original level, because the device sync has to "
            "surround that call. The app's displayed inference time still wraps "
            "only apply_model. Model load and model.to(device) are recorded "
            "separately and are not part of the timed runs."
        ),
        "memory_note": PROCESS_RSS_NOTE + " " + MPS_MEMORY_NOTE,
        "devices": {},
        "device_agreement": [],
        "process_lifetime_peak_rss": None,
        "skipped_devices": [],
    }
    if skip_cpu:
        payload["skipped_devices"].append({"device": "cpu", "reason": "Skipped with --skip-cpu."})
    if not mps_available:
        payload["skipped_devices"].append({"device": "mps", "reason": "MPS is not available in this PyTorch build."})

    stems_for_agreement: dict[str, dict[str, np.ndarray]] = {}
    within_device: dict[str, list[dict]] = {}
    for device in devices:
        print(f"device {device}: transferring model", flush=True)
        section = _benchmark_device(model, waveform, device, runs, clip.duration_seconds)
        payload["devices"][device] = section
        if section.get("agreement_stems") is not None:
            stems_for_agreement[device] = section.pop("agreement_stems")
            within_device[device] = section.pop("within_device_runs")
        else:
            section.pop("agreement_stems", None)
            section.pop("within_device_runs", None)
        summary = section.get("summary")
        if summary:
            print(
                f"device {device}: mean {summary['latency_mean_seconds']:.2f} s, "
                f"real-time factor {summary['real_time_factor_mean']:.3f}",
                flush=True,
            )
        elif section.get("error"):
            print(f"device {device}: failed: {section['error']}", flush=True)

    if "mps" in stems_for_agreement and "cpu" in stems_for_agreement:
        agreement = stem_agreement(stems_for_agreement["mps"], stems_for_agreement["cpu"])
        agreement["reference_device"] = "mps"
        agreement["other_device"] = "cpu"
        agreement["note"] = (
            "Numerical difference between the last timed CPU run and the last "
            "timed MPS run. This is not a separation-quality score."
        )
        payload["device_agreement"].append(agreement)
    for device, run_stems in within_device.items():
        if len(run_stems) < 2:
            continue
        internal = stem_agreement(run_stems[0], run_stems[-1])
        internal["reference_device"] = f"{device} timed run 1"
        internal["other_device"] = f"{device} timed run {len(run_stems)}"
        internal["note"] = "Difference between the first and last timed run on the same device."
        payload["device_agreement"].append(internal)

    payload["process_lifetime_peak_rss"] = process_lifetime_peak_rss_bytes()
    destination = Path(output_dir)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    payload["json_path"] = str(write_json(payload, destination / f"benchmark-{stamp}.json"))
    payload["csv_path"] = str(write_csv(payload, destination / f"benchmark-{stamp}.csv"))
    chart = write_latency_chart(payload, destination / f"benchmark-{stamp}.png")
    payload["chart_path"] = None if chart is None else str(chart)
    write_json(payload, Path(payload["json_path"]))
    return payload


def _benchmark_device(model, waveform, device: str, runs: int, duration_seconds: float) -> dict:
    sync_name = None
    try:
        transfer_started = time.perf_counter()
        # Drain MPS before a later CPU pass so leftover GPU work is not timed as CPU.
        if device != "mps":
            synchronize_device("mps")
        model.to(device)
        sync_name = synchronize_device(device)
        transfer_seconds = time.perf_counter() - transfer_started
        print(f"device {device}: warm-up", flush=True)
        synchronize_device(device)
        warmup_started = time.perf_counter()
        apply_htdemucs(
            model,
            waveform,
            device,
            shifts=BENCHMARK_SHIFTS,
            segment=BENCHMARK_SEGMENT,
        )
        synchronize_device(device)
        warmup_seconds = time.perf_counter() - warmup_started
    except Exception as exc:
        return {
            "error": f"{exc.__class__.__name__}: {exc}",
            "summary": None,
            "timed_runs": [],
            "agreement_stems": None,
            "within_device_runs": None,
        }

    timed_runs = []
    run_stems = []
    rss_samples = []
    allocated_samples = []
    driver_samples = []
    try:
        for index in range(1, runs + 1):
            print(f"device {device}: timed run {index}/{runs}", flush=True)
            synchronize_device(device)
            started = time.perf_counter()
            separated = apply_htdemucs(
                model,
                waveform,
                device,
                shifts=BENCHMARK_SHIFTS,
                segment=BENCHMARK_SEGMENT,
            )
            synchronize_device(device)
            latency = time.perf_counter() - started
            frames = {name: tensor_to_frames(tensor) for name, tensor in separated.items()}
            run_stems.append(frames)
            rss = process_rss_bytes()
            allocator = mps_allocator_bytes(device)
            if rss is not None:
                rss_samples.append(rss)
            if allocator["mps_current_allocated_bytes"] is not None:
                allocated_samples.append(allocator["mps_current_allocated_bytes"])
            if allocator["mps_driver_allocated_bytes"] is not None:
                driver_samples.append(allocator["mps_driver_allocated_bytes"])
            timed_runs.append(
                {
                    "run_index": index,
                    "latency_seconds": latency,
                    "real_time_factor": latency / duration_seconds,
                    "process_rss_bytes_after_run": rss,
                    **allocator,
                }
            )
    except Exception as exc:
        return {
            "error": f"{exc.__class__.__name__}: {exc}",
            "device_transfer_seconds": transfer_seconds,
            "warmup_seconds": warmup_seconds,
            "synchronize": sync_name,
            "summary": None,
            "timed_runs": timed_runs,
            "agreement_stems": None,
            "within_device_runs": None,
        }

    summary = latency_summary([run["latency_seconds"] for run in timed_runs], duration_seconds)
    return {
        "error": None,
        "device_transfer_seconds": transfer_seconds,
        "warmup_seconds": warmup_seconds,
        "synchronize": sync_name,
        "process_rss_bytes_max_sampled_after_runs": max(rss_samples) if rss_samples else None,
        "mps_current_allocated_bytes_max_sampled": max(allocated_samples) if allocated_samples else None,
        "mps_driver_allocated_bytes_max_sampled": max(driver_samples) if driver_samples else None,
        "summary": summary,
        "timed_runs": timed_runs,
        "agreement_stems": run_stems[-1],
        "within_device_runs": run_stems,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pretrained htdemucs on a local audio file.")
    parser.add_argument("audio", help="Existing WAV or MP3 file. The clip is taken from the start.")
    parser.add_argument("--duration", type=float, required=True, help="Clip length in seconds.")
    parser.add_argument("--runs", type=int, default=3, help="Timed runs after the warm-up. Default: 3.")
    parser.add_argument(
        "--skip-cpu",
        action="store_true",
        help="Do not run the CPU passes. Use this when a CPU benchmark would take too long.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/benchmarks",
        help="Directory for the JSON, CSV, and chart.",
    )
    args = parser.parse_args()
    payload = run_benchmark(
        args.audio,
        args.duration,
        runs=args.runs,
        skip_cpu=args.skip_cpu,
        output_dir=args.output_dir,
    )
    print(f"JSON: {payload['json_path']}")
    print(f"CSV: {payload['csv_path']}")
    print(f"Chart: {payload['chart_path']}")


if __name__ == "__main__":
    main()
