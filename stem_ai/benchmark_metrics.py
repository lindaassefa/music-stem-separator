"""Latency, agreement, and file output for the Demucs device benchmark.

These helpers do not run the model. A missing standard deviation stays missing.
Real-time factor is inference latency divided by audio duration. A value below
1 means the timed forward pass finished faster than the clip plays.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stem_ai.export import STEM_NAMES

REAL_TIME_FACTOR = "inference_latency_seconds / audio_duration_seconds"


class BenchmarkMetricError(ValueError):
    """A latency list or a stem comparison could not be summarized."""


def latency_summary(latencies: list[float], audio_duration_seconds: float) -> dict:
    """Mean, sample standard deviation, and real-time factor for timed runs.

    Standard deviation uses the sample divisor (n - 1). One run has no sample
    standard deviation, so those fields are None rather than zero.
    """

    if audio_duration_seconds <= 0 or not np.isfinite(audio_duration_seconds):
        raise BenchmarkMetricError("Audio duration must be a positive finite number.")
    if len(latencies) < 1:
        raise BenchmarkMetricError("At least one timed run is required.")
    values = np.asarray(latencies, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise BenchmarkMetricError("Latencies must be finite and cannot be negative.")

    mean = float(values.mean())
    std = float(values.std(ddof=1)) if values.size >= 2 else None
    factors = values / audio_duration_seconds
    return {
        "runs": int(values.size),
        "latency_seconds": [float(item) for item in values],
        "latency_mean_seconds": mean,
        "latency_std_seconds": std,
        "latency_std_ddof": 1 if std is not None else None,
        "real_time_factor_per_run": [float(item) for item in factors],
        "real_time_factor_mean": float(factors.mean()),
        "real_time_factor_std": float(factors.std(ddof=1)) if std is not None else None,
        "real_time_factor_definition": REAL_TIME_FACTOR,
    }


def stem_agreement(reference: dict[str, np.ndarray], other: dict[str, np.ndarray]) -> dict:
    """Numerical difference between two sets of stems.

    This is agreement between two runs, not a source-separation quality score.
    ``relative_rmse`` is RMSE divided by the reference RMS. It is None when the
    reference is silent because that ratio is undefined.
    """

    per_stem = {}
    reference_parts = []
    other_parts = []
    for name in STEM_NAMES:
        if name not in reference or name not in other:
            raise BenchmarkMetricError(f"Both stem sets must include {name}.")
        left = np.asarray(reference[name], dtype=np.float64)
        right = np.asarray(other[name], dtype=np.float64)
        if left.shape != right.shape or left.size == 0:
            raise BenchmarkMetricError(f"{name} stems do not share a non-empty shape.")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise BenchmarkMetricError(f"{name} contains non-finite samples.")
        per_stem[name] = _error_block(left, right)
        reference_parts.append(left.reshape(-1))
        other_parts.append(right.reshape(-1))
    overall = _error_block(np.concatenate(reference_parts), np.concatenate(other_parts))
    return {"per_stem": per_stem, "overall": overall}


def write_json(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def write_csv(payload: dict, path: Path) -> Path:
    """Write timed runs, per-device summaries, and cross-device agreement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "kind",
        "device",
        "reference_device",
        "run_index",
        "latency_seconds",
        "real_time_factor",
        "latency_mean_seconds",
        "latency_std_seconds",
        "real_time_factor_mean",
        "real_time_factor_std",
        "process_rss_bytes_after_run",
        "mps_current_allocated_bytes",
        "mps_driver_allocated_bytes",
        "warmup_seconds",
        "device_transfer_seconds",
        "stem",
        "rmse",
        "relative_rmse",
        "peak_absolute_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for device, section in payload.get("devices", {}).items():
            for run in section.get("timed_runs", []):
                writer.writerow(
                    {
                        "kind": "timed_run",
                        "device": device,
                        "run_index": run.get("run_index"),
                        "latency_seconds": _cell(run.get("latency_seconds")),
                        "real_time_factor": _cell(run.get("real_time_factor")),
                        "process_rss_bytes_after_run": _cell(run.get("process_rss_bytes_after_run")),
                        "mps_current_allocated_bytes": _cell(run.get("mps_current_allocated_bytes")),
                        "mps_driver_allocated_bytes": _cell(run.get("mps_driver_allocated_bytes")),
                    }
                )
            summary = section.get("summary")
            if summary:
                writer.writerow(
                    {
                        "kind": "device_summary",
                        "device": device,
                        "latency_mean_seconds": _cell(summary.get("latency_mean_seconds")),
                        "latency_std_seconds": _cell(summary.get("latency_std_seconds")),
                        "real_time_factor_mean": _cell(summary.get("real_time_factor_mean")),
                        "real_time_factor_std": _cell(summary.get("real_time_factor_std")),
                        "warmup_seconds": _cell(section.get("warmup_seconds")),
                        "device_transfer_seconds": _cell(section.get("device_transfer_seconds")),
                    }
                )
        for comparison in payload.get("device_agreement", []):
            overall = comparison.get("overall", {})
            writer.writerow(
                {
                    "kind": "device_agreement",
                    "device": comparison.get("other_device"),
                    "reference_device": comparison.get("reference_device"),
                    "stem": "all",
                    "rmse": _cell(overall.get("rmse")),
                    "relative_rmse": _cell(overall.get("relative_rmse")),
                    "peak_absolute_error": _cell(overall.get("peak_absolute_error")),
                }
            )
            for stem, metrics in comparison.get("per_stem", {}).items():
                writer.writerow(
                    {
                        "kind": "device_agreement",
                        "device": comparison.get("other_device"),
                        "reference_device": comparison.get("reference_device"),
                        "stem": stem,
                        "rmse": _cell(metrics.get("rmse")),
                        "relative_rmse": _cell(metrics.get("relative_rmse")),
                        "peak_absolute_error": _cell(metrics.get("peak_absolute_error")),
                    }
                )
    return path


def write_latency_chart(payload: dict, path: Path) -> Path | None:
    """Bar chart of mean timed latency. Devices that did not finish are omitted."""

    names = []
    means = []
    errors = []
    for device, section in payload.get("devices", {}).items():
        summary = section.get("summary")
        if not summary or summary.get("latency_mean_seconds") is None:
            continue
        names.append(device)
        means.append(summary["latency_mean_seconds"])
        std = summary.get("latency_std_seconds")
        errors.append(0.0 if std is None else std)
    if not names:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.5, 4))
    try:
        figure.patch.set_facecolor("#1a1715")
        axis.set_facecolor("#1a1715")
        positions = np.arange(len(names))
        axis.bar(
            positions,
            means,
            yerr=errors,
            capsize=6,
            color=["#7c2a36", "#c4a574"][: len(names)],
            ecolor="#f4efe8",
        )
        axis.set_xticks(positions, names)
        axis.set_ylabel("Mean inference latency (s)")
        duration = payload.get("audio_duration_seconds")
        axis.set_title("htdemucs timed runs\nerror bars are the sample standard deviation")
        axis.tick_params(colors="#f4efe8")
        axis.yaxis.label.set_color("#f4efe8")
        axis.title.set_color("#f4efe8")
        axis.xaxis.label.set_color("#f4efe8")
        for spine in axis.spines.values():
            spine.set_color("#3a332f")
        if duration:
            axis.axhline(duration, color="#c9b8ae", linestyle="--", linewidth=1)
            axis.text(
                0.01,
                duration,
                "audio duration",
                transform=axis.get_yaxis_transform(),
                ha="left",
                va="bottom",
                color="#c9b8ae",
                fontsize=8,
            )
        figure.tight_layout()
        figure.savefig(path, dpi=120, facecolor=figure.get_facecolor())
    finally:
        plt.close(figure)
    return path


def _error_block(reference: np.ndarray, other: np.ndarray) -> dict:
    residual = other - reference
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    reference_rms = float(np.sqrt(np.mean(np.square(reference))))
    relative = None if reference_rms < 1e-12 else float(rmse / reference_rms)
    return {
        "rmse": rmse,
        "relative_rmse": relative,
        "peak_absolute_error": float(np.max(np.abs(residual))),
    }


def _cell(value) -> str:
    if value is None:
        return ""
    return str(value)


def format_saved_benchmark(payload: dict) -> str:
    """Render a benchmark file that already exists. Missing numbers stay missing."""

    lines = [
        "Saved measurements from one benchmark file. Nothing here is estimated.",
        "Collected on a local Mac, on CPU and Apple MPS. These are not Hugging Face server benchmarks.",
    ]
    audio = payload.get("audio_path") or "unknown file"
    lines.append(f"Audio: {audio}")
    duration = payload.get("audio_duration_seconds")
    if isinstance(duration, (int, float)) and duration > 0:
        lines.append(f"Clip duration: {float(duration):.2f} s")
    settings = payload.get("inference_settings") or {}
    lines.append(
        "Shared settings: "
        f"shifts={settings.get('shifts')}, overlap={settings.get('overlap')}, "
        f"split={settings.get('split')}, segment={settings.get('segment')}."
    )
    load_seconds = payload.get("model_load_seconds")
    if isinstance(load_seconds, (int, float)):
        lines.append(f"Model load, excluded from the inference mean: {float(load_seconds):.3f} s.")
    lines.append("")
    notes: list[str] = []
    lines.append("| Device | Mean latency (s) | Sample std (s) | Real-time factor |")
    lines.append("| --- | --- | --- | --- |")
    devices = payload.get("devices") or {}
    if not devices:
        lines.append("| none | not measured |  |  |")
    for name, section in devices.items():
        summary = (section or {}).get("summary") if isinstance(section, dict) else None
        if not summary or summary.get("latency_mean_seconds") is None:
            lines.append(f"| {name} | not measured |  |  |")
            error = (section or {}).get("error") if isinstance(section, dict) else None
            if error:
                notes.append(f"{name} was not completed: {error}")
            continue
        std = summary.get("latency_std_seconds")
        std_text = "not available" if std is None else f"{float(std):.3f}"
        lines.append(
            f"| {name} | {float(summary['latency_mean_seconds']):.3f} | "
            f"{std_text} | {float(summary['real_time_factor_mean']):.3f} |"
        )
    lines.append("")
    for skipped in payload.get("skipped_devices") or []:
        if isinstance(skipped, dict):
            notes.append(f"Skipped {skipped.get('device')}: {skipped.get('reason')}")
    for item in payload.get("device_agreement") or []:
        if not isinstance(item, dict):
            continue
        if item.get("reference_device") == "mps" and item.get("other_device") == "cpu":
            rmse = (item.get("overall") or {}).get("rmse")
            if isinstance(rmse, (int, float)):
                notes.append(
                    f"Last timed MPS and CPU stems differ by RMSE {float(rmse):.6e}. "
                    "That is device agreement, not separation accuracy."
                )
    notes.append(
        "Real-time factor is inference latency divided by the clip duration. "
        "Reconstruction RMSE elsewhere in this app is also not a quality score."
    )
    return "\n".join(lines + [""] + notes)


def load_saved_benchmark(directory: str | Path = "outputs/benchmarks") -> tuple[str, str | None]:
    """Read the newest `benchmark-*.json` file. An empty folder stays empty."""

    folder = Path(directory)
    files = sorted(folder.glob("benchmark-*.json")) if folder.is_dir() else []
    if not files:
        return (
            "No saved benchmark is in this folder yet. "
            "Numbers appear here only after a benchmark is run on a local Mac. "
            "They are not Hugging Face server measurements, and this panel does not estimate them.",
            None,
        )
    path = files[-1]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (f"The newest benchmark file could not be read ({exc.__class__.__name__}).", None)
    if not isinstance(payload, dict):
        return ("The newest benchmark file is not a JSON object.", None)
    chart = payload.get("chart_path")
    chart_path = None
    if isinstance(chart, str) and chart:
        candidate = Path(chart)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if candidate.is_file():
            chart_path = str(candidate.resolve())
    text = format_saved_benchmark(payload) + f"\n\nSource file: {path.name}."
    if isinstance(chart, str) and chart and chart_path is None:
        text += " The chart named in that file is not on disk, so no chart is shown."
    return text, chart_path
