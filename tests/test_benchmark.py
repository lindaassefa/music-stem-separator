"""Metric and serialization tests. These do not run Demucs."""

from __future__ import annotations

import json

import numpy as np
import pytest

from stem_ai.benchmark_metrics import (
    BenchmarkMetricError,
    latency_summary,
    stem_agreement,
    write_csv,
    write_json,
    write_latency_chart,
)
from stem_ai.export import STEM_NAMES


def test_three_latencies_have_sample_standard_deviation():
    summary = latency_summary([1.0, 2.0, 3.0], audio_duration_seconds=2.0)

    assert summary["latency_mean_seconds"] == pytest.approx(2.0)
    assert summary["latency_std_seconds"] == pytest.approx(1.0)
    assert summary["latency_std_ddof"] == 1
    assert summary["real_time_factor_mean"] == pytest.approx(1.0)
    assert summary["real_time_factor_std"] == pytest.approx(0.5)
    assert summary["real_time_factor_per_run"] == pytest.approx([0.5, 1.0, 1.5])


def test_one_run_does_not_invent_a_standard_deviation():
    summary = latency_summary([1.5], audio_duration_seconds=3.0)

    assert summary["latency_mean_seconds"] == pytest.approx(1.5)
    assert summary["latency_std_seconds"] is None
    assert summary["real_time_factor_mean"] == pytest.approx(0.5)
    assert summary["real_time_factor_std"] is None


def test_invalid_latency_or_duration_is_rejected():
    with pytest.raises(BenchmarkMetricError):
        latency_summary([], 1.0)
    with pytest.raises(BenchmarkMetricError):
        latency_summary([1.0], 0.0)
    with pytest.raises(BenchmarkMetricError):
        latency_summary([-0.1], 1.0)


def test_identical_stems_have_zero_agreement_error():
    stems = {name: np.full((8, 2), index + 0.25) for index, name in enumerate(STEM_NAMES)}

    agreement = stem_agreement(stems, {name: array.copy() for name, array in stems.items()})

    assert agreement["overall"]["rmse"] == pytest.approx(0.0, abs=1e-12)
    assert agreement["overall"]["peak_absolute_error"] == pytest.approx(0.0, abs=1e-12)
    assert agreement["per_stem"]["vocals"]["relative_rmse"] == pytest.approx(0.0, abs=1e-12)


def test_constant_stem_offset_is_the_rmse():
    reference = {name: np.zeros((4, 1)) for name in STEM_NAMES}
    other = {name: np.zeros((4, 1)) for name in STEM_NAMES}
    reference["bass"] = np.ones((4, 1))
    other["bass"] = np.full((4, 1), 1.25)

    agreement = stem_agreement(reference, other)

    assert agreement["per_stem"]["bass"]["rmse"] == pytest.approx(0.25)
    assert agreement["per_stem"]["bass"]["peak_absolute_error"] == pytest.approx(0.25)
    assert agreement["per_stem"]["bass"]["relative_rmse"] == pytest.approx(0.25)
    assert agreement["per_stem"]["vocals"]["rmse"] == pytest.approx(0.0, abs=1e-12)
    assert agreement["per_stem"]["vocals"]["relative_rmse"] is None


def test_json_and_csv_round_trip_without_filling_missing_numbers(tmp_path):
    payload = {
        "audio_duration_seconds": 4.0,
        "devices": {
            "mps": {
                "summary": latency_summary([2.0, 2.0, 2.0], 4.0),
                "timed_runs": [
                    {
                        "run_index": 1,
                        "latency_seconds": 2.0,
                        "real_time_factor": 0.5,
                        "process_rss_bytes_after_run": 1024,
                        "mps_current_allocated_bytes": 2048,
                        "mps_driver_allocated_bytes": None,
                    }
                ],
            },
            "cpu": {"summary": None, "timed_runs": [], "error": "not run"},
        },
        "device_agreement": [
            {
                "reference_device": "mps",
                "other_device": "cpu",
                "overall": {"rmse": 0.125, "relative_rmse": None, "peak_absolute_error": 0.5},
                "per_stem": {
                    "vocals": {"rmse": 0.125, "relative_rmse": 0.25, "peak_absolute_error": 0.5}
                },
            }
        ],
    }

    json_path = write_json(payload, tmp_path / "benchmark.json")
    csv_path = write_csv(payload, tmp_path / "benchmark.csv")
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    csv_text = csv_path.read_text(encoding="utf-8")

    assert loaded["devices"]["cpu"]["summary"] is None
    assert loaded["devices"]["mps"]["summary"]["latency_std_seconds"] == pytest.approx(0.0)
    assert "not run" not in csv_text
    assert "timed_run,mps,,1,2.0,0.5" in csv_text.replace(" ", "")
    assert ",," in csv_text
    assert "device_agreement,cpu,mps" in csv_text.replace(" ", "")


def test_device_selection_does_not_invent_a_missing_processor():
    from stem_ai.benchmark import BenchmarkError, select_benchmark_devices

    assert select_benchmark_devices(skip_cpu=False, mps_available=True) == ["mps", "cpu"]
    assert select_benchmark_devices(skip_cpu=True, mps_available=True) == ["mps"]
    assert select_benchmark_devices(skip_cpu=False, mps_available=False) == ["cpu"]
    with pytest.raises(BenchmarkError):
        select_benchmark_devices(skip_cpu=True, mps_available=False)


def test_a_short_file_is_not_padded(tmp_path):
    import soundfile as sf

    from stem_ai.benchmark import BenchmarkError, load_clip

    path = tmp_path / "one_second.wav"
    sf.write(path, np.full((44100, 1), 0.2, dtype=np.float32), 44100)

    with pytest.raises(BenchmarkError, match="not padded"):
        load_clip(path, 2.0)

    clip = load_clip(path, 0.5)
    assert clip.samples.shape == (22050, 1)
    assert clip.duration_seconds == pytest.approx(0.5)


def test_saved_benchmark_text_keeps_an_unmeasured_device_blank():
    from stem_ai.benchmark_metrics import format_saved_benchmark

    text = format_saved_benchmark(
        {
            "audio_path": "/tmp/clip.wav",
            "audio_duration_seconds": 10.0,
            "model_load_seconds": 0.5,
            "inference_settings": {"shifts": 0, "overlap": 0.25, "split": True, "segment": None},
            "devices": {
                "mps": {"summary": latency_summary([4.0, 4.2, 3.8], 10.0)},
                "cpu": {"summary": None, "error": "Skipped with --skip-cpu."},
            },
            "skipped_devices": [{"device": "cpu", "reason": "Skipped with --skip-cpu."}],
        }
    )

    assert "4.000" in text
    assert "| cpu | not measured |" in text
    assert "Skipped cpu: Skipped with --skip-cpu." in text
    assert "0.000" not in text.split("cpu")[1]


def test_missing_benchmark_folder_does_not_invent_a_chart(tmp_path):
    from stem_ai.benchmark_metrics import load_saved_benchmark

    text, chart = load_saved_benchmark(tmp_path / "missing")

    assert chart is None
    assert "does not estimate" in text


def test_saved_benchmark_loads_the_chart_that_exists(tmp_path):
    from stem_ai.benchmark_metrics import load_saved_benchmark

    chart = tmp_path / "latency.png"
    chart.write_bytes(b"\x89PNG\r\n\x1a\n")
    payload = {
        "audio_duration_seconds": 10,
        "model_load_seconds": 0.2,
        "inference_settings": {"shifts": 0, "overlap": 0.25, "split": True, "segment": None},
        "devices": {"mps": {"summary": latency_summary([2.0, 2.0], 10.0)}},
        "chart_path": str(chart),
    }
    (tmp_path / "benchmark-20261014-010101.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    text, loaded = load_saved_benchmark(tmp_path)

    assert loaded == str(chart.resolve())
    assert "2.000" in text
    assert "benchmark-20261014-010101.json" in text


def test_chart_is_a_png_and_skips_a_device_without_a_summary(tmp_path):
    payload = {
        "audio_duration_seconds": 5.0,
        "devices": {
            "mps": {"summary": latency_summary([1.0, 1.2, 1.4], 5.0)},
            "cpu": {"summary": None},
        },
    }

    chart = write_latency_chart(payload, tmp_path / "latency.png")
    missing = write_latency_chart({"devices": {"cpu": {"summary": None}}}, tmp_path / "none.png")

    assert chart is not None
    assert chart.read_bytes().startswith(b"\x89PNG")
    assert missing is None
