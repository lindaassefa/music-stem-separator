"""Device selection tests. The live check compares against this Mac's PyTorch."""

from __future__ import annotations

import pytest

from stem_ai.device import DeviceError, select_device


def test_auto_prefers_cuda_over_mps(monkeypatch):
    monkeypatch.delenv("STEM_DEVICE", raising=False)

    assert select_device(cuda_available=True, mps_available=True) == "cuda"


def test_auto_uses_mps_when_cuda_is_absent(monkeypatch):
    monkeypatch.delenv("STEM_DEVICE", raising=False)

    assert select_device(cuda_available=False, mps_available=True) == "mps"


def test_auto_uses_cpu_without_accelerators(monkeypatch):
    monkeypatch.delenv("STEM_DEVICE", raising=False)

    assert select_device(cuda_available=False, mps_available=False) == "cpu"


def test_cpu_override_ignores_accelerators(monkeypatch):
    monkeypatch.setenv("STEM_DEVICE", "cpu")

    assert select_device(cuda_available=True, mps_available=True) == "cpu"


def test_mps_request_fails_when_mps_is_unavailable(monkeypatch):
    monkeypatch.setenv("STEM_DEVICE", "mps")

    with pytest.raises(DeviceError, match="MPS is not available"):
        select_device(cuda_available=False, mps_available=False)


def test_unknown_device_name_is_rejected(monkeypatch):
    monkeypatch.setenv("STEM_DEVICE", "gpu")

    with pytest.raises(DeviceError, match="auto, cpu, mps, or cuda"):
        select_device(cuda_available=False, mps_available=False)


def test_live_selection_matches_installed_pytorch(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("STEM_DEVICE", raising=False)

    chosen = select_device()

    if torch.cuda.is_available():
        assert chosen == "cuda"
    elif torch.backends.mps.is_available():
        assert chosen == "mps"
    else:
        assert chosen == "cpu"
