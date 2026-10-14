"""Choose the processor that will run Demucs."""

from __future__ import annotations

import os
import platform
import sys


class DeviceError(RuntimeError):
    """The requested torch device is not available on this computer."""


def probe_accelerators() -> tuple[bool, bool]:
    """Return (cuda_available, mps_available) from the installed PyTorch."""

    import torch

    return bool(torch.cuda.is_available()), bool(torch.backends.mps.is_available())


def select_device(
    cuda_available: bool | None = None,
    mps_available: bool | None = None,
) -> str:
    """Pick cuda, mps, or cpu.

    STEM_DEVICE may be auto, cpu, mps, or cuda. auto prefers CUDA, then Apple
    MPS, then CPU. Passing the two flags avoids importing torch, which the
    unit tests use. When either flag is omitted, the installed PyTorch is asked.
    """

    if cuda_available is None or mps_available is None:
        cuda_available, mps_available = probe_accelerators()

    preference = os.environ.get("STEM_DEVICE", "auto").strip().lower() or "auto"
    if preference == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"
    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        if not cuda_available:
            raise DeviceError("STEM_DEVICE is set to cuda, but CUDA is not available.")
        return "cuda"
    if preference == "mps":
        if not mps_available:
            raise DeviceError(
                "STEM_DEVICE is set to mps, but MPS is not available on this Mac."
            )
        return "mps"
    raise DeviceError("STEM_DEVICE must be auto, cpu, mps, or cuda.")


def runtime_report() -> str:
    """Describe the Python build and the device Demucs will use."""

    lines = [
        f"Python: {sys.version.split()[0]} ({platform.machine()})",
        f"Platform: {platform.platform()}",
    ]
    try:
        import torch
    except ImportError:
        lines.append("PyTorch: not installed")
        return "\n".join(lines)

    cuda_available, mps_available = probe_accelerators()
    lines.extend(
        [
            f"PyTorch: {torch.__version__}",
            f"MPS built: {bool(torch.backends.mps.is_built())}",
            f"MPS available: {mps_available}",
            f"CUDA available: {cuda_available}",
            f"Selected device: {select_device(cuda_available, mps_available)}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    try:
        print(runtime_report())
    except DeviceError as exc:
        print(exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
