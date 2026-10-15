"""Limits and ZeroGPU wiring for a public Hugging Face Space.

The Mac app does not set SPACE_ID, so these limits stay off there. The
ZeroGPU decorator is a no-op when the spaces package is not installed.
A public Gradio share link is also off until STEM_SHARE is set.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

# Local use, the share link, and a Space share one tested ceiling:
# ten minutes and 250 MB. A longer song is rejected rather than described
# as unlimited. A 3.5-minute song has been separated on this Mac within that cap.
PUBLIC_MAX_DURATION_SECONDS = 600.0
PUBLIC_MAX_FILE_BYTES = 250 * 1024 * 1024
SHARE_MAX_DURATION_SECONDS = 600.0
SHARE_MAX_FILE_BYTES = 250 * 1024 * 1024
# How long a 30-second clip may hold a ZeroGPU slice, including the forward
# pass and the DSP that follows it. This is a ceiling, not a measured runtime.
ZERO_GPU_DURATION_SECONDS = 90
OUTPUT_MAX_AGE_SECONDS = 30 * 60
# Keep stems available while someone is listening and downloading.
# A 10-minute cap was deleting the WAV cache before a full song finished loading.
SHARE_OUTPUT_MAX_AGE_SECONDS = 2 * 60 * 60
_SHARE_TRUTHY = {"1", "true", "yes", "on"}


def on_huggingface_space() -> bool:
    """True only when Hugging Face sets SPACE_ID for this process."""

    return bool(os.environ.get("SPACE_ID", "").strip())


def public_share_enabled() -> bool:
    """True when this process should open a temporary Gradio share link.

    Local development stays private. Set STEM_SHARE=1 to publish.
    GRADIO_SHARE is not consulted, so an inherited value cannot turn sharing
    on without the upload limits that go with it.
    """

    return os.environ.get("STEM_SHARE", "").strip().lower() in _SHARE_TRUTHY


def prepare_public_share_dirs() -> Path:
    """Create empty temporary folders for one public process.

    Stems are written to the returned directory. Uploads are stored in
    GRADIO_TEMP_DIR. Neither path is the project's outputs folder, so older
    local separations are not part of the public file list.
    """

    output_dir = Path(tempfile.mkdtemp(prefix="music-stem-separator-"))
    upload_dir = Path(tempfile.mkdtemp(prefix="music-stem-separator-uploads-"))
    os.environ["STEM_OUTPUT_DIR"] = str(output_dir)
    os.environ["GRADIO_TEMP_DIR"] = str(upload_dir)
    return output_dir


def blocked_paths_for_share() -> list[str]:
    """Directories a public visitor must not be able to fetch.

    Gradio checks this list before allowed paths. It covers credentials,
    the git checkout, and this project's saved outputs and weights.
    """

    home = Path.home()
    root = Path(__file__).resolve().parent.parent
    paths = [
        home / ".ssh",
        home / ".aws",
        home / ".gnupg",
        home / ".kube",
        home / ".huggingface",
        home / ".cache" / "huggingface",
        home / ".config" / "gh",
        home / "Library" / "Keychains",
        root / ".git",
        root / "outputs",
        root / "uploads",
        root / "separated",
        root / "models",
        root / "weights",
        root / ".gradio",
    ]
    return [str(path) for path in paths]


def upload_is_contained(path: Path) -> bool:
    """True when a public upload resolves to a file inside Gradio's upload folder."""

    from gradio.utils import get_upload_folder

    try:
        resolved = path.resolve()
    except OSError:
        return False
    if not resolved.is_file():
        return False
    try:
        resolved.relative_to(Path(get_upload_folder()).resolve())
    except ValueError:
        return False
    return True


def bind_zero_gpu(function, *, duration: int = ZERO_GPU_DURATION_SECONDS):
    """Reserve the default ZeroGPU size around `function`.

    `size` stays at the decorator default, `large` (half an RTX Pro 6000).
    The full GPU (`xlarge`) costs twice the quota and is not requested.
    Outside a ZeroGPU Space, or when `spaces` is not installed, the function
    runs unchanged.
    """

    try:
        import spaces
    except ImportError:
        return function
    return spaces.GPU(duration=duration)(function)


def cleanup_old_outputs(directory: str | Path, *, max_age_seconds: int = OUTPUT_MAX_AGE_SECONDS) -> int:
    """Delete old generated run folders. Returns how many were removed.

    Fresh results stay so the person who just separated a clip can still
    download them. This does not delete files outside `directory`.
    """

    root = Path(directory)
    if not root.is_dir():
        return 0
    cutoff = time.time() - max_age_seconds
    removed = 0
    for child in root.iterdir():
        if not child.is_dir():
            continue
        try:
            older = child.stat().st_mtime < cutoff
        except OSError:
            continue
        if not older:
            continue
        shutil.rmtree(child, ignore_errors=True)
        removed += 1
    return removed
