"""Available pretrained Demucs separation modes."""

MODEL_MODES = {
    "Fast": "htdemucs",
    "High Quality": "htdemucs_ft",
}

DEFAULT_MODE = "Fast"


def model_name_for_mode(mode: str) -> str:
    """Return the Demucs model name for a UI mode."""
    try:
        return MODEL_MODES[mode]
    except KeyError as exc:
        raise ValueError(
            f"Unknown separation mode: {mode}"
        ) from exc
