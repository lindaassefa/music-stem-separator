"""Gradio dashboard for pretrained htdemucs stem separation."""

from __future__ import annotations

import logging
import os
from pathlib import Path

# Checked by Gradio on import. Keeps the local app from phoning home.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from stem_ai.analysis import AnalysisError
from stem_ai.benchmark_metrics import load_saved_benchmark
from stem_ai.deployment import (
    SHARE_OUTPUT_MAX_AGE_SECONDS,
    bind_zero_gpu,
    blocked_paths_for_share,
    cleanup_old_outputs,
    on_huggingface_space,
    prepare_public_share_dirs,
    public_share_enabled,
    upload_is_contained,
)
from stem_ai.device import DeviceError
from stem_ai.export import STEM_NAMES, OutputIntegrityError
from stem_ai.separation import SeparationError, load_htdemucs, separate
from stem_ai.validation import AudioValidationError, limits_text, load_audio

LABELS = {
    "vocals": "Vocals",
    "drums": "Drums",
    "bass": "Bass",
    "other": "Other",
}

HEADER = """
# MUSIC STEM SEPARATOR

## Pull a song apart.

Separate vocals, drums, bass, and everything else.
"""

PREPARING = "Preparing audio"

BENCHMARK_NOTE = """
### Benchmark

Any table or chart below is a local Mac measurement of CPU and Apple MPS. It is not a measurement of this server, and separating audio here does not rerun that comparison. CPU is included only when that Mac benchmark actually ran it.
"""

FOOTER = """
[Project source](https://github.com/lindaassefa/music-stem-separator)
"""

INK = "#10141E"
SLATE = "#20283A"
BLUE = "#526BFF"
BLUE_HOVER = "#6A7EFF"
LIME = "#D9FF66"
WHITE = "#F5F5F0"
MUTED = "#93A0B8"
LINE = "#2C3548"


def dashboard_theme() -> gr.themes.Base:
    """Ink field, slate cards, electric blue actions. Same in light and dark mode."""

    sans = ["Avenir Next", "Helvetica Neue", "Segoe UI", "sans-serif"]
    return gr.themes.Base(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=sans,
        font_mono=sans,
        radius_size="sm",
        text_size="md",
    ).set(
        body_background_fill=INK,
        body_background_fill_dark=INK,
        body_text_color=WHITE,
        body_text_color_dark=WHITE,
        body_text_color_subdued=MUTED,
        body_text_color_subdued_dark=MUTED,
        background_fill_primary=INK,
        background_fill_primary_dark=INK,
        background_fill_secondary=SLATE,
        background_fill_secondary_dark=SLATE,
        block_background_fill=INK,
        block_background_fill_dark=INK,
        block_border_color="transparent",
        block_border_color_dark="transparent",
        block_label_background_fill="transparent",
        block_label_background_fill_dark="transparent",
        block_label_text_color=MUTED,
        block_label_text_color_dark=MUTED,
        block_title_text_color=WHITE,
        block_title_text_color_dark=WHITE,
        block_info_text_color=MUTED,
        block_info_text_color_dark=MUTED,
        border_color_primary=LINE,
        border_color_primary_dark=LINE,
        border_color_accent=BLUE,
        border_color_accent_dark=BLUE,
        color_accent=BLUE,
        color_accent_soft=SLATE,
        color_accent_soft_dark=SLATE,
        link_text_color=WHITE,
        link_text_color_dark=WHITE,
        button_primary_background_fill=BLUE,
        button_primary_background_fill_dark=BLUE,
        button_primary_background_fill_hover=BLUE_HOVER,
        button_primary_background_fill_hover_dark=BLUE_HOVER,
        button_primary_text_color=INK,
        button_primary_text_color_dark=INK,
        button_primary_border_color=BLUE,
        button_primary_border_color_dark=BLUE,
        button_secondary_background_fill=SLATE,
        button_secondary_background_fill_dark=SLATE,
        button_secondary_background_fill_hover="#2A3348",
        button_secondary_background_fill_hover_dark="#2A3348",
        button_secondary_text_color=WHITE,
        button_secondary_text_color_dark=WHITE,
        button_secondary_border_color="transparent",
        button_secondary_border_color_dark="transparent",
        input_background_fill=INK,
        input_background_fill_dark=INK,
        input_border_color=LINE,
        input_border_color_dark=LINE,
        input_placeholder_color=MUTED,
        input_placeholder_color_dark=MUTED,
        panel_background_fill=INK,
        panel_background_fill_dark=INK,
        panel_border_color="transparent",
        panel_border_color_dark="transparent",
        block_shadow="none",
        block_shadow_dark="none",
        shadow_drop="none",
        shadow_drop_lg="none",
        loader_color=LIME,
        loader_color_dark=LIME,
        code_background_fill=INK,
        code_background_fill_dark=INK,
    )


PUBLIC_BENCHMARK_MESSAGE = (
    "This public link does not publish the saved Mac benchmark. "
    "Those CPU and Apple MPS measurements stay on the local machine. "
    "They are not a measurement of this shared server."
)

def saved_benchmark_for_display():
    """Local Mac benchmark for a private session. The public link omits the file."""

    if public_share_enabled():
        return PUBLIC_BENCHMARK_MESSAGE, None
    return load_saved_benchmark()


def serving_policy(output_dir: Path) -> dict:
    """File-serving and queue settings for this process.

    A share link may serve only `output_dir`, which main() sets to a fresh
    temporary folder. Credential directories and the project's saved outputs
    are blocked. Private local launches do not open a tunnel.
    """

    sharing = public_share_enabled()
    on_space = on_huggingface_space()
    public = sharing or on_space
    return {
        "share": sharing,
        "allowed_paths": [str(output_dir.resolve())],
        "blocked_paths": blocked_paths_for_share() if sharing else [],
        "max_file_size": "250mb" if (sharing or on_space) else None,
        "max_size": 1 if public else None,
        "run_history": False if sharing else None,
        "enable_monitoring": False if sharing else None,
        "mcp_server": False if sharing else None,
    }


def build_demo() -> gr.Blocks:
    public = public_share_enabled() or on_huggingface_space()
    cache_policy = (30, SHARE_OUTPUT_MAX_AGE_SECONDS) if public else None
    with gr.Blocks(title="Music Stem Separator", fill_width=True, delete_cache=cache_policy) as demo:
        gr.Markdown(HEADER, elem_classes=["masthead"])
        with gr.Column(elem_classes=["deck"]):
            gr.Markdown(limits_text(), elem_classes=["limits"])
            upload = gr.File(
                label="Song",
                show_label=False,
                file_count="single",
                file_types=[".wav", ".mp3"],
                type="filepath",
            )
            duration = gr.Markdown("", elem_classes=["duration"])
            separate_button = gr.Button("Separate Audio", variant="primary")
            status = gr.Markdown("", elem_classes=["status"])
        gr.Markdown("YOUR STEMS", elem_classes=["stems-heading"])
        players = [_stem_player(name) for name in STEM_NAMES]
        gr.Markdown(FOOTER, elem_classes=["fine-print"])
        upload.change(fn=describe_upload, inputs=upload, outputs=duration)
        separate_button.click(
            fn=start_separation,
            outputs=[separate_button, status, *players],
            concurrency_limit=1,
        ).then(
            fn=run_separation,
            inputs=upload,
            outputs=[*players, status],
            concurrency_limit=1,
        ).then(
            fn=enable_separate,
            outputs=separate_button,
        )
    return demo


def show_separating() -> str:
    """Shown while the upload is checked, before the model runs."""

    return PREPARING


def start_separation():
    """Block a second click and clear stems from the previous song."""

    return (
        gr.update(interactive=False),
        show_separating(),
        None,
        None,
        None,
        None,
    )


def enable_separate():
    return gr.update(interactive=True)


def describe_upload(upload) -> str:
    """Duration of the selected file, or the validation message if it cannot be read."""

    if not upload:
        return ""
    try:
        loaded = load_audio(_coerce_upload(upload))
    except AudioValidationError as exc:
        return str(exc)
    whole = int(round(loaded.duration_seconds))
    minutes, seconds = divmod(whole, 60)
    return f"Duration {minutes}:{seconds:02d}"


def _separate_checked(path: Path, loaded):
    """Run pretrained htdemucs. On ZeroGPU this is the GPU-reserved call.

    Plots and DSP measurements stay off this path. The model and the WAV
    export are unchanged. `analyze=True` remains available to the tests.
    """

    return separate(path, analyze=False, loaded=loaded)


separate_checked = bind_zero_gpu(_separate_checked)


def run_separation(upload):
    """Validate and separate one upload. Returns the four WAV paths and a status.

    This is one return, not a stream. The original working page filled the
    players the same way. Validation happens before the ZeroGPU reservation,
    so a bad upload does not hold a GPU. One separation runs at a time.
    """

    prepared = _prepare_upload(upload)
    if prepared[0] is None:
        return _idle(prepared[2])
    path, loaded, _error = prepared
    paths, error = _finish_separation(path, loaded)
    if error:
        return _idle(error)
    return paths + ("",)


def _prepare_upload(upload):
    if public_share_enabled():
        share_output = os.environ.get("STEM_OUTPUT_DIR")
        if share_output:
            cleanup_old_outputs(share_output, max_age_seconds=SHARE_OUTPUT_MAX_AGE_SECONDS)
    elif on_huggingface_space():
        cleanup_old_outputs(os.environ.get("STEM_OUTPUT_DIR", "outputs"))

    if upload is None:
        return None, None, "Upload a WAV or MP3 file."
    try:
        path = _coerce_upload(upload)
        if public_share_enabled() and not upload_is_contained(path):
            return None, None, "Upload a WAV or MP3 file."
        loaded = load_audio(path)
    except AudioValidationError as exc:
        return None, None, str(exc)
    return path, loaded, None


def _finish_separation(path: Path, loaded):
    try:
        result = separate_checked(path, loaded)
    except (SeparationError, OutputIntegrityError, DeviceError, AnalysisError) as exc:
        logging.getLogger("stem_ai.app").exception("Separation did not finish")
        return None, str(exc)
    except Exception as exc:
        logging.getLogger("stem_ai.app").exception("Unexpected separation failure")
        return None, f"Separation failed: {exc}"
    paths = result.paths()
    return tuple(str(paths[name]) for name in STEM_NAMES), None


def _idle(message: str):
    return (None, None, None, None, message)


def _stem_player(stem: str) -> gr.Audio:
    # The built-in player supplies play, pause, seek, the waveform, and the WAV download.
    with gr.Row(elem_classes=["stem-track"]):
        gr.Markdown(LABELS[stem], elem_classes=["track-name"])
        return gr.Audio(
            label=f"{LABELS[stem]} WAV",
            show_label=False,
            type="filepath",
            interactive=False,
            editable=False,
            buttons=["download"],
            waveform_options=gr.WaveformOptions(
                waveform_color=LIME,
                waveform_progress_color=BLUE,
            ),
            elem_classes=["stem-player"],
        )


def _coerce_upload(upload) -> Path:
    if isinstance(upload, dict):
        raw = upload.get("path") or upload.get("name")
    elif isinstance(upload, (str, Path)):
        raw = upload
    else:
        raw = getattr(upload, "path", None) or getattr(upload, "name", None)
    if not raw:
        raise AudioValidationError("Upload a WAV or MP3 file.")
    return Path(str(raw))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    on_space = on_huggingface_space()
    if public_share_enabled():
        output_dir = prepare_public_share_dirs()
    else:
        output_dir = Path(os.environ.get("STEM_OUTPUT_DIR", "outputs")).resolve()
    demo = build_demo()
    if on_space:
        # ZeroGPU's CUDA emulation accepts this placement before any visitor
        # request. The real GPU is held only inside the reserved call.
        load_htdemucs().cuda()
    port = int(os.environ.get("PORT" if on_space else "STEM_PORT", "7860"))
    css_path = Path(__file__).resolve().parent / "stem_ai" / "dashboard.css"
    policy = serving_policy(output_dir)
    demo.queue(default_concurrency_limit=1, max_size=policy["max_size"])
    demo.launch(
        server_name="0.0.0.0" if on_space else "127.0.0.1",
        server_port=port,
        inbrowser=False,
        share=policy["share"],
        max_file_size=policy["max_file_size"],
        allowed_paths=policy["allowed_paths"],
        blocked_paths=policy["blocked_paths"],
        run_history=policy["run_history"],
        enable_monitoring=policy["enable_monitoring"],
        mcp_server=policy["mcp_server"],
        theme=dashboard_theme(),
        css_paths=[css_path],
        footer_links=[],
    )


if __name__ == "__main__":
    main()
