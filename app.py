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
from stem_ai.device import DeviceError
from stem_ai.export import STEM_NAMES, OutputIntegrityError
from stem_ai.separation import SeparationError, SeparationResult, format_report, separate
from stem_ai.validation import AudioValidationError

LABELS = {
    "vocals": "Vocals",
    "drums": "Drums",
    "bass": "Bass",
    "other": "Other",
}

HEADER = """
# STEM

Neural Audio Intelligence

A mix goes in. Four stems come out: vocals, drums, bass, and other. The split uses Meta's published Hybrid Transformer Demucs weights, `htdemucs`. This application runs that model. It does not train it.
"""

UPLOAD_NOTE = """
WAV or MP3. Empty files, silence, and unreadable audio stop here, before the weights load.
"""

SEPARATING = (
    "Separating this mix with pretrained htdemucs. "
    "The clock for inference starts after the weights are on the processor. "
    "CPU can take longer than the clip."
)

ANALYSIS_NOTE = """
### Audio analysis

RMS, spectral centroid, spectral bandwidth, and zero-crossing rate are librosa and NumPy measurements on the mono average of each stem. They are not neural-network outputs.

Each spectrogram is a log-frequency view, in decibels relative to that image's own peak, so the colors do not rank loudness across stems. RMS does.

Reconstruction RMSE is the difference between the mixture Demucs received and the sum of the four stems, taken before 16-bit clipping. It is not a source-separation accuracy metric. A small residual only means the stems add back toward the mix.
"""

BENCHMARK_NOTE = """
### Benchmark

The table and chart below are the newest file in `outputs/benchmarks`. Separating a song on this page does not rerun that comparison. CPU is included only when that benchmark actually ran it.
"""

FOOTER = """
Pretrained model: Hybrid Transformer Demucs (`htdemucs`), Meta Demucs. Rouard, Massa, and Défossez, ICASSP 2023. Weights are loaded for inference only.
"""

INK = "#0e0d0c"
CHARCOAL = "#1a1715"
PANEL = "#241f1c"
LINE = "#3a332f"
IVORY = "#f4efe8"
MUTED = "#c9b8ae"
BURGUNDY = "#7c2a36"
BURGUNDY_HOVER = "#9a3846"


def dashboard_theme() -> gr.themes.Base:
    """Near-black field, charcoal panels, burgundy accent. Same in light and dark mode."""

    return gr.themes.Base(
        primary_hue="red",
        secondary_hue="stone",
        neutral_hue="stone",
        font=["Avenir Next", "Segoe UI", "sans-serif"],
        font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
        radius_size="sm",
        text_size="md",
    ).set(
        body_background_fill=INK,
        body_background_fill_dark=INK,
        body_text_color=IVORY,
        body_text_color_dark=IVORY,
        body_text_color_subdued=MUTED,
        body_text_color_subdued_dark=MUTED,
        background_fill_primary=CHARCOAL,
        background_fill_primary_dark=CHARCOAL,
        background_fill_secondary=PANEL,
        background_fill_secondary_dark=PANEL,
        block_background_fill=CHARCOAL,
        block_background_fill_dark=CHARCOAL,
        block_border_color=LINE,
        block_border_color_dark=LINE,
        block_label_background_fill=PANEL,
        block_label_background_fill_dark=PANEL,
        block_label_text_color=IVORY,
        block_label_text_color_dark=IVORY,
        block_title_text_color=IVORY,
        block_title_text_color_dark=IVORY,
        block_info_text_color=MUTED,
        block_info_text_color_dark=MUTED,
        border_color_primary=LINE,
        border_color_primary_dark=LINE,
        border_color_accent=BURGUNDY,
        border_color_accent_dark=BURGUNDY,
        color_accent=BURGUNDY,
        color_accent_soft="#3a2226",
        color_accent_soft_dark="#3a2226",
        link_text_color="#e7c4c8",
        link_text_color_dark="#e7c4c8",
        button_primary_background_fill=BURGUNDY,
        button_primary_background_fill_dark=BURGUNDY,
        button_primary_background_fill_hover=BURGUNDY_HOVER,
        button_primary_background_fill_hover_dark=BURGUNDY_HOVER,
        button_primary_text_color=IVORY,
        button_primary_text_color_dark=IVORY,
        button_primary_border_color=BURGUNDY,
        button_primary_border_color_dark=BURGUNDY,
        button_secondary_background_fill=PANEL,
        button_secondary_background_fill_dark=PANEL,
        button_secondary_background_fill_hover="#2e2926",
        button_secondary_background_fill_hover_dark="#2e2926",
        button_secondary_text_color=IVORY,
        button_secondary_text_color_dark=IVORY,
        button_secondary_border_color=LINE,
        button_secondary_border_color_dark=LINE,
        input_background_fill="#141210",
        input_background_fill_dark="#141210",
        input_border_color=LINE,
        input_border_color_dark=LINE,
        input_placeholder_color=MUTED,
        input_placeholder_color_dark=MUTED,
        table_text_color=IVORY,
        table_text_color_dark=IVORY,
        table_border_color=LINE,
        table_border_color_dark=LINE,
        table_even_background_fill=CHARCOAL,
        table_even_background_fill_dark=CHARCOAL,
        table_odd_background_fill=PANEL,
        table_odd_background_fill_dark=PANEL,
        panel_background_fill=CHARCOAL,
        panel_background_fill_dark=CHARCOAL,
        panel_border_color=LINE,
        panel_border_color_dark=LINE,
        block_shadow="none",
        block_shadow_dark="none",
        shadow_drop="none",
        shadow_drop_lg="none",
        loader_color=BURGUNDY,
        loader_color_dark=BURGUNDY,
        code_background_fill="#141210",
        code_background_fill_dark="#141210",
    )


def build_demo() -> gr.Blocks:
    benchmark_text, benchmark_chart = load_saved_benchmark()
    with gr.Blocks(title="STEM — Neural Audio Intelligence", fill_width=True) as demo:
        gr.Markdown(HEADER, elem_classes=["masthead"])
        with gr.Row():
            with gr.Column(scale=3):
                upload = gr.File(
                    label="Audio",
                    file_types=[".wav", ".mp3"],
                    type="filepath",
                )
            with gr.Column(scale=2):
                gr.Markdown(UPLOAD_NOTE, elem_classes=["section-label"])
                separate_button = gr.Button("Separate Audio", variant="primary", size="lg")
        report = gr.Textbox(
            label="This run",
            lines=12,
            interactive=False,
            placeholder="Duration, device, inference time, and the reconstruction residual appear after a separation.",
        )
        gr.Markdown(
            "### Stems\n\nPlayback and a WAV download sit with each stem. Waveforms and spectrograms fill in after a successful run. A missing file, silence, or a failed pass leaves these panels empty and writes the reason above.",
            elem_classes=["section-label"],
        )
        panels = {}
        with gr.Row():
            for name in ("vocals", "drums"):
                panels[name] = _stem_panel(name)
        with gr.Row():
            for name in ("bass", "other"):
                panels[name] = _stem_panel(name)
        players = [panels[name][0] for name in STEM_NAMES]
        images = []
        for name in STEM_NAMES:
            images.extend(panels[name][1:])
        gr.Markdown(ANALYSIS_NOTE, elem_classes=["section-label"])
        table = gr.Dataframe(
            headers=[
                "Stem",
                "RMS",
                "Spectral centroid (Hz)",
                "Spectral bandwidth (Hz)",
                "Zero-crossing rate",
            ],
            datatype=["str", "number", "number", "number", "number"],
            interactive=False,
            label="Stem metrics",
            wrap=True,
        )
        gr.Markdown(BENCHMARK_NOTE, elem_classes=["section-label"])
        benchmark_summary = gr.Markdown(benchmark_text)
        benchmark_image = gr.Image(
            value=benchmark_chart,
            label="Latency comparison",
            type="filepath",
            interactive=False,
            sources=[],
            buttons=["download", "fullscreen"],
        )
        refresh_benchmark = gr.Button("Reload saved benchmark", variant="secondary")
        gr.Markdown(FOOTER, elem_classes=["fine-print"])
        separate_button.click(
            fn=show_separating,
            outputs=report,
            concurrency_limit=1,
        ).then(
            fn=run_separation,
            inputs=upload,
            outputs=players + [report, table, *images],
            concurrency_limit=1,
        )
        refresh_benchmark.click(
            fn=load_saved_benchmark,
            outputs=[benchmark_summary, benchmark_image],
        )
    return demo


def show_separating() -> str:
    """Shown while htdemucs is running. The measured times replace it."""

    return SEPARATING


def run_separation(upload):
    """Validate and separate one upload. Errors stay in the details box."""

    if upload is None:
        return _idle("Upload a WAV or MP3 file.")

    try:
        result = separate(_coerce_upload(upload))
    except AudioValidationError as exc:
        return _idle(str(exc))
    except (SeparationError, OutputIntegrityError, DeviceError, AnalysisError) as exc:
        logging.getLogger("stem_ai.app").exception("Separation did not finish")
        return _idle(str(exc))
    except Exception as exc:
        logging.getLogger("stem_ai.app").exception("Unexpected separation failure")
        return _idle(f"Separation failed: {exc}")

    paths = result.paths()
    return (
        tuple(str(paths[name]) for name in STEM_NAMES)
        + (format_report(result), _measurement_rows(result))
        + _figure_paths(result)
    )


def _idle(message: str):
    return (None, None, None, None, message, None, *([None] * (len(STEM_NAMES) * 2)))


def _measurement_rows(result: SeparationResult):
    if not result.measurements:
        return None
    by_name = {item.stem: item for item in result.measurements}
    return [
        [
            LABELS[name],
            by_name[name].rms,
            by_name[name].spectral_centroid_hz,
            by_name[name].spectral_bandwidth_hz,
            by_name[name].zero_crossing_rate,
        ]
        for name in STEM_NAMES
    ]


def _figure_paths(result: SeparationResult):
    by_name = {item.stem: item for item in result.measurements}
    paths = []
    for name in STEM_NAMES:
        item = by_name.get(name)
        if item is None:
            paths.extend([None, None])
        else:
            paths.extend([str(item.waveform_plot), str(item.spectrogram_plot)])
    return tuple(paths)


def _stem_panel(name: str):
    with gr.Column(elem_classes=["stem-panel"]):
        gr.Markdown(f"### {LABELS[name]}")
        player = _player(name)
        waveform = _figure(f"{LABELS[name]} waveform")
        spectrogram = _figure(f"{LABELS[name]} spectrogram")
    return player, waveform, spectrogram


def _figure(label: str) -> gr.Image:
    return gr.Image(
        label=label,
        type="filepath",
        interactive=False,
        sources=[],
        buttons=["download", "fullscreen"],
    )


def _player(stem: str) -> gr.Audio:
    # The download button serves the 16-bit WAV written by the pipeline.
    return gr.Audio(
        label=LABELS[stem],
        type="filepath",
        interactive=False,
        editable=False,
        buttons=["download"],
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
    demo = build_demo()
    port = int(os.environ.get("STEM_PORT", "7860"))
    output_dir = Path(os.environ.get("STEM_OUTPUT_DIR", "outputs")).resolve()
    css_path = Path(__file__).resolve().parent / "stem_ai" / "dashboard.css"
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=False,
        share=False,
        allowed_paths=[str(output_dir)],
        theme=dashboard_theme(),
        css_paths=[css_path],
        footer_links=[],
    )


if __name__ == "__main__":
    main()
