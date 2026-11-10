"""A localhost-only interface for generating and correcting captions."""

import json
from dataclasses import asdict
from pathlib import Path

from .models import Caption
from .subtitles import validate_captions, wrap_text, write_subtitles


def save_reviewed(rows: list, state: dict) -> tuple[list[str], str]:
    if not state:
        raise ValueError("Generate captions before saving edits.")
    if not rows:
        raise ValueError("Add at least one caption before saving.")
    captions = []
    for row in rows:
        if len(row) < 3 or not isinstance(row[2], str):
            raise ValueError("Each row needs a start, end, and caption text.")
        captions.append(Caption(float(row[0]), float(row[1]), wrap_text(row[2])))
    validate_captions(captions, float(state["duration"]))
    directory = Path(state["directory"])
    srt, vtt = write_subtitles(captions, directory, "captions.reviewed")
    reviewed = directory / "transcript.reviewed.json"
    reviewed.write_text(json.dumps({
        "schema_version": 1, "language": "ht", "reviewed": True,
        "source_transcript": "transcript.json",
        "segments": [{"id": i, **asdict(caption)} for i, caption in enumerate(captions)],
    }, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return [str(srt), str(vtt), str(reviewed)], "Edits saved. Download the reviewed subtitle files below."


def build_app(*, output: Path, cache_dir: Path, offline: bool = False, review_model: Path | None = None):
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError('Install the interface with: pip install -e ".[ui]"') from error
    from .transcribe import EXTENSIONS, run

    def generate(source, model_name, progress=gr.Progress()):  # noqa: B008 - Gradio injects progress.
        if not source:
            raise gr.Error("Choose an audio or video file first.")
        try:
            transcript, directory, captions = run(
                Path(source), output, model_name=model_name, cache_dir=cache_dir,
                offline=offline, review_model=review_model, progress=progress,
            )
        except Exception as error:
            raise gr.Error(str(error)) from error
        rows = [[c.start, c.end, c.text, any(
            s.needs_review and s.start < c.end and s.end > c.start for s in transcript.segments)]
                for c in captions]
        files = [str(directory / name) for name in
                 ("captions.srt", "captions.vtt", "transcript.json", "transcript.txt", "review.csv")]
        count = sum(s.needs_review for s in transcript.segments)
        status = (f"**{len(captions)} draft captions** · {transcript.duration:.1f} seconds · "
                  f"{count} speech segments suggested for review. Check all captions before publishing.")
        if not captions:
            status = "No speech detected. Try a recording with clearer speech."
        return rows, {"directory": str(directory), "duration": transcript.duration}, files, status, None

    def save(rows, state):
        try:
            return save_reviewed(rows, state)
        except (ValueError, TypeError, OSError) as error:
            raise gr.Error(str(error)) from error

    def preview(source):
        suffix = Path(source).suffix.lower() if source else ""
        video = suffix in {".mp4", ".webm"}
        audio = suffix in EXTENSIONS - {".mp4", ".webm", ".mov", ".mkv", ".avi"}
        return (gr.update(value=source if video else None, visible=video),
                gr.update(value=source if audio else None, visible=audio),
                [], {}, None, None, "Ready to generate captions for this recording.")

    css = ".gradio-container {max-width: 1180px !important;} footer {display:none !important;}"
    with gr.Blocks(title="Kreyòl captions", analytics_enabled=False,
                   theme=gr.themes.Soft(primary_hue="teal"), css=css) as demo:
        gr.Markdown("# Kreyòl captions\nHaitian Creole audio and video, turned into editable captions.")
        gr.Markdown("Runs on this computer. " + (
            "Offline mode: choose a speech model already downloaded to this computer."
            if offline else "The first transcription downloads a free speech model; "
            "later runs can use the cached model offline."))
        state = gr.State({})
        with gr.Row():
            with gr.Column(scale=1):
                source = gr.File(label="1. Choose audio or video", type="filepath",
                                 file_types=sorted(EXTENSIONS))
                model = gr.Radio(["tiny", "base", "small"], value="base", label="Speech model",
                                 info="Start with base. Compare small for accuracy or tiny for speed.")
                generate_button = gr.Button("Generate Haitian Creole captions", variant="primary")
                method = "A trained classifier suggests" if review_model else "Preliminary review rules suggest"
                gr.Markdown(f"{method} passages to check. Suggestions can miss errors.")
                video = gr.Video(label="Video preview", visible=False, interactive=False)
                audio = gr.Audio(label="Audio preview", visible=False, interactive=False, format=None)
                gr.Markdown("Browser video preview supports MP4 and WebM. Other listed video formats "
                            "can still be transcribed.")
            with gr.Column(scale=2):
                status = gr.Markdown("Choose a recording to begin.")
                table = gr.Dataframe(
                    headers=["Start (seconds)", "End (seconds)", "Caption", "Suggested review"],
                    datatype=["number", "number", "str", "bool"],
                    type="array", interactive=True, col_count=(4, "fixed"),
                    label="2. Review text and timing", wrap=True,
                )
                gr.Markdown("Edit the text and timestamps. The review column is advisory. "
                            "Saving edits preserves the original transcription.")
                save_button = gr.Button("Save reviewed captions", variant="primary")
                reviewed_files = gr.File(label="3. Download reviewed captions", file_count="multiple", interactive=False)
                with gr.Accordion("Draft files and labeling CSV", open=False):
                    draft_files = gr.File(label="Original outputs", file_count="multiple", interactive=False)
        source.change(preview, inputs=source,
                      outputs=[video, audio, table, state, draft_files, reviewed_files, status], queue=False)
        generate_button.click(generate, inputs=[source, model],
                              outputs=[table, state, draft_files, status, reviewed_files], concurrency_limit=1)
        save_button.click(save, inputs=[table, state], outputs=[reviewed_files, status])
    return demo


def launch(*, port: int = 7860, output: Path = Path("runs"), cache_dir: Path = Path(".cache/models"),
           offline: bool = False, review_model: Path | None = None) -> None:
    # No public tunnel and no network listener beyond localhost.
    demo = build_app(output=output, cache_dir=cache_dir, offline=offline, review_model=review_model)
    demo.queue(max_size=8).launch(server_name="127.0.0.1", server_port=port,
                                 share=False, show_api=False, inbrowser=False)
