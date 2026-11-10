"""Command-line entry points. Heavy dependencies are imported only when needed."""

import argparse
import json
import sys
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Generate Haitian Creole captions locally.")
    sub = root.add_subparsers(dest="command", required=True)
    transcribe = sub.add_parser("transcribe", help="Transcribe audio or video to SRT/VTT/JSON")
    transcribe.add_argument("input", type=Path)
    transcribe.add_argument("--output", type=Path, default=Path("runs"))
    transcribe.add_argument("--model", default="base", help="Multilingual Whisper model or local model directory")
    transcribe.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    transcribe.add_argument("--compute-type", default="int8")
    transcribe.add_argument("--cache-dir", type=Path, default=Path(".cache/models"))
    transcribe.add_argument("--offline", action="store_true", help="Require cached model files; no download")
    transcribe.add_argument("--review-model", type=Path)
    evaluate = sub.add_parser("evaluate", help="Compare a reference and prediction (.txt or transcript .json)")
    evaluate.add_argument("--reference", type=Path, required=True)
    evaluate.add_argument("--hypothesis", type=Path, required=True)
    evaluate.add_argument("--literal", action="store_true", help="Keep case and punctuation")
    evaluate.add_argument("--output", type=Path)
    training = sub.add_parser("train-review", help="Train traditional ML from human-labeled review CSVs")
    training.add_argument("csv", nargs="+", type=Path)
    training.add_argument("--output", type=Path, default=Path("models/review-model.json"))
    training.add_argument("--threshold", type=float, default=0.5)
    training.add_argument("--seed", type=int, default=42)
    app = sub.add_parser("app", help="Launch the local web interface")
    app.add_argument("--port", type=int, default=7860)
    app.add_argument("--output", type=Path, default=Path("runs"))
    app.add_argument("--cache-dir", type=Path, default=Path(".cache/models"))
    app.add_argument("--offline", action="store_true")
    app.add_argument("--review-model", type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "transcribe":
            if args.model.endswith(".en"):
                raise ValueError("Choose a multilingual model, such as base or small, without .en.")
            from .transcribe import run
            transcript, directory, captions = run(
                args.input, args.output, model_name=args.model, device=args.device,
                compute_type=args.compute_type, cache_dir=args.cache_dir, offline=args.offline,
                review_model=args.review_model,
            )
            print(json.dumps({"output_directory": str(directory), "language": transcript.language,
                              "segments": len(transcript.segments), "captions": len(captions),
                              "needs_review": sum(s.needs_review for s in transcript.segments),
                              "message": "Draft captions generated. Review them before use." if captions
                              else "No speech detected; subtitle files contain no captions."}, indent=2))
        elif args.command == "evaluate":
            from .evaluation import evaluate, read_text
            report = evaluate(read_text(args.reference), read_text(args.hypothesis), normalized=not args.literal)
            rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered + "\n", encoding="utf-8")
            print(rendered)
        elif args.command == "train-review":
            from .review import train
            print(json.dumps(train(args.csv, args.output, threshold=args.threshold, seed=args.seed), indent=2))
            print(f"Review model saved to {args.output.resolve()}")
        elif args.command == "app":
            from .app import launch
            launch(port=args.port, output=args.output, cache_dir=args.cache_dir,
                   offline=args.offline, review_model=args.review_model)
        return 0
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
