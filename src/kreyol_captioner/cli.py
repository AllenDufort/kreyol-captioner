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
    prep = sub.add_parser("prepare-cmu", help="Download CMU Haitian Creole data and make speaker-separated splits")
    prep.add_argument("--output", type=Path, default=Path("data/cmu"))
    prep.add_argument("--shards", type=int, nargs="+", help="Parquet shards 0-4 (default: all, about 2 GB)")
    prep.add_argument("--dev", type=float, default=0.1, help="Approximate dev fraction")
    prep.add_argument("--test", type=float, default=0.1, help="Approximate test fraction")
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--keep-text-overlap", action="store_true",
                      help="Keep dev/test rows whose text also appears in train")
    bench = sub.add_parser("benchmark", help="Measure corpus WER/CER on a prepared split")
    bench.add_argument("--data", type=Path, default=Path("data/cmu"))
    bench.add_argument("--split", default="dev", help="Manifest name in --data, e.g. dev, test, dev.clean")
    bench.add_argument("--model", default="base")
    bench.add_argument("--limit", type=int, help="Only the first N utterances (smoke test)")
    bench.add_argument("--output", type=Path, default=Path("evaluations"))
    bench.add_argument("--cache-dir", type=Path, default=Path(".cache/models"))
    bench.add_argument("--offline", action="store_true")
    clean = sub.add_parser("clean-cmu", help="Filter broken audio/text from prepared splits")
    clean.add_argument("--data", type=Path, default=Path("data/cmu"))
    clean.add_argument("--asr-results", type=Path,
                       help="train utterances.jsonl from `benchmark --split train`; drops train items with high CER")
    clean.add_argument("--max-cer", type=float, default=0.8)
    app = sub.add_parser("app", help="Launch the local web interface")
    app.add_argument("--port", type=int, default=7860)
    app.add_argument("--output", type=Path, default=Path("runs"))
    app.add_argument("--cache-dir", type=Path, default=Path(".cache/models"))
    app.add_argument("--offline", action="store_true")
    app.add_argument("--review-model", type=Path)
    app.add_argument("--model", type=Path, help="Fine-tuned faster-whisper model folder to offer (and preselect)")
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
        elif args.command == "prepare-cmu":
            from .data import prepare
            print(json.dumps(prepare(args.output, shards=args.shards, dev=args.dev, test=args.test,
                                     seed=args.seed, keep_text_overlap=args.keep_text_overlap),
                             indent=2))
        elif args.command == "clean-cmu":
            from .clean import clean_split
            summary = {}
            for split in ("train", "dev", "test"):
                kept, rejected = clean_split(args.data, split,
                                             asr_results=args.asr_results if split == "train" else None,
                                             max_cer=args.max_cer)
                for name, rows in ((f"{split}.clean.jsonl", kept), (f"{split}.rejected.jsonl", rejected)):
                    (args.data / name).write_text(
                        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
                reasons = {}
                for r in rejected:
                    for p in r["problems"]:
                        reasons[p] = reasons.get(p, 0) + 1
                summary[split] = {"kept": len(kept), "rejected": len(rejected), "reasons": reasons}
            (args.data / "clean_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(summary, indent=2))
        elif args.command == "benchmark":
            from .benchmark import run_benchmark
            summary = run_benchmark(args.data, args.split, args.model, args.output, limit=args.limit,
                                    cache_dir=args.cache_dir, offline=args.offline)
            print(json.dumps({k: summary[k] for k in ("model", "split", "overall", "empty_hypotheses",
                                                      "real_time_factor")}, indent=2))
        elif args.command == "app":
            from .app import launch
            launch(port=args.port, output=args.output, cache_dir=args.cache_dir,
                   offline=args.offline, review_model=args.review_model, custom_model=args.model)
        return 0
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
