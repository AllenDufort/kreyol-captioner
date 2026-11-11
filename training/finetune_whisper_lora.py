"""LoRA fine-tuning of multilingual Whisper on the cleaned CMU Haitian Creole splits.

Run on a CUDA GPU (Colab/Kaggle T4 or better). It is NOT part of the app package and
needs its own dependencies:

    pip install -r training/requirements-train.txt

Inputs: a folder with audio/ and train.clean.jsonl / dev.clean.jsonl, as written by
`kreyol-captioner prepare-cmu` then `clean-cmu`. Zip data/cmu and upload it.

    python training/finetune_whisper_lora.py --data data/cmu --output finetuned/small-ht

Outputs:
    <output>/best-adapter/      LoRA adapter selected by dev WER
    <output>/merged/            full Transformers model (base + adapter)
    <output>/ctranslate2/       faster-whisper model: use with --model <that folder>
    <output>/history.json       dev WER/CER per evaluation

The test split is never read by this script. Report test once, at the very end, with
`kreyol-captioner benchmark --split test.clean --model <output>/ctranslate2`.
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

SAMPLE_RATE = 16000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-model", default="openai/whisper-small")
    p.add_argument("--train-manifest", default="train.clean.jsonl")
    p.add_argument("--dev-manifest", default="dev.clean.jsonl")
    p.add_argument("--epochs", type=float, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3, help="LoRA learning rate (adapters tolerate high LR)")
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--eval-every", type=int, default=200, help="Steps between dev evaluations")
    p.add_argument("--patience", type=int, default=4, help="Stop after N evals without dev WER gain")
    p.add_argument("--max-train", type=int, help="Use only N train rows (smoke test)")
    p.add_argument("--max-dev", type=int, help="Use only N dev rows (smoke test)")
    p.add_argument("--speed-perturb", action="store_true", help="Random 0.9-1.1x speed on train audio")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2, help="DataLoader worker processes")
    p.add_argument("--no-convert", action="store_true", help="Skip CTranslate2 conversion")
    return p.parse_args()


def read_manifest(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def load_audio(path: Path):
    import av
    import numpy as np

    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        chunks = []
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype="float32")


def speed_perturb(samples, rng: random.Random):
    import numpy as np

    factor = rng.uniform(0.9, 1.1)
    positions = np.arange(0, len(samples) - 1, factor)
    return np.interp(positions, np.arange(len(samples)), samples).astype("float32")


class Collator:
    """Top-level (picklable) so DataLoader workers also work with spawn on macOS/Windows."""

    def __init__(self, *, data: Path, processor, max_label_tokens: int, decoder_start: int,
                 perturb: bool, seed: int):
        self.data, self.processor, self.max_label_tokens = data, processor, max_label_tokens
        self.decoder_start, self.perturb, self.rng = decoder_start, perturb, random.Random(seed)

    def __call__(self, batch: list[dict]):
        import numpy as np
        import torch

        features, label_ids = [], []
        for row in batch:
            audio = load_audio(self.data / row["audio"])
            if self.perturb:
                audio = speed_perturb(audio, self.rng)
            features.append(self.processor.feature_extractor(
                audio, sampling_rate=SAMPLE_RATE, return_tensors="np").input_features[0])
            ids = self.processor.tokenizer(row["text"]).input_ids
            if ids and ids[0] == self.decoder_start:  # the model prepends this itself
                ids = ids[1:]
            label_ids.append(ids[:self.max_label_tokens])
        labels = torch.full((len(batch), max(map(len, label_ids))), -100, dtype=torch.long)
        for i, ids in enumerate(label_ids):
            labels[i, :len(ids)] = torch.tensor(ids)
        return torch.tensor(np.stack(features), dtype=torch.float32), labels, batch


def main():
    args = parse_args()
    import numpy as np
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        get_linear_schedule_with_warmup,
    )

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from kreyol_captioner.evaluation import edit_counts, normalize

    if not torch.cuda.is_available():
        print("Warning: no CUDA GPU found. Training on CPU will be extremely slow.", file=sys.stderr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    train_rows = read_manifest(args.data / args.train_manifest, args.max_train)
    dev_rows = read_manifest(args.data / args.dev_manifest, args.max_dev)
    if any("test" in r.get("split", "") for r in train_rows + dev_rows):
        raise SystemExit("Refusing to run: a test row was found in train/dev manifests.")
    print(f"train {len(train_rows)} rows, dev {len(dev_rows)} rows", flush=True)

    processor = WhisperProcessor.from_pretrained(args.base_model, language="ht", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    model.generation_config.language = "ht"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.use_cache = False

    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"], bias="none")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.to(device)

    collator_args = dict(data=args.data, processor=processor, max_label_tokens=model.config.max_target_positions,
                         decoder_start=model.config.decoder_start_token_id)
    workers = args.workers
    train_loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True, num_workers=workers,
                              collate_fn=Collator(**collator_args, perturb=args.speed_perturb, seed=args.seed),
                              drop_last=True, persistent_workers=workers > 0)
    dev_loader = DataLoader(dev_rows, batch_size=args.batch_size, shuffle=False, num_workers=workers,
                            collate_fn=Collator(**collator_args, perturb=False, seed=args.seed),
                            persistent_workers=workers > 0)
    if len(train_loader) == 0:
        raise SystemExit("Fewer train rows than --batch-size.")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def evaluate_dev():
        model.eval()
        words = chars = word_errors = char_errors = 0
        samples = []
        with torch.no_grad():
            for features, _, rows in dev_loader:
                with torch.autocast(device_type=device, enabled=use_amp, dtype=torch.float16):
                    generated = model.generate(input_features=features.to(device), language="ht",
                                               task="transcribe", num_beams=1, max_new_tokens=225)
                texts = processor.batch_decode(generated, skip_special_tokens=True)
                for row, hyp in zip(rows, texts, strict=True):
                    ref_n, hyp_n = normalize(row["text"]), normalize(hyp)
                    w = edit_counts(ref_n.split(), hyp_n.split())
                    c = edit_counts(list(ref_n.replace(" ", "")), list(hyp_n.replace(" ", "")))
                    words += w["reference_units"]
                    word_errors += w["errors"]
                    chars += c["reference_units"]
                    char_errors += c["errors"]
                    if len(samples) < 5:
                        samples.append({"ref": row["text"], "hyp": hyp})
        model.train()
        return {"wer": word_errors / max(words, 1), "cer": char_errors / max(chars, 1), "samples": samples}

    history = []
    baseline = evaluate_dev()
    history.append({"step": 0, **{k: baseline[k] for k in ("wer", "cer")}})
    print(f"step 0 (pretrained, greedy) dev WER {baseline['wer']:.3f} CER {baseline['cer']:.3f}", flush=True)
    best_wer, bad_evals, step = baseline["wer"], 0, 0
    best_dir = args.output / "best-adapter"
    model.save_pretrained(best_dir)

    model.train()
    done = False
    epoch = 0
    while not done:
        epoch += 1
        for batch_index, (features, labels, _) in enumerate(train_loader):
            with torch.autocast(device_type=device, enabled=use_amp, dtype=torch.float16):
                loss = model(input_features=features.to(device), labels=labels.to(device)).loss / args.grad_accum
            scaler.scale(loss).backward()
            if (batch_index + 1) % args.grad_accum:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            if step % 25 == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {loss.item() * args.grad_accum:.3f}", flush=True)
            if step % args.eval_every == 0 or step >= total_steps:
                result = evaluate_dev()
                history.append({"step": step, "epoch": epoch, "wer": result["wer"], "cer": result["cer"]})
                print(f"step {step} dev WER {result['wer']:.3f} CER {result['cer']:.3f}", flush=True)
                for s in result["samples"][:2]:
                    print(f"  REF {s['ref']}\n  HYP {s['hyp']}", flush=True)
                if result["wer"] < best_wer:
                    best_wer, bad_evals = result["wer"], 0
                    model.save_pretrained(best_dir)
                else:
                    bad_evals += 1
                (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
                if bad_evals >= args.patience:
                    print("Early stopping: dev WER stopped improving.", flush=True)
                    done = True
            if step >= total_steps or done:
                done = True
                break

    summary = {"base_model": args.base_model, "best_dev_wer_greedy": best_wer,
               "pretrained_dev_wer_greedy": baseline["wer"], "steps": step, "args": vars(args) | {
                   "data": str(args.data), "output": str(args.output)}}
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if best_wer >= baseline["wer"]:
        print("Warning: no checkpoint beat the pretrained model on dev. The saved adapter equals the base model.",
              file=sys.stderr)

    # Merge the best adapter into the base model and export for faster-whisper.
    base = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    merged = PeftModel.from_pretrained(base, best_dir).merge_and_unload()
    merged.generation_config.language = "ht"
    merged.generation_config.task = "transcribe"
    merged_dir = args.output / "merged"
    merged.save_pretrained(merged_dir)
    processor.save_pretrained(merged_dir)
    if not args.no_convert:
        subprocess.run([sys.executable, "-m", "ctranslate2.converters.transformers", "--model", str(merged_dir), "--output_dir",
                        str(args.output / "ctranslate2"), "--copy_files", "tokenizer.json",
                        "preprocessor_config.json", "--quantization", "float16", "--force"], check=True)
        print(f"faster-whisper model ready: {args.output / 'ctranslate2'}", flush=True)


if __name__ == "__main__":
    main()
