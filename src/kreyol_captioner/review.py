"""Human labeling, traditional ML training, and safe JSON model inference."""

import csv
import hashlib
import json
import math
from pathlib import Path

from .models import Segment, Transcript

FEATURES = ["avg_logprob", "no_speech_prob", "compression_ratio", "duration",
            "characters_per_second", "word_count"]


def features(segment: Segment) -> list[float]:
    duration = segment.end - segment.start
    values = [segment.avg_logprob, segment.no_speech_prob, segment.compression_ratio,
              duration, len(segment.text.strip()) / duration, len(segment.text.split())]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Review features must be finite numbers.")
    return values


def load_review_model(path: Path) -> dict:
    model = json.loads(path.read_text(encoding="utf-8"))
    if model.get("schema_version") != 1 or model.get("features") != FEATURES:
        raise ValueError("Unsupported review model schema or feature order.")
    for key in ("mean", "scale", "coefficients"):
        values = model.get(key)
        if not isinstance(values, list) or len(values) != len(FEATURES):
            raise ValueError(f"Review model has invalid {key}.")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError(f"Review model has invalid {key}.")
    if any(value <= 0 for value in model["scale"]):
        raise ValueError("Review model scales must be positive.")
    for key in ("intercept", "threshold"):
        value = model.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Review model has invalid {key}.")
    if not 0 < model["threshold"] < 1:
        raise ValueError("Review threshold must be between 0 and 1.")
    return model


def score(values: list[float], model: dict) -> float:
    z = model["intercept"] + sum(
        coefficient * (value - mean) / scale
        for value, mean, scale, coefficient in zip(
            values, model["mean"], model["scale"], model["coefficients"], strict=True)
    )
    # Stable sigmoid; do not describe this as calibrated transcription confidence.
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1 + exp_z)


def annotate(segments: list[Segment], model: dict | None = None) -> None:
    for segment in segments:
        segment.review_reasons = []
        segment.review_score = None
        values = features(segment)
        if model is not None:
            segment.review_score = score(values, model)
            segment.needs_review = segment.review_score >= model["threshold"]
            if segment.needs_review:
                segment.review_reasons.append("Flagged by the trained review classifier")
            continue
        if segment.avg_logprob < -1.0:
            segment.review_reasons.append("Low recognition score")
        if segment.no_speech_prob > 0.5:
            segment.review_reasons.append("Possible silence or background sound")
        if segment.compression_ratio > 2.4:
            segment.review_reasons.append("Possible repeated text")
        if values[4] > 25:
            segment.review_reasons.append("High reading speed")
        segment.needs_review = bool(segment.review_reasons)


def write_label_template(transcript: Transcript, path: Path) -> None:
    """Labels stay empty until a human audits each segment.

    group_id is a content hash by default. Replace it with a shared speaker ID
    for multiple recordings of one speaker before training.
    """
    fieldnames = ["group_id", "recording_id", "segment_id", "start", "end", "prediction",
                  *FEATURES, "suggested_review", "label"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for segment in transcript.segments:
            writer.writerow({
                "group_id": transcript.metadata["source_sha256"],
                "recording_id": transcript.recording_id, "segment_id": segment.id,
                "start": segment.start, "end": segment.end, "prediction": segment.text,
                **dict(zip(FEATURES, features(segment), strict=True)),
                "suggested_review": int(segment.needs_review), "label": "",
            })


def train(paths: list[Path], output: Path, *, threshold: float = 0.5, seed: int = 42) -> dict:
    """Train logistic regression using labeled CSVs and a grouped validation split."""
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be strictly between 0 and 1.")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise RuntimeError('Install ML support with: pip install -e ".[ml]"') from error
    x, y, groups = [], [], []
    seen = set()
    duplicates = 0
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"group_id", "recording_id", "segment_id", "label", *FEATURES}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{path.name} is missing required review CSV columns.")
            for row in reader:
                label = row["label"].strip()
                if not label:
                    continue
                if label not in {"0", "1"}:
                    raise ValueError("Labels must be 0 (acceptable), 1 (needs correction), or empty.")
                group = row["group_id"].strip()
                if not group:
                    raise ValueError("Every labeled row needs a group_id.")
                values = [float(row[name]) for name in FEATURES]
                if not all(math.isfinite(v) for v in values):
                    raise ValueError("Training features must be finite numbers.")
                identity = (row["recording_id"], row["segment_id"])
                if identity in seen:
                    duplicates += 1
                    continue
                seen.add(identity)
                x.append(values)
                y.append(int(label))
                groups.append(group)
    if len(x) < 20 or len(set(y)) != 2 or len(set(groups)) < 3:
        raise ValueError("Provide at least 20 human-labeled segments, both label classes, "
                         "and at least 3 independent speaker/source groups.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    training, validation = next(splitter.split(x, y, groups))
    if len({y[i] for i in training}) < 2:
        raise ValueError("Grouped training split contains only one label class. "
                         "Collect more groups with both correct and incorrect captions.")
    scaler = StandardScaler()
    train_x = scaler.fit_transform([x[i] for i in training])
    classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    classifier.fit(train_x, [y[i] for i in training])
    validation_scores = classifier.predict_proba(scaler.transform([x[i] for i in validation]))[:, 1]
    predicted = (validation_scores >= threshold).astype(int)
    actual = [y[i] for i in validation]
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, average="binary", zero_division=0)
    report = {
        "training_segments": len(training), "validation_segments": len(validation),
        "training_groups": len({groups[i] for i in training}),
        "validation_groups": len({groups[i] for i in validation}),
        "validation_has_both_classes": len(set(actual)) == 2,
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "flagged_fraction": float(predicted.mean()),
        "confusion_matrix_labels_0_1": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
        "duplicate_rows_skipped": duplicates,
        "note": "Grouped development validation, not a final test or probability calibration. "
                "Minimum sample checks are software guards, not evidence of model quality.",
    }
    model = {"schema_version": 1, "model_type": "standardized_logistic_regression",
             "features": FEATURES, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
             "coefficients": classifier.coef_[0].tolist(), "intercept": float(classifier.intercept_[0]),
             "threshold": threshold, "seed": seed, "validation": report,
             "training_csv_sha256": [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report
