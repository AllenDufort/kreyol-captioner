import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kreyol_captioner.app import save_reviewed
from kreyol_captioner.evaluation import evaluate, normalize
from kreyol_captioner.models import Caption, Segment, Word
from kreyol_captioner.review import FEATURES, annotate, load_review_model, score, train
from kreyol_captioner.subtitles import make_captions, subtitle_text, timestamp, validate_captions
from kreyol_captioner.transcribe import run


class SubtitleTests(unittest.TestCase):
    def test_timestamp_rollover(self):
        self.assertEqual(timestamp(59.9996), "00:01:00,000")
        self.assertEqual(timestamp(3661.125, vtt=True), "01:01:01.125")

    def test_keep_original_time_after_silence(self):
        captions = make_captions([Segment(0, 10, 12, "Mwen pale kreyòl.", words=[
            Word(10, 10.5, "Mwen"), Word(10.6, 11, " pale"), Word(11.1, 12, " kreyòl.")])])
        self.assertEqual(captions[0].start, 10)
        self.assertEqual(captions[0].text, "Mwen pale kreyòl.")

    def test_split_long_caption_preserving_every_word(self):
        words = [Word(i * 0.4, (i + 1) * 0.4, f" mo{i}") for i in range(40)]
        captions = make_captions([Segment(0, 0, 16, "".join(w.text for w in words).strip(), words)])
        self.assertGreater(len(captions), 1)
        self.assertEqual(" ".join(" ".join(c.text.split()) for c in captions),
                         "".join(w.text for w in words).strip())
        self.assertTrue(all(len(c.text.splitlines()) <= 2 for c in captions))
        validate_captions(captions, 16)

    def test_missing_words_does_not_invent_timings(self):
        captions = make_captions([Segment(0, 8, 14, "Bonjou tout moun.")])
        self.assertEqual((captions[0].start, captions[0].end), (8, 14))

    def test_zero_duration_word_kept(self):
        captions = make_captions([Segment(0, 0, 1, "Bonjou zanmi", words=[
            Word(0, 0.5, "Bonjou"), Word(0.5, 0.5, " zanmi")])])
        self.assertIn("zanmi", captions[0].text)

    def test_zero_duration_word_at_segment_end_kept(self):
        captions = make_captions([Segment(0, 0, 1, "Bonjou", words=[Word(1, 1, "Bonjou")])])
        self.assertEqual(captions[0].text, "Bonjou")
        validate_captions(captions, 1)

    def test_attached_punctuation_is_not_separated(self):
        words = [Word(0, 0.3, "M"), Word(0.3, 0.5, "'ap"), Word(0.5, 1, " pale.")]
        captions = make_captions([Segment(0, 0, 1, "M'ap pale.", words=words)])
        self.assertEqual(captions[0].text, "M'ap pale.")

    def test_unicode_and_vtt_markup(self):
        caption = [Caption(1, 2, "Kreyòl <è> & ò")]
        self.assertIn("Kreyòl <è> & ò", subtitle_text(caption))
        self.assertIn("Kreyòl &lt;è&gt; &amp; ò", subtitle_text(caption, vtt=True))

    def test_invalid_timestamps_and_overlaps(self):
        for start, end in [(-1, 2), (1, 1), (2, 1), (float("nan"), 2)]:
            with self.assertRaises(ValueError):
                Caption(start, end, "Bonjou")
        with self.assertRaises(ValueError):
            validate_captions([Caption(0, 2, "A"), Caption(1, 3, "B")])
        with self.assertRaises(ValueError):
            validate_captions([Caption(0, 2, "A")], duration=1)
        with self.assertRaises(ValueError):
            validate_captions([Caption(0, 0.0001, "A")])

    def test_silence_exports(self):
        self.assertEqual(subtitle_text([]), "")
        self.assertEqual(subtitle_text([], vtt=True), "WEBVTT\n\n")


class EvaluationTests(unittest.TestCase):
    def test_normalization_preserves_meaningful_accents(self):
        self.assertEqual(normalize("  KREYO\u0300L,   ayisyen! "), "kreyòl ayisyen")
        self.assertEqual(evaluate("Mwen pale kreyòl.", "MWEN pale kreyòl!")["wer"], 0)
        self.assertGreater(evaluate("pè", "pe")["cer"], 0)

    def test_edit_counts(self):
        report = evaluate("a b c", "a x c d")
        self.assertEqual(report["words"]["substitutions"], 1)
        self.assertEqual(report["words"]["insertions"], 1)
        self.assertAlmostEqual(report["wer"], 2 / 3)
        self.assertEqual(evaluate("a b", "a")["words"]["deletions"], 1)

    def test_silence_and_literal(self):
        self.assertIsNone(evaluate("", "invented words")["wer"])
        self.assertEqual(evaluate("", "")["wer"], 0)
        self.assertGreater(evaluate("Bonjou", "bonjou", normalized=False)["wer"], 0)


class ReviewTests(unittest.TestCase):
    def test_heuristic_is_not_a_trained_probability(self):
        segments = [Segment(0, 0, 1, "Bonjou", avg_logprob=-2)]
        annotate(segments)
        self.assertTrue(segments[0].needs_review)
        self.assertIsNone(segments[0].review_score)

    def test_invalid_model_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            path.write_text('{"schema_version": 99}')
            with self.assertRaises(ValueError):
                load_review_model(path)

    def test_training_and_portable_inference(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("Optional scikit-learn is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            path, model_path = Path(tmp) / "labels.csv", Path(tmp) / "model.json"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "group_id", "recording_id", "segment_id", "label", *FEATURES])
                writer.writeheader()
                # Synthetic fixtures exercise the trainer; they are not project training data.
                for group in range(8):
                    for index in range(12):
                        label = index % 2
                        values = [-2.0 if label else -0.1, 0.8 if label else 0.01,
                                  3 if label else 1, 3, 30 if label else 10, 8]
                        writer.writerow({"group_id": f"speaker-{group}", "recording_id": f"r{group}",
                                         "segment_id": index, "label": label, **dict(zip(FEATURES, values, strict=True))})
            report = train([path], model_path)
            model = load_review_model(model_path)
            self.assertEqual(report["training_groups"] + report["validation_groups"], 8)
            self.assertLess(score([-0.1, 0.01, 1, 3, 10, 8], model), 0.5)
            self.assertGreater(score([-2, 0.8, 3, 3, 30, 8], model), 0.5)
            report = train([path, path], model_path)
            self.assertEqual(report["duplicate_rows_skipped"], 96)


class WorkflowTests(unittest.TestCase):
    def test_transcription_exports_with_language_and_blank_labels(self):
        raw = SimpleNamespace(start=3.0, end=4.0, text="Bonjou Ayiti.",
                              words=[SimpleNamespace(start=3.0, end=3.5, word="Bonjou"),
                                     SimpleNamespace(start=3.5, end=4.0, word=" Ayiti.")],
                              avg_logprob=-0.2, no_speech_prob=0.01, compression_ratio=1)
        model = SimpleNamespace(transcribe=lambda *a, **kw: (iter([raw]), SimpleNamespace(duration=5.0)))
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "sample.wav"
            source.write_bytes(b"mock audio; decoder is replaced in this test")
            with patch("kreyol_captioner.transcribe.get_model", return_value=model), \
                 patch("kreyol_captioner.transcribe.version", return_value="test"):
                transcript, directory, captions = run(source, Path(tmp) / "runs")
            self.assertEqual(transcript.language, "ht")
            self.assertEqual(captions[0].start, 3)
            data = json.loads((directory / "transcript.json").read_text())
            self.assertEqual(data["metadata"]["task"], "transcribe")
            self.assertEqual(data["metadata"]["review_method"], "heuristic")
            with (directory / "review.csv").open() as handle:
                row = next(csv.DictReader(handle))
                self.assertEqual(row["label"], "")
                self.assertEqual(row["group_id"], data["metadata"]["source_sha256"])
            original = (directory / "transcript.json").read_bytes()
            paths, _ = save_reviewed([[3, 4, "Bonjou, Ayiti!", False]],
                                     {"directory": str(directory), "duration": 5})
            self.assertEqual(len(paths), 3)
            self.assertEqual((directory / "transcript.json").read_bytes(), original)
            self.assertIn("Bonjou, Ayiti!", Path(paths[0]).read_text())

    def test_reviewed_captions_reject_overlap(self):
        with self.assertRaises(ValueError):
            save_reviewed([[0, 2, "A"], [1, 3, "B"]], {"directory": ".", "duration": 3})

    def test_missing_input_fails_before_model_load(self):
        with patch("kreyol_captioner.transcribe.get_model") as model:
            with self.assertRaises(ValueError):
                run(Path("/no-such-recording.wav"), Path("runs"))
            model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
