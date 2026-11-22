import json
from pathlib import Path
from statistics import mean
import sys
from unittest.mock import patch

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batch import run_batch
from pipeline import BASE, DocumentPipeline

SAMPLES = BASE / "samples"
GOOD = SAMPLES / "good_inn.jpg"
BAD = SAMPLES / "bad_blur.jpg"


@pytest.fixture(scope="module")
def pipeline():
    return DocumentPipeline()


def test_schema(pipeline):
    result = pipeline.run(GOOD)
    types = dict(file=str, status=str, quality=str, quality_confidence=float,
                 document_type=str, document_type_confidence=float, reason=type(None),
                 model_versions=dict, latency_ms=float, timestamp=str)
    assert set(result) == set(types)
    assert all(isinstance(result[key], kind) for key, kind in types.items())
    assert set(result["model_versions"]) == {"quality", "doctype"}


@pytest.mark.parametrize("kind", ["inn", "passport", "snils"])
def test_good_document(pipeline, kind):
    result = pipeline.run(SAMPLES / f"good_{kind}.jpg")
    assert result["status"] == "accepted"
    assert result["document_type"] == kind.upper()


@pytest.mark.parametrize("defect", ["blur", "glare"])
def test_bad_photo_stops_before_classifier(pipeline, defect):
    with patch.object(pipeline.doctype_model, "predict_proba", side_effect=AssertionError):
        result = pipeline.run(SAMPLES / f"bad_{defect}.jpg")
    assert result["status"] == "rejected"
    assert result["reason"] == "poor_quality"
    assert result["document_type"] is result["document_type_confidence"] is None


@pytest.mark.parametrize("name", ["corrupt.jpg", "renamed_text.jpg"])
def test_unsupported_file(pipeline, name):
    result = pipeline.run(SAMPLES / name)
    assert (result["status"], result["reason"]) == ("error", "unsupported_file")


def test_cpu_latency(pipeline):
    paths = sorted((BASE / "benchmark_samples").glob("*.jpg"))
    assert len(paths) == 20
    pipeline.run(paths[0])
    results = [pipeline.run(path) for path in paths]
    assert all(result["status"] != "error" for result in results)
    latency = mean(result["latency_ms"] for result in results)
    print(f"CPU mean over 20 samples: {latency:.3f} ms")
    assert latency < 500


def config_in(tmp_path, **overrides):
    values = yaml.safe_load((BASE / "config.yaml").read_text())
    for key in ("quality_model", "doctype_model"):
        values[key] = str(BASE / values[key])
    values.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values))
    return path


def test_threshold_changes_decision(pipeline, tmp_path):
    confidence = pipeline.run(GOOD)["quality_confidence"]
    boundary = 1.0 - confidence
    assert 0 < boundary < 1
    margin = min(boundary, 1 - boundary) / 2
    for threshold, expected in [(boundary - margin, "rejected"), (boundary + margin, "accepted")]:
        configured = DocumentPipeline(config_in(tmp_path, bad_threshold=threshold))
        assert configured.run(GOOD)["status"] == expected


def test_missing_model(tmp_path):
    configured = DocumentPipeline(config_in(tmp_path, doctype_model="missing.joblib"))
    result = configured.run(GOOD)
    assert (result["status"], result["reason"]) == ("error", "missing_model")


def test_batch(pipeline, tmp_path):
    output = tmp_path / "predictions.json"
    report = run_batch(pipeline, SAMPLES, output)
    assert json.loads(output.read_text()) == report
    assert report["counts"] == {"accepted": 3, "rejected": 2, "error": 2}
