from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from time import perf_counter
import logging

import joblib
import numpy as np
import yaml
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLO

from doctype import features

BASE = Path(__file__).resolve().parent
LOG = logging.getLogger(__name__)


class DocumentPipeline:
    def __init__(self, config_path=BASE / "config.yaml"):
        path = Path(config_path).resolve()
        self.versions = {"quality": None, "doctype": None}
        self.error = None
        self.lock = Lock()
        try:
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.versions = {name: config[f"{name}_version"] for name in self.versions}
            self.threshold = float(config["bad_threshold"])
            if not 0 <= self.threshold <= 1:
                raise ValueError("bad_threshold must be between 0 and 1")
            quality_path = path.parent / config["quality_model"]
            doctype_path = path.parent / config["doctype_model"]
            if not quality_path.is_file() or not doctype_path.is_file():
                self.error = "missing_model"
                return
            self.quality_model = YOLO(str(quality_path))
            self.bad_index = next(i for i, name in self.quality_model.names.items()
                                  if name.lower() == "bad")
            self.doctype_model = joblib.load(doctype_path)
        except Exception:
            LOG.exception("Pipeline initialization failed")
            self.error = "initialization_failed"

    def run(self, source, *, filename=None):
        started = perf_counter()
        result = {
            "file": filename or Path(source).name,
            "status": "error",
            "quality": None,
            "quality_confidence": None,
            "document_type": None,
            "document_type_confidence": None,
            "reason": None,
            "model_versions": self.versions.copy(),
            "latency_ms": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            if self.error:
                result["reason"] = self.error
                return result
            try:
                with Image.open(source) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                result["reason"] = "unsupported_file"
                return result
            with self.lock:
                # Ultralytics interprets ndarray input as BGR.
                pixels = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
                prediction = self.quality_model.predict(pixels, imgsz=224, device="cpu",
                                                       verbose=False)[0]
                prob_bad = float(prediction.probs.data[self.bad_index])
                if prob_bad >= self.threshold:
                    result.update(status="rejected", quality="bad",
                                  quality_confidence=prob_bad, reason="poor_quality")
                    return result
                result.update(quality="good", quality_confidence=1 - prob_bad)
                probabilities = self.doctype_model.predict_proba([features(image)])[0]
            index = int(np.argmax(probabilities))
            result.update(status="accepted", document_type=str(self.doctype_model.classes_[index]),
                          document_type_confidence=float(probabilities[index]))
        except Exception:
            LOG.exception("Inference failed")
            result["reason"] = "inference_failed"
        finally:
            result["latency_ms"] = round((perf_counter() - started) * 1000, 3)
        return result
