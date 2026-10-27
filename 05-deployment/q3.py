"""Homework 5, Q3: load the frozen pipeline and predict conversion."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

from model import normalize_record


ROOT = Path(__file__).resolve().parent
EXPECTED_SHA256 = "1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1"


def main() -> None:
    model_path = ROOT / "pipeline.bin"
    actual_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual_sha256 != EXPECTED_SHA256:
        raise RuntimeError("pipeline.bin does not match the frozen 2026 release")

    with model_path.open("rb") as file:
        pipeline = pickle.load(file)

    client = json.loads((ROOT / "examples/q3.json").read_text(encoding="utf-8"))
    probability = float(pipeline.predict_proba([normalize_record(client)])[0, 1])

    print(f"conversion_probability: {probability:.12f}")
    print(f"Q3: {probability:.3f}")


if __name__ == "__main__":
    main()
