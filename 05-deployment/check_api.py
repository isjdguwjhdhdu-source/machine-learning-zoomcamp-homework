"""Check health, model checksums and Q4/Q6 prediction of a running API."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:9696")
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    metadata = json.loads((ROOT / "model_metadata.json").read_text(encoding="utf-8"))
    with urlopen(f"{base_url}/health", timeout=10) as response:
        health = json.load(response)

    if health.get("status") != "ok":
        raise RuntimeError(f"API is not healthy: {health}")
    for key in ("model_sha256", "feature_defaults_sha256"):
        if health.get(key) != metadata[key]:
            raise RuntimeError(f"Checksum mismatch: {key}")

    request = Request(
        f"{base_url}/predict",
        data=(ROOT / "examples/q4.json").read_bytes(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        result = json.load(response)

    probability = result.get("conversion_probability")
    if type(probability) not in (int, float) or not 0 <= probability <= 1:
        raise RuntimeError(f"Invalid probability: {result}")
    if not isinstance(result.get("conversion"), bool):
        raise RuntimeError(f"conversion must be boolean: {result}")
    if abs(probability - 0.770) > 0.005:
        raise RuntimeError(f"Prediction is outside the homework tolerance: {result}")
    if result["conversion"] is not True:
        raise RuntimeError(f"Expected conversion=true for the Q4/Q6 lead: {result}")

    print("Health and checksums: OK")
    print(json.dumps(result, indent=2))
    print(f"Q4/Q6: {probability:.3f}")


if __name__ == "__main__":
    main()
