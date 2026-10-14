"""Append-only, point-in-time prediction ledger for research audits."""
import json
from datetime import datetime, timezone
from pathlib import Path

REQUIRED = {"event_id", "market", "selection", "model_version", "predicted_at_utc"}

def append_prediction(path, prediction):
    missing = REQUIRED - set(prediction)
    if missing: raise ValueError("Missing required prediction fields: "+", ".join(sorted(missing)))
    stamp = datetime.fromisoformat(str(prediction["predicted_at_utc"]).replace("Z","+00:00"))
    if stamp.tzinfo is None: raise ValueError("Prediction timestamp must be timezone aware")
    if "outcome" in prediction or "result" in prediction:
        raise ValueError("Predictions must be recorded separately from settled outcomes")
    destination=Path(path)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("a",encoding="utf-8") as f:
        f.write(json.dumps(prediction,sort_keys=True,separators=(",",":"))+"\n")
