"""Append-only, hash-chained prediction ledgers: SHADOW (logged, not wagered) and ACTUAL (placed).

Every line is one JSON record with prev_hash and record_hash = sha256(prev_hash + canonical body),
so any edit, deletion or reordering of an earlier line breaks verification. Records are never
updated: closing lines and settlements are separate records that reference a prediction_id.

Record kinds
  prediction  written BEFORE tip; carries the full decision (line, price, probabilities, EV, tier, stake%)
  close       closing quote for that prediction (from the live logger), written after tip
  settlement  outcome and P&L, written after the game is final
Shadow and actual ledgers are separate files; an ACTUAL prediction must carry bet confirmation
fields, and a shadow ledger refuses them, so shadow returns can never be reported as money won.
"""
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

from v13.pricing import no_vig_pair, parse_utc, profit_per_unit

GENESIS = "0" * 64
PREDICTION_FIELDS = ("prediction_id", "predicted_at_utc", "tip_utc", "game_id", "market", "side", "line", "price",
                     "opp_price", "sportsbook", "p_model", "p_market", "ev", "tier", "stake_pct_of_bankroll",
                     "model_version", "git_commit", "source")
ACTUAL_ONLY = ("confirmed_price", "stake_amount", "bet_reference")
TIERS = ("A", "A+", "A++")


def _canonical(body):
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def _hash(prev, body):
    return hashlib.sha256((prev + _canonical(body)).encode()).hexdigest()


def read(path):
    p = Path(path)
    return [json.loads(x) for x in p.open() if x.strip()] if p.exists() else []


def verify(path):
    """Raise ValueError at the first broken link; return the number of valid records."""
    prev = GENESIS
    for i, rec in enumerate(read(path)):
        body = {k: v for k, v in rec.items() if k not in ("prev_hash", "record_hash")}
        if rec.get("prev_hash") != prev or rec.get("record_hash") != _hash(prev, body):
            raise ValueError(f"ledger chain broken at record {i}")
        prev = rec["record_hash"]
    return i + 1 if prev != GENESIS else 0


def _append(path, body):
    records = read(path)
    verify(path)
    prev = records[-1]["record_hash"] if records else GENESIS
    rec = {**body, "prev_hash": prev, "record_hash": _hash(prev, body)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as f:
        f.write(_canonical(rec) + "\n")
    return rec


def append_prediction(path, pred, kind="shadow", now=None):
    missing = [k for k in PREDICTION_FIELDS if pred.get(k) is None]
    if missing:
        raise ValueError(f"missing prediction fields: {missing}")
    if pred["tier"] not in TIERS:
        raise ValueError("only A/A+/A++ selections are recorded as predictions (PASS is never a recommendation)")
    if not 0 < float(pred["stake_pct_of_bankroll"]) <= 0.03:
        raise ValueError("stake must be a fraction of current bankroll in (0, 0.03]")
    stamped, tip = parse_utc(pred["predicted_at_utc"]), parse_utc(pred["tip_utc"])
    now = now or datetime.now(timezone.utc)
    if stamped is None or tip is None or not stamped < tip:
        raise ValueError("prediction must be timestamped before tip")
    if stamped > now:
        raise ValueError("prediction timestamp is in the future")
    has_actual = any(pred.get(k) is not None for k in ACTUAL_ONLY)
    if kind == "shadow" and has_actual:
        raise ValueError("shadow ledger cannot carry bet confirmation fields")
    if kind == "actual" and not all(pred.get(k) is not None for k in ACTUAL_ONLY):
        raise ValueError(f"actual ledger requires {ACTUAL_ONLY}")
    if any(r.get("record") == "prediction" and r["prediction_id"] == pred["prediction_id"] for r in read(path)):
        raise ValueError("duplicate prediction_id")
    return _append(path, {"record": "prediction", "ledger_kind": kind,
                          "logged_at_utc": now.isoformat(), **{k: pred.get(k) for k in PREDICTION_FIELDS + ACTUAL_ONLY}})


def _prediction(path, pid):
    for r in read(path):
        if r.get("record") == "prediction" and r["prediction_id"] == pid:
            return r
    raise ValueError(f"unknown prediction_id {pid}")


def append_close(path, pid, close_line, close_price, close_opp_price, observed_at_utc, source):
    pred = _prediction(path, pid)
    if parse_utc(observed_at_utc) is None:
        raise ValueError("close needs an observed_at_utc")
    return _append(path, {"record": "close", "prediction_id": pid, "close_line": close_line, "close_price": close_price,
                          "close_opp_price": close_opp_price, "observed_at_utc": observed_at_utc, "source": source,
                          "ledger_kind": pred["ledger_kind"]})


def append_settlement(path, pid, outcome, settled_at_utc, source):
    pred = _prediction(path, pid)
    if outcome not in ("win", "loss", "push", "void"):
        raise ValueError("bad outcome")
    if any(r.get("record") == "settlement" and r["prediction_id"] == pid for r in read(path)):
        raise ValueError("already settled")
    price = pred["confirmed_price"] if pred["ledger_kind"] == "actual" else pred["price"]
    units = profit_per_unit(price) if outcome == "win" else (-1.0 if outcome == "loss" else 0.0)
    return _append(path, {"record": "settlement", "prediction_id": pid, "outcome": outcome, "units_per_unit_staked": units,
                          "settled_at_utc": settled_at_utc, "source": source, "ledger_kind": pred["ledger_kind"]})


def report(path):
    """Summary that labels itself with the ledger kind; shadow results are never called profit."""
    recs = read(path); verify(path)
    preds = {r["prediction_id"]: r for r in recs if r["record"] == "prediction"}
    sett = {r["prediction_id"]: r for r in recs if r["record"] == "settlement"}
    kinds = {r["ledger_kind"] for r in preds.values()}
    decided = [s for s in sett.values() if s["outcome"] in ("win", "loss")]
    w = sum(s["outcome"] == "win" for s in decided)
    clv = []
    for c in (r for r in recs if r["record"] == "close"):
        p = preds.get(c["prediction_id"])
        if p and c.get("close_opp_price") is not None and float(c["close_line"]) == float(p["line"]):
            clv.append(no_vig_pair(c["close_price"], c["close_opp_price"])[0] - no_vig_pair(p["price"], p["opp_price"])[0])
    label = "SHADOW (not wagered; hypothetical)" if kinds == {"shadow"} else ("ACTUAL" if kinds == {"actual"} else "MIXED-INVALID")
    return {"ledger": label, "predictions": len(preds), "settled": len(sett), "wins": w, "losses": len(decided) - w,
            "flat_units_per_bet": round(sum(s["units_per_unit_staked"] for s in decided) / len(decided), 4) if decided else None,
            "by_tier": {t: sum(1 for p in preds.values() if p["tier"] == t) for t in TIERS},
            "clv_same_line": {"n": len(clv), "mean_prob_pts": round(100 * sum(clv) / len(clv), 3) if clv else None,
                              "note": "positive = the closing price implied a higher win chance than the price we logged"}}
