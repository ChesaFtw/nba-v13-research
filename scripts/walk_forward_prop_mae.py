"""Leakage-safe expanding-window benchmark on precomputed player features.

This is a projection MAE experiment, never a sportsbook backtest.
Input: JSONL from scripts.build_prop_research (pregame rolling features).
Each evaluation block trains only on earlier dates; rolling baseline is frozen.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from v13.prop_features import MARKETS

def evaluate(rows, min_train_dates=60, test_dates=20, alpha=25.0):
    dates = sorted({r["game_date"] for r in rows})
    if len(dates) <= min_train_dates:
        raise ValueError("Insufficient dates for walk-forward evaluation")
    if len({(r["player_id"], r["game_date"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate player-date records")
    results = {}
    for market in MARKETS:
        keys = [f"{market}_avg_{n}" for n in (5, 10, 20)]
        actual_key = f"{market}_actual"
        errors = {"last10": [], "ridge": []}
        folds = []
        for start in range(min_train_dates, len(dates), test_dates):
            cutoff = dates[start]
            end = dates[min(start + test_dates, len(dates)) - 1]
            train = [r for r in rows if r["game_date"] < cutoff]
            test = [r for r in rows if cutoff <= r["game_date"] <= end]
            if len(train) < 100 or not test:
                continue
            X = np.asarray([[float(r[k]) for k in keys] for r in train])
            y = np.asarray([float(r[actual_key]) for r in train])
            Z = np.asarray([[float(r[k]) for k in keys] for r in test])
            target = np.asarray([float(r[actual_key]) for r in test])
            if not (np.isfinite(X).all() and np.isfinite(y).all() and np.isfinite(Z).all() and np.isfinite(target).all()):
                raise ValueError("Non-finite feature or target")
            mu, sd = X.mean(0), X.std(0)
            sd[sd < 1e-8] = 1
            A = (X - mu) / sd
            coef = np.linalg.solve(A.T @ A + alpha * np.eye(len(keys)), A.T @ (y - y.mean()))
            ridge = y.mean() + ((Z - mu) / sd) @ coef
            baseline = Z[:, 1]
            errors["ridge"].extend(np.abs(ridge - target).tolist())
            errors["last10"].extend(np.abs(baseline - target).tolist())
            folds.append({"train_before": cutoff, "test_through": end, "train_n": len(train), "test_n": len(test)})
        if not folds:
            raise ValueError("No valid chronological folds")
        results[market] = {"n": len(errors["ridge"]), "ridge_mae": round(float(np.mean(errors["ridge"])), 5),
                           "last10_mae": round(float(np.mean(errors["last10"])), 5),
                           "ridge_minus_last10": round(float(np.mean(errors["ridge"]) - np.mean(errors["last10"])), 5),
                           "folds": folds}
    return {"markets": results, "kind": "pregame projection MAE, not betting hit rate",
            "data_scope": "Input must be independently verified regular-season-only; this function does not validate source coverage."}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-train-dates", type=int, default=60)
    p.add_argument("--test-dates", type=int, default=20)
    a = p.parse_args()
    rows = [json.loads(line) for line in Path(a.input).open() if line.strip()]
    report = evaluate(rows, a.min_train_dates, a.test_dates)
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: {x: v for x, v in val.items() if x != "folds"} for k, val in report["markets"].items()}, indent=2))

if __name__ == "__main__":
    main()
