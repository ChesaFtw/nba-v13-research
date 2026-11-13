"""Research-only cluster bootstrap for paired pregame minutes forecasts."""
from datetime import datetime
from math import isfinite
from random import Random

def audit_minutes(rows, *, repetitions=2000, seed=0):
    """Paired game-cluster bootstrap. Inputs must be independently held-out."""
    rows = list(rows)
    if not isinstance(repetitions, int) or repetitions < 200:
        raise ValueError("repetitions must be at least 200")
    groups, base_errors, new_errors = {}, [], []
    for row in rows:
        game = str(row.get("game_id") or "").strip()
        if not game:
            raise ValueError("game_id required")
        pred = datetime.fromisoformat(str(row["predicted_at_utc"]).replace("Z","+00:00"))
        tip = datetime.fromisoformat(str(row["tipoff_at_utc"]).replace("Z","+00:00"))
        if pred.tzinfo is None or tip.tzinfo is None or pred >= tip:
            raise ValueError("prediction must precede tipoff with timezone-aware timestamps")
        actual, base, new = (float(row[k]) for k in ("actual","baseline","challenger"))
        if not all(map(isfinite,(actual,base,new))):
            raise ValueError("minutes must be finite")
        if not 0 <= actual <= 60:
            raise ValueError("actual minutes outside plausible NBA game range")
        a,b=abs(actual-base),abs(actual-new)
        base_errors.append(a); new_errors.append(b)
        groups.setdefault(game,[]).append(a-b)
    if len(groups)<5:
        raise ValueError("at least five independent games required")
    keys=sorted(groups); rng=Random(seed); boot=[]
    for _ in range(repetitions):
        sampled=rng.choices(keys,k=len(keys))
        boot.append(sum(sum(groups[k]) for k in sampled)/sum(len(groups[k]) for k in sampled))
    boot.sort(); n=len(rows)
    return {"n":n,"games":len(groups),"baseline_mae":sum(base_errors)/n,
            "challenger_mae":sum(new_errors)/n,
            "mae_improvement":(sum(base_errors)-sum(new_errors))/n,
            "ci95":(boot[int(.025*(repetitions-1))],boot[int(.975*(repetitions-1))]),
            "lower_bound_positive":boot[int(.025*(repetitions-1))]>0,
            "repetitions":repetitions}
