"""Research-only: fit minutes correction for >=3 missed team games."""
from statistics import mean
from research.return_gap_probe import cluster_ci

WINDOWS=(("2026-01-15","2026-02-15"),("2026-02-15","2026-03-12"),("2026-03-12","2026-04-13"))

def fit_gap_correction(rows,start,min_train=30):
    train=[r for r in rows if r["date"]<start and r["gap_team_games"]>=3]
    if len(train)<min_train:
        raise ValueError("insufficient prior returner appearances")
    return max(-4.0,min(0.0,mean(r["signed_error_last10"] for r in train))),len(train)

def evaluate(rows,windows=WINDOWS):
    windows=tuple(windows)
    previous=None
    for start,end in windows:
        if start>=end or (previous is not None and start<previous):
            raise ValueError("overlapping or invalid chronological windows")
        previous=end
    results={}
    for start,end in windows:
        adjustment,train_n=fit_gap_correction(rows,start)
        test=[r for r in rows if start<=r["date"]<end and r["gap_team_games"]>=3]
        if len({r["game_id"] for r in test})<15:
            raise ValueError("insufficient independent held-out games")
        improvements=[{"game_id":r["game_id"],
            "signed_error_last10":abs(r["minutes"]-r["last10"])-abs(r["minutes"]-r["last10"]-adjustment)}
            for r in test]
        results[start]={
            "train_n":train_n,"test_n":len(test),
            "correction_minutes":round(adjustment,3),
            "baseline_mae":round(mean(abs(r["minutes"]-r["last10"]) for r in test),3),
            "challenger_mae":round(mean(abs(r["minutes"]-r["last10"]-adjustment) for r in test),3),
            "mae_gain":round(mean(r["signed_error_last10"] for r in improvements),3),
            "gain_game_cluster_ci95":cluster_ci(improvements)}
    return results
