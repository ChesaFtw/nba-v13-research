"""Aggregate existing season holdouts without additional API downloads."""
import json,sys
from pathlib import Path
def main():
    files=sorted(Path("research_results/input").rglob("trained_*.json"))
    if len(files)!=4:raise SystemExit(f"Expected four season results; found {len(files)}")
    rows=[];market_summary={}
    for f in files:
        data=json.loads(f.read_text())
        for market,values in data["markets"].items():
            n=int(data["holdout_n"])
            rows.append({"season":data["season"],"market":market,"holdout_n":n,**{k:values[k] for k in ("ridge_mae","last10_mae","ridge_vs_last10_mae_gain")}})
    for market in sorted({r["market"] for r in rows}):
        group=[r for r in rows if r["market"]==market]
        n=sum(r["holdout_n"] for r in group)
        gain=sum(r["ridge_vs_last10_mae_gain"]*r["holdout_n"] for r in group)/n
        market_summary[market]={"holdout_rows":n,"weighted_mae_gain":round(gain,5),"ridge_wins_seasons":sum(r["ridge_mae"]<r["last10_mae"] for r in group),"promote_ridge":gain>0 and sum(r["ridge_mae"]<r["last10_mae"] for r in group)>=3}
    report={"status":"complete","per_season":rows,"markets":market_summary,"caveat":"Ridge compared to last10 on identical holdouts; no historical sportsbook odds or validated betting win rate. Data for each season is independently modeled."}
    out=Path("research_results/model_review");out.mkdir(parents=True,exist_ok=True)
    (out/"ridge_four_season_review.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
