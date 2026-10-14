"""Historical game-data quality and scheduling research in parallel per season."""
import argparse,json
from pathlib import Path
from collections import Counter
from v13.history import fetch_season
from v13.schedule_features import schedule_features
from v13.team_features import team_features

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--season",type=int,required=True)
    a=p.parse_args()
    games=fetch_season(a.season)
    if len(games)!=1230: raise RuntimeError(f"Incomplete season: {len(games)}")
    dates=[str(g.get("date",""))[:10] for g in games]
    ids=[g.get("id") for g in games]
    if len(set(ids))!=len(ids): raise RuntimeError("Duplicate game IDs")
    rest=schedule_features(games)
    team=team_features(games)
    if len(rest)!=len(games) or len(team)!=len(games): raise RuntimeError("Feature count mismatch")
    missing_scores=sum(g.get("home_team_score") is None or g.get("visitor_team_score") is None for g in games)
    report={"season":a.season,"games":len(games),"unique_dates":len(set(dates)),"missing_scores":missing_scores,"home_back_to_backs":sum(r["home_back_to_back"] for r in rest),"away_back_to_backs":sum(r["away_back_to_back"] for r in rest),"team_history_rows":sum(r["home_games_seen"]>0 and r["away_games_seen"]>0 for r in team),"source":"BALLDONTLIE /v1/games","note":"Schedule and scoring data; no verified travel mileage, odds, injuries or possession-adjusted ratings"}
    root=Path("research_data/season_quality");root.mkdir(parents=True,exist_ok=True)
    (root/f"quality_{a.season}.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
