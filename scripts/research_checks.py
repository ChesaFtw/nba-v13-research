"""Auditable research checks from existing historical NBA game records."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path
from v13.baseline import evaluate_seasons, summarize
from v13.schedule_features import schedule_features
from v13.team_features import team_features

TASKS = ("data_quality","rest_features","team_features","home_away","elo_holdout","elo_calibration","totals_baseline","scoring_distribution","confidence_audit","readiness")

def run(task, root):
    seasons=[]
    for season in (2022,2023,2024,2025):
        p=root/f"nba_games_regular_{season}.jsonl"
        if not p.exists(): raise RuntimeError(f"Missing {p}")
        games=[json.loads(line) for line in p.open() if line.strip()]
        if len(games)!=1230: raise RuntimeError(f"Season {season}: expected 1230 games, got {len(games)}")
        ids=[g["id"] for g in games]
        if len(set(ids))!=len(ids): raise RuntimeError(f"Duplicate games in {season}")
        seasons.append((season,games))
    all_games=[g for _,gs in seasons for g in gs]
    report={"task":task,"games":len(all_games),"seasons":[s for s,_ in seasons],"status":"complete","limitations":[]}
    if task=="data_quality":
        report["unique_ids"]=len(set(g["id"] for g in all_games))
        if report["unique_ids"]!=len(all_games): raise RuntimeError("Duplicate IDs across seasons")
    elif task=="rest_features":
        # Reset across seasons; offseason rest days are not useful.
        results=[r for _,gs in seasons for r in schedule_features(gs)]
        report["rows"]=len(results)
        report["back_to_back_home"]=sum(r["home_back_to_back"] for r in results)
        report["back_to_back_away"]=sum(r["away_back_to_back"] for r in results)
    elif task=="team_features":
        rows=[r for _,gs in seasons for r in team_features(gs)]
        report["rows"]=len(rows)
        report["history_available"]=sum(r["home_points_for_avg"] is not None and r["away_points_for_avg"] is not None for r in rows)
        report["limitations"].append("Scoring averages are not possession-adjusted efficiency ratings")
    elif task=="home_away":
        finals=[g for g in all_games if g.get("status")=="Final"]
        report["home_win_rate"]=round(sum(g["home_team_score"]>g["visitor_team_score"] for g in finals)/len(finals),5)
    elif task in ("elo_holdout","elo_calibration","confidence_audit"):
        records=evaluate_seasons(seasons)
        holdout=[r for r in records if r["season"]==2025]
        report["holdout_season"]=2025
        report["holdout"]=summarize(holdout)
        report["limitations"].append("Elo parameters were not independently tuned; this is an observational season slice, not a certified untouched holdout")
        if task!="elo_holdout":
            bins=[]
            for lo in range(0,10):
                bucket=[r for r in holdout if lo/10<=r["p_home"]<(lo+1)/10]
                if bucket:
                    bins.append({"bin":lo,"n":len(bucket),"mean_prediction":round(sum(r["p_home"] for r in bucket)/len(bucket),4),"observed_home_win":round(sum(r["actual_home_win"] for r in bucket)/len(bucket),4)})
            report["calibration_bins"]=bins
            report["limitations"].append("Confidence grades are not calibrated from these bins")
    elif task=="totals_baseline":
        totals=[g["home_team_score"]+g["visitor_team_score"] for g in all_games if g.get("status")=="Final"]
        report["mean_total"]=round(sum(totals)/len(totals),3)
        report["limitations"].append("Descriptive totals only; no point-in-time projection or market comparison")
    elif task=="scoring_distribution":
        margins=[g["home_team_score"]-g["visitor_team_score"] for g in all_games if g.get("status")=="Final"]
        report["mean_margin"]=round(sum(margins)/len(margins),3)
        report["margin_sd"]=round(math.sqrt(sum((v-sum(margins)/len(margins))**2 for v in margins)/len(margins)),3)
    elif task=="readiness":
        report["ready_for_live_wagers"]=False
        report["missing"]=["historical market lines","point-in-time injury history","verified prop probability calibration","live prediction tracking integration"]
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--task",required=True,choices=TASKS)
    p.add_argument("--data",default="research_data")
    a=p.parse_args()
    report=run(a.task,Path(a.data))
    out=Path("research_results/research_checks")
    out.mkdir(parents=True,exist_ok=True)
    (out/f"{a.task}.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
