"""Inventory each missing V13 research capability without fabricating unavailable datasets."""
import argparse,json
from pathlib import Path
from datetime import datetime,timezone
REQUIREMENTS={
"historical_prop_lines":{"inputs":["research_data/market_history/player_props.jsonl"],"why":"Historical executable prop prices and lines, with timestamps"},
"historical_injuries_lineups":{"inputs":["research_data/injuries/history.jsonl","research_data/lineups/history.jsonl"],"why":"Point-in-time injury and starting lineup archives"},
"play_by_play_rotations":{"inputs":["research_data/play_by_play/history.jsonl"],"why":"Possessions, substitutions and minutes"},
"defensive_matchups":{"inputs":["research_data/matchups/defense.jsonl"],"why":"Verified opponent/position defensive features"},
"coaching_roster_history":{"inputs":["research_data/rosters/changes.jsonl"],"why":"Coaching, trades and rotation history"},
"prop_models":{"inputs":["research_results/models/prop_model_manifest.json"],"why":"Fitted prop model with dated training window"},
"ensemble_comparison":{"inputs":["research_results/models/ensemble_comparison.json"],"why":"Out-of-sample model comparison"},
"real_line_backtest":{"inputs":["research_results/backtests/real_line_backtest.json"],"why":"Walk-forward graded wagers at actual historical lines"},
"confidence_calibration":{"inputs":["research_results/calibration/confidence_report.json"],"why":"Independent confidence reliability and grade monotonicity"},
"daily_prediction_tracking":{"inputs":["research_results/live/prediction_ledger.jsonl"],"why":"Timestamped predictions written before game outcomes"},
}
def audit(task,base):
    item=REQUIREMENTS[task]
    checks=[]
    for path in item["inputs"]:
        p=base/path
        checks.append({"path":path,"exists":p.is_file(),"bytes":p.stat().st_size if p.is_file() else 0})
    ready=all(x["exists"] and x["bytes"]>0 for x in checks)
    return {"task":task,"state":"inputs_present_needs_validation" if ready else "blocked_missing_verified_inputs","checks":checks,"purpose":item["why"],"checked_at_utc":datetime.now(timezone.utc).isoformat(),"caution":"A file existing does not establish trustworthy historical availability, model validity, or betting edge"}
def main():
    p=argparse.ArgumentParser();p.add_argument("--task",choices=REQUIREMENTS,required=True);p.add_argument("--base",default=".");a=p.parse_args()
    report=audit(a.task,Path(a.base))
    out=Path("research_results/readiness");out.mkdir(parents=True,exist_ok=True)
    (out/f"{a.task}.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
