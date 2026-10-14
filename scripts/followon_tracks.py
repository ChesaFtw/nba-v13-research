"""Summarize seven-track artifacts and explicitly gate follow-on work.

This does not infer availability from an Actions job succeeding: the JSON
status field is authoritative for whether a track produced usable output.
"""
import argparse,json
from pathlib import Path

TASKS=("historical_lines","injuries_lineups","play_by_play","prop_ensemble","real_line_backtest","confidence_calibration","daily_predictions")
def main():
    p=argparse.ArgumentParser();p.add_argument("--input",default="research_results/track_inputs");p.add_argument("--out",default="research_results/followon");a=p.parse_args()
    src=Path(a.input);dest=Path(a.out);dest.mkdir(parents=True,exist_ok=True)
    results={}
    for task in TASKS:
        files=list(src.rglob(f"{task}.json"))
        if not files:
            results[task]={"status":"missing_artifact","next":"No automatic downstream execution"}
            continue
        item=json.loads(files[0].read_text())
        status=item.get("status","unknown")
        results[task]={"status":status,"next":{
            "historical_lines":"Require timestamped actual prop market records before line-based backtest",
            "injuries_lineups":"Require historical point-in-time archives; snapshot is not a backtest source",
            "play_by_play":"Verify seasons, plays, and substitutions before rotation features",
            "prop_ensemble":"Download season feature artifacts; fit on past and evaluate future games",
            "real_line_backtest":"Only grade when historical odds and features match by timestamp",
            "confidence_calibration":"Require out-of-sample probabilities and resolved outcomes",
            "daily_predictions":"Requires a trained versioned model and pregame schedule feed",
        }[task],"source":str(files[0])}
    (dest/"followon_status.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results,indent=2),flush=True)
if __name__=="__main__":main()
