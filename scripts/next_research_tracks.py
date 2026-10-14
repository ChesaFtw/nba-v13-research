"""Seven independent V13 research tracks with explicit evidence gates.

Tracks that lack verified point-in-time market data MUST NOT fabricate outcomes.
"""
import argparse,json,os,math
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import requests

TASKS=("historical_lines","injuries_lineups","play_by_play","prop_ensemble","real_line_backtest","confidence_calibration","daily_predictions")
def emit(task, status, **kwargs):
    path=Path("research_results/next_tracks");path.mkdir(parents=True,exist_ok=True)
    report={"task":task,"status":status,"generated_at_utc":datetime.now(timezone.utc).isoformat(),**kwargs}
    (path/f"{task}.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)

def api_probe(endpoint,params=None):
    key=os.environ.get("BALLDONTLIE_API_KEY")
    if not key:return {"state":"blocked","reason":"Missing API credential"}
    try:
        r=requests.get(endpoint,headers={"Authorization":key},params=params or {},timeout=35)
        result={"http_status":r.status_code,"url":r.url.split("?")[0]}
        if r.status_code!=200:
            result.update(state="unavailable_or_invalid_request",detail=r.text[:350])
            return result
        body=r.json();rows=body.get("data")
        result.update(state="accessible",sample_rows=len(rows) if isinstance(rows,list) else None,has_next_cursor=bool((body.get("meta") or {}).get("next_cursor")),sample_fields=sorted(rows[0])[:35] if isinstance(rows,list) and rows else [])
        return result
    except (requests.RequestException,ValueError) as exc:
        return {"state":"request_failed","reason":str(exc)[:300]}

def scan_jsonl(path):
    p=Path(path)
    if not p.is_file():return {"path":path,"present":False}
    n=0
    with p.open() as f:
        for line in f:
            if line.strip():json.loads(line);n+=1
    return {"path":path,"present":True,"rows":n}

def task_lines():
    # Opening odds requires documented parameters; probe returns HTTP response, never invent historical coverage.
    probe=api_probe("https://api.balldontlie.io/nba/v2/odds/opening",{"per_page":1})
    emit("historical_lines","source_probe",opening_odds=probe,required_for_backtest="Timestamped historical player prop lines and odds; opening game odds alone insufficient")

def task_injuries():
    probe=api_probe("https://api.balldontlie.io/v1/player_injuries",{"per_page":1})
    emit("injuries_lineups","source_probe",injury_snapshot=probe,limitation="Current snapshot cannot be used as historical pregame injury/lineup information")

def task_pbp():
    # Verify endpoint availability; do not assume historical rotation completeness.
    probe=api_probe("https://api.balldontlie.io/nba/v1/plays",{"per_page":1})
    emit("play_by_play","source_probe",plays=probe,limitation="A successful sample does not establish historical play-by-play or substitution coverage")

def task_ensemble():
    candidates=list(Path("research_data/prop_research").glob("props_*.jsonl"))
    if not candidates:
        emit("prop_ensemble","blocked",reason="No locally available historical prop features; model benchmark workflow is producing season artifacts",required="Download and validate historical feature artifacts before fitting ensemble")
        return
    from scripts.benchmark_prop_models import evaluate
    results=[]
    for p in candidates:
        rows=[json.loads(line) for line in p.open() if line.strip()]
        season=int(p.stem.rsplit("_",1)[-1])
        results.append(evaluate(rows,season))
    emit("prop_ensemble","baseline_comparison_complete",seasons=results,limitation="This compares rolling-feature baselines, not a trained advanced ensemble")

def task_backtest():
    paths=[scan_jsonl("research_data/market_history/player_props.jsonl"),scan_jsonl("research_data/prop_research/props_2025.jsonl")]
    if not all(x.get("present") and x.get("rows",0)>0 for x in paths):
        emit("real_line_backtest","blocked",inputs=paths,reason="Missing historical player prop lines or model feature rows; no legitimate bet grading possible")
        return
    emit("real_line_backtest","inputs_present_requires_schema_audit",inputs=paths,reason="Point-in-time timestamps and matching game/player/market keys must be verified before settlement")

def task_calibration():
    p=Path("research_results/backtests/real_line_backtest.json")
    if not p.is_file():
        emit("confidence_calibration","blocked",reason="No verified out-of-sample wager-level probability/outcome pairs",required=str(p))
        return
    data=json.loads(p.read_text())
    bets=data.get("bets",[])
    usable=[b for b in bets if isinstance(b.get("predicted_probability"),(int,float)) and b.get("result") in ("win","loss")]
    if len(usable)<100:
        emit("confidence_calibration","blocked",reason="Insufficient verified probability/outcome pairs",usable=len(usable))
        return
    bins=defaultdict(list)
    for b in usable:
        prob=b["predicted_probability"]
        if not 0<=prob<=1:continue
        bins[min(9,int(prob*10))].append((prob,int(b["result"]=="win")))
    table=[{"bucket":k,"n":len(v),"mean_prediction":sum(a for a,b in v)/len(v),"actual_hit_rate":sum(b for a,b in v)/len(v)} for k,v in sorted(bins.items())]
    emit("confidence_calibration","descriptive_reliability_only",bins=table,limitation="Requires independent holdout to validate any recalibration")

def task_daily():
    # Only write a template. No invented games, odds, or actionable picks.
    path=Path("research_results/next_tracks/daily_prediction_schema.json")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({"required_fields":["created_at_utc","game_id","player_id","market","projection","model_version","feature_cutoff_utc","data_sources"],"optional_fields":["verified_market_line","verified_price","estimated_win_probability","confidence_score"],"policy":"Predictions must be written before game start, immutable, and never represented as placed bets"},indent=2)+"\n")
    emit("daily_predictions","schema_created",schema=str(path),reason="Automated pregame forecasts require schedule ingestion and validated production model; not yet issuing picks")

def main():
    p=argparse.ArgumentParser();p.add_argument("--task",choices=TASKS,required=True);a=p.parse_args()
    {"historical_lines":task_lines,"injuries_lineups":task_injuries,"play_by_play":task_pbp,"prop_ensemble":task_ensemble,"real_line_backtest":task_backtest,"confidence_calibration":task_calibration,"daily_predictions":task_daily}[a.task]()
if __name__=="__main__":main()
