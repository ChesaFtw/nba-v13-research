"""Grade pregame V13 selections against real historical thresholds at assumed -110.

Input JSONL: prediction_time, game_time, market, side, line, actual,
probability, game_id, model_version, line_source, line_observed_at.
Prices are hypothetical -110; thresholds must come from sourced historical lines.
"""
import argparse, json, math
from datetime import datetime, timezone
from pathlib import Path

def timestamp(value):
    if not value: raise ValueError("Missing timestamp")
    dt=datetime.fromisoformat(value.replace("Z","+00:00"))
    if dt.tzinfo is None: raise ValueError("Timezone required")
    return dt.astimezone(timezone.utc)

def grade(rows, starting_bankroll=10000, stake_fraction=0.01):
    if not 0 < stake_fraction <= .02: raise ValueError("Stake must be 0-2%")
    seen=set(); bankroll=float(starting_bankroll); peak=bankroll
    counts={"win":0,"loss":0,"push":0}
    ledger=[]
    for row in sorted(rows,key=lambda x:(x["game_time"],x["game_id"],x["market"])):
        required=("prediction_time","game_time","market","side","line","actual","probability","game_id","model_version","line_source","line_observed_at")
        if any(k not in row or row[k] is None for k in required): raise ValueError("Missing required field")
        key=(row["game_id"],row.get("player_id"),row["market"],row["side"],float(row["line"]))
        if key in seen: raise ValueError("Duplicate selection")
        seen.add(key)
        if timestamp(row["prediction_time"])>=timestamp(row["game_time"]) or timestamp(row["line_observed_at"])>=timestamp(row["game_time"]):
            raise ValueError("Post-tip prediction or market line")
        if timestamp(row["prediction_time"])<timestamp(row["line_observed_at"]):
            raise ValueError("Line observed after prediction")
        prob=float(row["probability"])
        if not 0<=prob<=1: raise ValueError("Invalid probability")
        line=float(row["line"]); actual=float(row["actual"])
        if not math.isfinite(line) or not math.isfinite(actual): raise ValueError("Invalid number")
        side=row["side"].lower()
        if side not in ("over","under"): raise ValueError("Use over/under for normalized selection")
        outcome="push" if actual==line else ("win" if (actual>line)==(side=="over") else "loss")
        stake=round(bankroll*stake_fraction,2)
        pnl=round(stake*(100/110 if outcome=="win" else -1 if outcome=="loss" else 0),2)
        bankroll=round(bankroll+pnl,2);peak=max(peak,bankroll)
        counts[outcome]+=1
        ledger.append({"game_id":row["game_id"],"market":row["market"],"side":side,"line":line,"actual":actual,"outcome":outcome,"probability":prob,"stake":stake,"pnl":pnl,"bankroll":bankroll,"drawdown_pct":round(100*(peak-bankroll)/peak,3)})
    decided=counts["win"]+counts["loss"]
    return {"pricing":"assumed -110, NOT executable historical price","selections":len(ledger),"record":counts,"hit_rate":round(counts["win"]/decided,5) if decided else None,"start_bankroll":starting_bankroll,"end_bankroll":bankroll,"ledger":ledger}

def main():
    p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--out",required=True);p.add_argument("--stake-fraction",type=float,default=.01)
    a=p.parse_args()
    rows=[json.loads(x) for x in Path(a.input).read_text().splitlines() if x.strip()]
    if not rows: raise ValueError("No historical sourced market lines: refusing to invent results")
    result=grade(rows,stake_fraction=a.stake_fraction)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="ledger"},indent=2))
if __name__=="__main__":main()
