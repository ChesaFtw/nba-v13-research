"""Score sourced NBA spread and total predictions at assumed -110.

One JSONL record per game/market with home_score, away_score, line,
prediction (predicted home margin for spread; predicted combined points for total),
game_id, game_time, prediction_time, line_observed_at, line_source, model_version.
A positive spread line is home handicap (e.g. -4.5 home favorite).
"""
import argparse, json, math
from datetime import datetime,timezone
from pathlib import Path

def stamp(s):
    t=datetime.fromisoformat(s.replace("Z","+00:00"))
    if t.tzinfo is None:raise ValueError("Timestamp must be timezone-aware")
    return t.astimezone(timezone.utc)

def evaluate(rows, bankroll=10000., fraction=.01, min_edge=0.):
    if not 0<fraction<=.02:raise ValueError("Stake fraction outside allowed range")
    seen=set();ledger=[];peak=bankroll
    for r in sorted(rows,key=lambda x:(x["game_time"],str(x["game_id"]),x["market"])):
        required=("game_id","game_time","prediction_time","line_observed_at","line_source","model_version","market","line","prediction","home_score","away_score")
        if any(k not in r or r[k] is None for k in required):raise ValueError("Missing sourced game prediction/market field")
        market=r["market"]
        if market not in ("spread","total"):raise ValueError("Unsupported market")
        key=(r["game_id"],market)
        if key in seen:raise ValueError("Duplicate game market")
        seen.add(key)
        if not stamp(r["line_observed_at"])<=stamp(r["prediction_time"])<stamp(r["game_time"]):raise ValueError("Line/prediction unavailable before tip")
        line=float(r["line"]);pred=float(r["prediction"]);home=float(r["home_score"]);away=float(r["away_score"])
        if not all(map(math.isfinite,(line,pred,home,away))):raise ValueError("Nonfinite number")
        if home<0 or away<0:raise ValueError("Negative score")
        if market=="spread":
            signal=pred+line
            result=home-away+line
            pick="home" if signal>0 else "away"
        else:
            signal=pred-line
            result=home+away-line
            pick="over" if signal>0 else "under"
        if abs(signal)<=min_edge:continue
        outcome="push" if result==0 else ("win" if (result>0)==(signal>0) else "loss")
        stake=round(bankroll*fraction,2)
        pnl=round(stake*(100/110 if outcome=="win" else -1 if outcome=="loss" else 0),2)
        bankroll=round(bankroll+pnl,2);peak=max(peak,bankroll)
        ledger.append({"game_id":r["game_id"],"market":market,"pick":pick,"line":line,"prediction":pred,"outcome":outcome,"stake":stake,"pnl":pnl,"bankroll":bankroll,"drawdown_pct":round(100*(peak-bankroll)/peak,3)})
    report={}
    for market in ("spread","total"):
        entries=[x for x in ledger if x["market"]==market]
        w=sum(x["outcome"]=="win" for x in entries);l=sum(x["outcome"]=="loss" for x in entries)
        report[market]={"bets":len(entries),"wins":w,"losses":l,"pushes":len(entries)-w-l,"hit_rate":round(w/(w+l),5) if w+l else None,"net_profit":round(sum(x["pnl"] for x in entries),2)}
    return {"pricing":"hypothetical -110","markets":report,"starting_bankroll":10000.,"ending_bankroll":bankroll,"ledger":ledger}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True);p.add_argument("--out",required=True)
    p.add_argument("--min-edge",type=float,default=0.)
    a=p.parse_args()
    rows=[json.loads(s) for s in Path(a.input).read_text().splitlines() if s.strip()]
    if not rows:raise ValueError("No historical line/prediction records; refusing fabricated backtest")
    result=evaluate(rows,min_edge=a.min_edge)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="ledger"},indent=2))
if __name__=="__main__":main()
