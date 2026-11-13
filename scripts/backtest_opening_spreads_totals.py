"""Backtest 2025 NBA spread/total baseline against sourced opening thresholds.

Uses BALLDONTLIE_API_KEY from CI secret, never prints it. All prices fixed -110.
One preferred vendor per game (deterministic ordering), no postgame line selection.
"""
import argparse,json,os,time,requests
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone,timedelta
from v13.pricing import pregame_status

URL="https://api.balldontlie.io/nba/v2/odds/opening"
VENDORS=("draftkings","fanduel","betmgm","caesars","betrivers","fanatics")
def load(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def fetch(ids,key):
    allrows=[];session=requests.Session()
    for start in range(0,len(ids),10):
        batch=ids[start:start+10];cursor=None;visited=set()
        while True:
            params=[("game_ids[]",str(x)) for x in batch]+[("per_page","100")]
            if cursor is not None:params.append(("cursor",str(cursor)))
            for attempt in range(7):
                r=session.get(URL,headers={"Authorization":key},params=params,timeout=45)
                if r.status_code in (429,500,502,503,504):
                    if attempt==6:r.raise_for_status()
                    time.sleep(min(5*(attempt+1),30));continue
                if r.status_code in (401,403):raise RuntimeError(f"Opening odds API access denied (HTTP {r.status_code}); verify subscription GOAT tier")
                r.raise_for_status();break
            payload=r.json()
            if not isinstance(payload.get("data"),list):raise ValueError("Malformed odds response")
            allrows.extend(payload["data"])
            cursor=(payload.get("meta") or {}).get("next_cursor")
            if cursor is None:break
            if str(cursor) in visited:raise ValueError("Repeated odds cursor")
            visited.add(str(cursor))
        if start%100==0:print(f"Opening odds progress: {min(start+10,len(ids))}/{len(ids)} games; {len(allrows)} vendor rows",flush=True)
        time.sleep(.25)
    return allrows

def choose(rows,tips=None):
    """One opening row per game. With tips (game_id -> UTC tip), rows not opened before tip are dropped."""
    result={}
    for r in rows:
        gid=str(r.get("game_id"));vendor=str(r.get("vendor") or "").lower()
        opened=r.get("opened_at")
        if not opened:continue
        if tips is not None and pregame_status(opened,tips.get(gid))!="pregame":continue
        try:stamp=datetime.fromisoformat(opened.replace("Z","+00:00"))
        except (ValueError,TypeError):continue
        if stamp.tzinfo is None:continue
        # Prefer pre-specified vendor priority, not whichever line looks favorable.
        priority=(VENDORS.index(vendor) if vendor in VENDORS else 99,stamp.isoformat(),str(r.get("id","")))
        if gid not in result or priority<result[gid][0]:result[gid]=(priority,r)
    return {k:v[1] for k,v in result.items()}

def backtest(predictions,odds,min_spread_edge=3.,min_total_edge=5.):
    bygame=choose(odds);bankroll=10000.;peak=bankroll;ledger=[];coverage=Counter()
    for p in sorted(predictions,key=lambda x:(x["date"],str(x["game_id"]))):
        gid=str(p["game_id"]);r=bygame.get(gid)
        if r is None:coverage["no_opening_odds"]+=1;continue
        coverage["games_with_opening_odds"]+=1
        # Date-only game index: opening timestamp may be next UTC date for late games.
        opened=datetime.fromisoformat(r["opened_at"].replace("Z","+00:00"))
        game_date=datetime.fromisoformat(p["date"]).replace(tzinfo=timezone.utc)
        if opened>=game_date+timedelta(days=2):
            coverage["invalid_opening_timestamp"]+=1;continue
        for market,field,pred,actual,minimum in (
            ("spread","spread_home_value",float(p["predicted_home_margin"]),float(p["actual_home_margin"]),min_spread_edge),
            ("total","total_value",float(p["predicted_total"]),float(p["actual_total"]),min_total_edge)):
            raw=r.get(field)
            if raw is None:coverage[f"missing_{market}_line"]+=1;continue
            try:line=float(raw)
            except (TypeError,ValueError):coverage[f"invalid_{market}_line"]+=1;continue
            edge=pred+line if market=="spread" else pred-line
            if abs(edge)<minimum:coverage[f"below_{market}_edge"]+=1;continue
            actual_delta=actual+line if market=="spread" else actual-line
            outcome="push" if abs(actual_delta)<1e-8 else ("win" if (actual_delta>0)==(edge>0) else "loss")
            pick=("home" if edge>0 else "away") if market=="spread" else ("over" if edge>0 else "under")
            stake=round(bankroll*.01,2)
            pnl=round(stake*(100/110 if outcome=="win" else -1 if outcome=="loss" else 0),2)
            bankroll=round(bankroll+pnl,2);peak=max(peak,bankroll)
            ledger.append({"game_id":gid,"date":p["date"],"market":market,"pick":pick,"line":line,"predicted":pred,"actual":actual,"edge":round(abs(edge),3),"vendor":r.get("vendor"),"opened_at":r["opened_at"],"outcome":outcome,"stake":stake,"pnl":pnl,"bankroll":bankroll,"drawdown_pct":round(100*(peak-bankroll)/peak,3)})
    markets={}
    for market in ("spread","total"):
        subset=[x for x in ledger if x["market"]==market]
        c=Counter(x["outcome"] for x in subset);decided=c["win"]+c["loss"]
        markets[market]={"bets":len(subset),"record":dict(c),"hit_rate":round(c["win"]/decided,4) if decided else None,"net_pnl":round(sum(x["pnl"] for x in subset),2)}
    return {"model":"rolling_team_10_home3_v1","season":"2025-26","price_assumption":"-110 for every bet (not observed odds)","line_type":"historical opening odds","vendor_priority":VENDORS,"min_spread_edge":min_spread_edge,"min_total_edge":min_total_edge,"games_with_predictions":len(predictions),"coverage":dict(coverage),"markets":markets,"starting_bankroll":10000,"ending_bankroll":bankroll,"max_drawdown_pct":max((x["drawdown_pct"] for x in ledger),default=0),"ledger":ledger}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--predictions",required=True);ap.add_argument("--out",required=True);ap.add_argument("--probe-only",action="store_true")
    a=ap.parse_args();key=os.getenv("BALLDONTLIE_API_KEY")
    if not key:raise RuntimeError("BALLDONTLIE_API_KEY not set")
    predictions=load(a.predictions)
    if len(predictions)<900:raise ValueError("Incomplete predictions")
    selected=predictions[:10] if a.probe_only else predictions
    rows=fetch([x["game_id"] for x in selected],key)
    result=backtest(selected,rows)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="ledger"},indent=2))
    if not rows:raise RuntimeError("API returned no historical opening odds; backtest cannot be validated")
if __name__=="__main__":main()
