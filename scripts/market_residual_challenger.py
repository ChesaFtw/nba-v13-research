"""Chronological market-anchored challenger for NBA spreads and totals.

Uses pregame rolling-score and rest projections, then learns residuals against
historical opening lines. Fits only early-season games, chooses thresholds only
on validation, and reports untouched final-season test.
"""
import argparse,json,math
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from scripts.opponent_rest_challenger import project
from scripts.backtest_opening_spreads_totals import choose,fetch
from scripts.holdout_spreads_totals import sim

def run(games,odds):
 preds=project(games);bygame=choose(odds);rows=[]
 for p in preds:
  o=bygame.get(str(p["game_id"]))
  if not o:continue
  try:
   spread=float(o["spread_home_value"]);total=float(o["total_value"])
  except (ValueError,TypeError,KeyError):continue
  if not all(math.isfinite(x) for x in (spread,total)):continue
  rows.append({"date":p["date"],"game_id":p["game_id"],"spread_line":spread,
   "total_line":total,"margin":float(p["actual_home_margin"]),"total":float(p["actual_total"]),
   "pred_margin":float(p["predicted_home_margin"]),"pred_total":float(p["predicted_total"]),
   "rest_delta":float(p["home_rest"]-p["away_rest"])})
 dates=sorted({r["date"] for r in rows})
 if len(dates)<100:raise RuntimeError("Insufficient dated opening lines")
 t1=dates[int(len(dates)*.55)];t2=dates[int(len(dates)*.8)]
 report={"model":"opening_market_residual_ridge_v1","price":"assumed -110","train_end":t1,
  "test_start":t2,"matched_games":len(rows),"markets":{}}
 for market in ("spread","total"):
  def features(r):
   if market=="spread":
    m=-r["spread_line"];p=r["pred_margin"];rest=r["rest_delta"]
    return [p-m,rest,abs(m),p]
   m=r["total_line"];p=r["pred_total"]
   return [p-m,m,p,abs(r["rest_delta"])]
  def actual(r):return r["margin"] if market=="spread" else r["total"]
  def market_value(r):return -r["spread_line"] if market=="spread" else r["total_line"]
  train=[r for r in rows if r["date"]<t1];val=[r for r in rows if t1<=r["date"]<t2];test=[r for r in rows if r["date"]>=t2]
  X=np.asarray([features(r) for r in train]);y=np.asarray([actual(r)-market_value(r) for r in train])
  mu=X.mean(axis=0);sd=X.std(axis=0);sd[sd<1e-9]=1.
  candidates=[]
  for alpha in (100,1000,10000):
   model=Ridge(alpha=alpha).fit((X-mu)/sd,y)
   def convert(items):
    if not items:return []
    preds=model.predict((np.asarray([features(r) for r in items])-mu)/sd)
    return [{"date":r["date"],"game_id":r["game_id"],"market":market,
      "edge":float(e),"delta":actual(r)-market_value(r)} for r,e in zip(items,preds)]
   vr=convert(val)
   for threshold in ((1,1.5,2,2.5,3) if market=="spread" else (1.5,2,3,4,5)):
    s=sim(vr,threshold)
    if s["bets"]>=20:candidates.append((s["return_pct"],s["bets"],alpha,threshold,model))
  if not candidates:
   report["markets"][market]={"status":"insufficient validation selections"};continue
  _,_,alpha,threshold,model=max(candidates,key=lambda c:(c[0],-c[1]))
  def selections(items):
   if not items:return []
   p=model.predict((np.asarray([features(r) for r in items])-mu)/sd)
   return [{"date":r["date"],"game_id":r["game_id"],"market":market,
    "edge":float(e),"delta":actual(r)-market_value(r)} for r,e in zip(items,p)]
  hold=sim(selections(test),threshold)
  baseline=sim([{"date":r["date"],"game_id":r["game_id"],"market":market,
   "edge":r["pred_margin"]+r["spread_line"] if market=="spread" else r["pred_total"]-r["total_line"],
   "delta":actual(r)-market_value(r)} for r in test],5 if market=="spread" else 8)
  report["markets"][market]={"alpha":alpha,"threshold":threshold,"train_games":len(train),
   "validation_games":len(val),"test_games":len(test),
   "test":{k:v for k,v in hold.items() if k!="ledger"},
   "opponent_rest_fixed_threshold_test":{k:v for k,v in baseline.items() if k!="ledger"},
   "test_ledger":hold["ledger"]}
 return report

def main():
 p=argparse.ArgumentParser();p.add_argument("--games",required=True);p.add_argument("--out",required=True)
 a=p.parse_args();games=[json.loads(x) for x in Path(a.games).open() if x.strip()]
 if len(games)!=1230:raise RuntimeError("Expected verified 1230-game index")
 import os
 preds=project(games);odds=fetch([x["game_id"] for x in preds],os.environ["BALLDONTLIE_API_KEY"])
 result=run(games,odds);out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
 out.write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps({**result,"markets":{k:{kk:vv for kk,vv in v.items() if kk!="test_ledger"} for k,v in result["markets"].items()}},indent=2))
if __name__=="__main__":main()
