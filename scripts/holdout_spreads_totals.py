"""Evaluate spread and total challenger against opening lines on chronological holdout.

Pre-register model candidates and edge thresholds; select on early games only.
"""
import argparse,json,math
from pathlib import Path
from collections import Counter
from scripts.backtest_opening_spreads_totals import choose
from v13.pricing import ASSUMED_110,game_price,profit_per_unit
def load(p):return [json.loads(x) for x in Path(p).open() if x.strip()]
def make(predictions,odds,tips=None,pricing="assumed110",skipped=None):
 """pricing='actual' grades each side at its own sourced opening price; rows lacking it are skipped."""
 selected=choose(odds,tips);rows=[]
 for p in predictions:
  r=selected.get(str(p["game_id"]))
  if r is None:continue
  for market,field,pred,actual in (("spread","spread_home_value",p["predicted_home_margin"],p["actual_home_margin"]),
                                    ("total","total_value",p["predicted_total"],p["actual_total"])):
   try:
    line=float(r[field]);pred=float(pred);actual=float(actual)
   except (TypeError,ValueError,KeyError):continue
   if not all(math.isfinite(x) for x in (line,pred,actual)):continue
   edge=pred+line if market=="spread" else pred-line
   delta=actual+line if market=="spread" else actual-line
   side=("home" if edge>0 else "away") if market=="spread" else ("over" if edge>0 else "under")
   price=ASSUMED_110 if pricing=="assumed110" else game_price(r,market,side)
   if price is None:
    if skipped is not None:skipped[f"{market}_missing_price"]=skipped.get(f"{market}_missing_price",0)+1
    continue
   rows.append({"date":p["date"],"game_id":p["game_id"],"market":market,
    "edge":edge,"delta":delta,"line":line,"vendor":r.get("vendor"),"price":price,"payout":profit_per_unit(price)})
 return rows
def sim(rows,threshold,day_staking=False):
 """day_staking=True sizes every bet from the bankroll at the start of its date."""
 bank=10000.;peak=bank;out=[];day=None;day_start=bank
 for r in sorted(rows,key=lambda x:(x["date"],str(x["game_id"]))):
  if r["date"]!=day:day=r["date"];day_start=bank
  if abs(r["edge"])<threshold:continue
  win=(r["edge"]>0)==(r["delta"]>0)
  outcome="push" if abs(r["delta"])<1e-9 else ("win" if win else "loss")
  stake=round((day_start if day_staking else bank)*.01,2);pnl=round(stake*(r.get("payout",100/110) if outcome=="win" else -1 if outcome=="loss" else 0),2)
  bank=round(bank+pnl,2);peak=max(peak,bank)
  out.append({"date":r["date"],"game_id":r["game_id"],"market":r["market"],
   "outcome":outcome,"edge":round(r["edge"],3),"stake":stake,"pnl":pnl})
 counts=Counter(x["outcome"] for x in out);n=counts["win"]+counts["loss"]
 return {"bets":len(out),"wins":counts["win"],"losses":counts["loss"],"pushes":counts["push"],
  "hit_rate":round(counts["win"]/n,4) if n else None,"ending_bankroll":bank,
  "return_pct":round(100*(bank/10000-1),2),"ledger":out}
def main():
 p=argparse.ArgumentParser();p.add_argument("--predictions",required=True);p.add_argument("--odds",required=True);p.add_argument("--out",required=True)
 a=p.parse_args();rows=make(load(a.predictions),load(a.odds))
 dates=sorted({r["date"] for r in rows});cutoff=dates[int(len(dates)*.65)]
 report={"holdout_start":cutoff,"pricing":"assumed -110","markets":{}}
 for market,thresholds in {"spread":[2,3,4,5,6,8],"total":[3,4,5,6,8,10]}.items():
  tr=[r for r in rows if r["market"]==market and r["date"]<cutoff]
  te=[r for r in rows if r["market"]==market and r["date"]>=cutoff]
  scored=[(t,sim(tr,t)) for t in thresholds]
  eligible=[(t,s) for t,s in scored if s["bets"]>=40]
  if not eligible:
   report["markets"][market]={"status":"insufficient training bets","train_rows":len(tr),"test_rows":len(te)};continue
  chosen=max(eligible,key=lambda x:(x[1]["return_pct"],-x[1]["bets"]))[0]
  hold=sim(te,chosen)
  report["markets"][market]={"threshold":chosen,"training":{k:v for k,v in sim(tr,chosen).items() if k!="ledger"},
   "holdout":hold}
 path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!="holdout"}|({"holdout":{kk:vv for kk,vv in v["holdout"].items() if kk!="ledger"}} if "holdout" in v else {}) for k,v in report["markets"].items()},indent=2))
if __name__=="__main__":main()
