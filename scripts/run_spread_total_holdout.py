"""Backtest opening spread and total lines with frozen training-selected edge thresholds.

This version retrieves odds from the already verified GOAT endpoint.
"""
import argparse,json,os
from pathlib import Path
from scripts.backtest_opening_spreads_totals import fetch,load
from scripts.holdout_spreads_totals import make,sim
def main():
 p=argparse.ArgumentParser();p.add_argument("--predictions",required=True);p.add_argument("--out",required=True)
 a=p.parse_args();pred=load(a.predictions)
 key=os.environ["BALLDONTLIE_API_KEY"]
 odds=fetch([x["game_id"] for x in pred],key)
 rows=make(pred,odds)
 dates=sorted({r["date"] for r in rows});cutoff=dates[int(len(dates)*.65)]
 report={"holdout_start":cutoff,"pricing":"assumed -110","market_lines":"historical opening","markets":{}}
 for market,thresholds in {"spread":[2,3,4,5,6,8],"total":[3,4,5,6,8,10]}.items():
  tr=[r for r in rows if r["market"]==market and r["date"]<cutoff]
  te=[r for r in rows if r["market"]==market and r["date"]>=cutoff]
  eligible=[(t,sim(tr,t)) for t in thresholds]
  eligible=[(t,s) for t,s in eligible if s["bets"]>=40]
  if not eligible:raise RuntimeError(f"No valid threshold for {market}")
  threshold=max(eligible,key=lambda x:(x[1]["return_pct"],-x[1]["bets"]))[0]
  hold=sim(te,threshold)
  report["markets"][market]={"threshold":threshold,"train_rows":len(tr),"holdout_rows":len(te),
   "training":{k:v for k,v in sim(tr,threshold).items() if k!="ledger"},"holdout":hold}
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps({k:{**v,"holdout":{a:b for a,b in v["holdout"].items() if a!="ledger"}} for k,v in report["markets"].items()},indent=2))
if __name__=="__main__":main()
