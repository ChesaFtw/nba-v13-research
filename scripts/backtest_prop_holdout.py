"""Out-of-time NBA prop market backtest, assumed -110 for every wager.

Never optimize parameters on the final holdout. Source thresholds are mandatory.
"""
import argparse,json,math
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge

MARKETS={"points":"points","rebounds":"rebounds","assists":"assists","threes":"threes","pra":"pra"}
ALIASES={"points":"points","rebounds":"rebounds","assists":"assists","threes":"threes","threes_made":"threes","three_pointers_made":"threes","pts_rebs_asts":"pra","points_rebounds_assists":"pra"}
VENDORS=("caesars","draftkings","fanduel","betmgm","betrivers","fanatics")
FEATURES=lambda market:[f"{market}_avg_{n}" for n in (5,10,20)]
def read(path):
 return [json.loads(x) for x in Path(path).open() if x.strip()]
def normalize_lines(rows):
 selected={}
 for r in rows:
  if (r.get("market") or {}).get("type")!="over_under":continue
  market=ALIASES.get(r.get("prop_type"))
  if market not in MARKETS:continue
  if not r.get("opened_at") or r.get("line_value") is None:continue
  try:line=float(r["line_value"])
  except (ValueError,TypeError):continue
  if not math.isfinite(line):continue
  key=(str(r["game_id"]),str(r["player_id"]),market)
  vendor=str(r.get("vendor") or "").lower()
  rank=(VENDORS.index(vendor) if vendor in VENDORS else 99,str(r["opened_at"]),str(r.get("id","")))
  if key not in selected or rank<selected[key][0]:selected[key]=(rank,r,line)
 return {k:(v[1],v[2]) for k,v in selected.items()}
def dataset(features,odds):
 lines=normalize_lines(odds);out=[]
 for f in features:
  for market in MARKETS:
   found=lines.get((str(f["game_id"]),str(f["player_id"]),market))
   if found is None:continue
   raw,line=found
   try:
    x=[float(f[k]) for k in FEATURES(market)]
    actual=float(f[f"{market}_actual"])
   except (ValueError,KeyError,TypeError):continue
   if not all(math.isfinite(v) for v in x+[actual]):continue
   out.append({"date":f["game_date"],"game_id":f["game_id"],"player_id":f["player_id"],
    "market":market,"line":line,"actual":actual,"x":x,"vendor":raw.get("vendor"),"opened_at":raw["opened_at"]})
 return out
def simulate(rows,cutoff,threshold,bankroll=10000):
 ledger=[];peak=bankroll
 for r in sorted(rows,key=lambda r:(r["date"],str(r["game_id"]),str(r["player_id"]))):
  diff=r["projection"]-r["line"]
  if abs(diff)<threshold:continue
  delta=r["actual"]-r["line"]
  outcome="push" if abs(delta)<1e-9 else ("win" if (delta>0)==(diff>0) else "loss")
  stake=round(bankroll*.01,2)
  pnl=round(stake*(100/110 if outcome=="win" else -1 if outcome=="loss" else 0),2)
  bankroll=round(bankroll+pnl,2);peak=max(peak,bankroll)
  ledger.append({"date":r["date"],"game_id":r["game_id"],"player_id":r["player_id"],
   "market":r["market"],"vendor":r["vendor"],"line":r["line"],"projection":round(r["projection"],3),
   "side":"over" if diff>0 else "under","outcome":outcome,"stake":stake,"pnl":pnl,
   "bankroll":bankroll,"drawdown_pct":round(100*(peak-bankroll)/peak,3)})
 c=Counter(r["outcome"] for r in ledger);decided=c["win"]+c["loss"]
 return {"bets":len(ledger),"wins":c["win"],"losses":c["loss"],"pushes":c["push"],
  "hit_rate":round(c["win"]/decided,4) if decided else None,
  "ending_bankroll":bankroll,"roi_on_start_pct":round(100*(bankroll/10000-1),2),
  "max_drawdown_pct":max((r["drawdown_pct"] for r in ledger),default=0),"ledger":ledger}
def evaluate(features,odds):
 rows=dataset(features,odds)
 if len(rows)<1000:raise RuntimeError(f"Insufficient matched historical prop rows: {len(rows)}")
 dates=sorted({r["date"] for r in rows});cutoff=dates[int(len(dates)*.65)]
 report={"source":"GOAT historical opening player props","price":"fixed assumed -110","holdout_start":cutoff,
  "matched_market_rows":len(rows),"markets":{}}
 for market in MARKETS:
  subset=[r for r in rows if r["market"]==market]
  train=[r for r in subset if r["date"]<cutoff];test=[r for r in subset if r["date"]>=cutoff]
  if len(train)<100 or len(test)<50:
   report["markets"][market]={"status":"insufficient coverage","train":len(train),"test":len(test)};continue
  X=np.array([r["x"] for r in train]);y=np.array([r["actual"] for r in train])
  mu=X.mean(axis=0);sd=X.std(axis=0);sd[sd<1e-8]=1
  model=Ridge(alpha=25.).fit((X-mu)/sd,y)
  for r in subset:
   r["projection"]=float(model.predict((np.array(r["x"])[None,:]-mu)/sd)[0])
  # Pre-registered thresholds; choose using TRAIN only, with >=60 training bets.
  thresholds={"points":[2,3,4,5],"rebounds":[1,1.5,2,2.5],"assists":[1,1.5,2],"threes":[.5,1,1.5],"pra":[3,4,5,6]}[market]
  scored=[(t,simulate(train,cutoff,t)) for t in thresholds]
  eligible=[(t,s) for t,s in scored if s["bets"]>=60]
  if not eligible:
   report["markets"][market]={"status":"no qualifying training threshold","train":len(train),"test":len(test)};continue
  # Conservative model selection: maximum training ROI, tie prefers fewer wagers.
  best=max(eligible,key=lambda item:(item[1]["roi_on_start_pct"],-item[1]["bets"]))
  chosen=best[0];holdout=simulate(test,cutoff,chosen)
  # Compare an untrained 10-game projection with identical threshold and holdout.
  ridge_proj=[r["projection"] for r in test]
  for r in test:r["projection"]=r["x"][1]
  baseline=simulate(test,cutoff,chosen)
  for r,p in zip(test,ridge_proj):r["projection"]=p
  report["markets"][market]={"status":"evaluated","train_rows":len(train),"holdout_rows":len(test),
   "chosen_edge_threshold":chosen,"training":{k:v for k,v in best[1].items() if k!="ledger"},
   "ridge_holdout":{k:v for k,v in holdout.items() if k!="ledger"},
   "last10_holdout":{k:v for k,v in baseline.items() if k!="ledger"},
   "ridge_holdout_ledger":holdout["ledger"]}
 return report
def main():
 p=argparse.ArgumentParser();p.add_argument("--features",required=True);p.add_argument("--lines",required=True);p.add_argument("--out",required=True)
 a=p.parse_args();result=evaluate(read(a.features),read(a.lines))
 path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps({"matched":result["matched_market_rows"],"holdout_start":result["holdout_start"],
  "markets":{k:{kk:vv for kk,vv in v.items() if kk!="ridge_holdout_ledger"} for k,v in result["markets"].items()}},indent=2))
if __name__=="__main__":main()
