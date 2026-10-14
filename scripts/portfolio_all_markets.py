"""Portfolio audit across independently generated NBA prop, spread, and total selections.

All input ledgers must represent sourced opening lines and -110 assumptions.
This is a portfolio simulator, not evidence of an optimized model.
"""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
def load(path):return json.loads(Path(path).read_text())
def merge(prop,game):
 picks=[]
 for market,details in prop["markets"].items():
  if details.get("status")!="evaluated":continue
  for x in details["ridge_holdout_ledger"]:
   picks.append({"date":x["date"],"game_id":str(x["game_id"]),"market":"prop_"+market,
    "player_id":str(x["player_id"]),"side":x["side"],"line":x["line"],"outcome":x["outcome"]})
 for x in game["ledger"]:
  if x["date"]<prop["holdout_start"]:continue
  picks.append({"date":x["date"],"game_id":str(x["game_id"]),"market":x["market"],
   "player_id":None,"side":x["pick"],"line":x["line"],"outcome":x["outcome"]})
 return picks
def simulate(picks,limit_per_game=2,max_daily_exposure=.06,stake_fraction=.01):
 bankroll=10000.;peak=bankroll;ledger=[];bydate=defaultdict(list)
 for p in picks:bydate[p["date"]].append(p)
 for date in sorted(bydate):
  game_counts=Counter();exposure=0.
  # Stable predetermined order, not hindsight win/loss or historical ROI ranking.
  priority={"prop_points":0,"prop_rebounds":1,"prop_assists":2,"prop_threes":3,"prop_pra":4,"spread":5,"total":6}
  for p in sorted(bydate[date],key=lambda x:(priority.get(x["market"],99),x["game_id"],str(x["player_id"]))):
   if game_counts[p["game_id"]]>=limit_per_game or exposure+stake_fraction>max_daily_exposure+1e-8:continue
   stake=round(bankroll*stake_fraction,2)
   outcome=p["outcome"]
   pnl=round(stake*(100/110 if outcome=="win" else -1 if outcome=="loss" else 0),2)
   bankroll=round(bankroll+pnl,2);peak=max(peak,bankroll)
   game_counts[p["game_id"]]+=1;exposure+=stake_fraction
   ledger.append({**p,"stake":stake,"pnl":pnl,"bankroll":bankroll,"drawdown_pct":round(100*(peak-bankroll)/peak,3)})
 c=Counter(x["outcome"] for x in ledger);n=c["win"]+c["loss"]
 bymarket={}
 for market in sorted({x["market"] for x in ledger}):
  subset=[x for x in ledger if x["market"]==market];cc=Counter(x["outcome"] for x in subset)
  bymarket[market]={"bets":len(subset),"wins":cc["win"],"losses":cc["loss"],
   "hit_rate":round(cc["win"]/(cc["win"]+cc["loss"]),4) if cc["win"]+cc["loss"] else None}
 return {"starting_bankroll":10000,"ending_bankroll":bankroll,"return_pct":round((bankroll/10000-1)*100,2),
  "bets":len(ledger),"wins":c["win"],"losses":c["loss"],"hit_rate":round(c["win"]/n,4) if n else None,
  "max_drawdown_pct":max((x["drawdown_pct"] for x in ledger),default=0),
  "per_game_limit":limit_per_game,"daily_exposure_limit":max_daily_exposure,
  "per_bet_stake_fraction":stake_fraction,"markets":bymarket,"ledger":ledger}
def main():
 p=argparse.ArgumentParser();p.add_argument("--props",required=True);p.add_argument("--games",required=True);p.add_argument("--out",required=True)
 a=p.parse_args();props=load(a.props);games=load(a.games)
 if not props.get("holdout_start") or not games.get("ledger"):raise RuntimeError("Missing verified ledgers")
 result=simulate(merge(props,games));dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
 dest.write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps({k:v for k,v in result.items() if k!="ledger"},indent=2))
if __name__=="__main__":main()
