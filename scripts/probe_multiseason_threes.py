"""Probe genuine older-season NBA opening three-pointer prop coverage before backtesting."""
import json,os,requests,time
from scripts.backfill_games import fetch_games
s=requests.Session();key=os.environ["BALLDONTLIE_API_KEY"]
report={}
for season in (2022,2023,2024):
 games=[g for g in fetch_games(season,session=s) if g.get("postseason") is False and g.get("status")=="Final"]
 games.sort(key=lambda g:(g.get("date",""),g["id"]))
 if len(games)<1100:raise RuntimeError(f"Only {len(games)} final games in {season}")
 picks=[games[int((i+.5)*len(games)/8)] for i in range(8)]
 samples=[]
 for g in picks:
  for attempt in range(6):
   r=s.get("https://api.balldontlie.io/nba/v2/odds/player_props/opening",params={"game_id":g["id"]},headers={"Authorization":key},timeout=60)
   if r.status_code in (429,500,502,503,504):
    time.sleep(5*(attempt+1));continue
   r.raise_for_status();break
  data=r.json().get("data",[])
  if not isinstance(data,list):raise RuntimeError("Unexpected API response")
  threes=[x for x in data if "three" in str(x.get("prop_type","")).lower() and (x.get("market") or {}).get("type")=="over_under"]
  samples.append({"game_id":g["id"],"date":g["date"],"total_opening_rows":len(data),"threes_rows":len(threes),"prop_types":sorted({str(x.get("prop_type")) for x in data})[:25]})
  time.sleep(.15)
 report[str(season)]={"final_regular_games":len(games),"sampled_games":len(samples),"sampled_three_rows":sum(x["threes_rows"] for x in samples),"samples":samples}
 print(json.dumps({season:{"sampled_three_rows":report[str(season)]["sampled_three_rows"],"games_with_threes":sum(x["threes_rows"]>0 for x in samples)}}),flush=True)
from pathlib import Path
out=Path("research_results/prop_multiseason_probe");out.mkdir(parents=True,exist_ok=True)
(out/"coverage.json").write_text(json.dumps(report,indent=2)+"\n")
