"""Collect verified complete older season game index, player box scores and opening threes lines."""
import argparse,json,os,requests,time
from pathlib import Path
from scripts.backfill_games import fetch_games
from scripts.backfill_player_stats import main as unused
p=argparse.ArgumentParser();p.add_argument("--season",type=int,required=True);a=p.parse_args()
season=a.season;key=os.environ["BALLDONTLIE_API_KEY"];s=requests.Session()
games=[g for g in fetch_games(season,session=s) if g.get("postseason") is False]
if len(games)!=1230:raise RuntimeError(f"Expected 1230 games, got {len(games)}")
root=Path(f"research_data/multiseason_threes/{season}");root.mkdir(parents=True,exist_ok=True)
(root/"games.jsonl").write_text("".join(json.dumps(g)+"\n" for g in games))
finished=sorted((g for g in games if g.get("status")=="Final"),key=lambda g:(g["date"],g["id"]))
n=0;usable=0
with (root/"opening_threes.jsonl").open("w") as f:
 for g in finished:
  for attempt in range(7):
   r=s.get("https://api.balldontlie.io/nba/v2/odds/player_props/opening",params={"game_id":g["id"]},headers={"Authorization":key},timeout=60)
   if r.status_code in (429,500,502,503,504):
    time.sleep(5*(attempt+1));continue
   r.raise_for_status();break
  for x in r.json().get("data",[]):
   if str(x.get("prop_type","")).lower() not in ("threes","threes_made","three_pointers_made"):continue
   if (x.get("market") or {}).get("type")!="over_under" or not x.get("opened_at") or x.get("line_value") is None:continue
   f.write(json.dumps(x)+"\n");usable+=1
  n+=1
  if n%100==0:print(json.dumps({"games":n,"three_prop_rows":usable}),flush=True)
  time.sleep(.06)
(root/"manifest.json").write_text(json.dumps({"season":season,"games":n,"three_prop_rows":usable},indent=2)+"\n")
if usable<5000:raise RuntimeError(f"Insufficient season coverage: {usable}")
