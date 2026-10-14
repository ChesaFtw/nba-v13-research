"""Collect opening prop thresholds across the 2025-26 regular season.

Use fixed sportsbook priority and retain timestamps. Betting prices assumed -110.
"""
import os,json,sys,time,requests
from pathlib import Path
key=os.environ["BALLDONTLIE_API_KEY"]
games=[json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()]
games=[g for g in games if g.get("status")=="Final" and g.get("postseason") is False]
games.sort(key=lambda g:(str(g.get("date","")),str(g.get("id",""))))
out=Path("research_results/prop_openings_full");out.mkdir(parents=True,exist_ok=True)
p=out/"opening_props_2025.jsonl";counts={"games":0,"rows":0,"usable":0}
session=requests.Session()
with p.open("w") as f:
 for g in games:
  gid=g["id"]
  for attempt in range(7):
   r=session.get("https://api.balldontlie.io/nba/v2/odds/player_props/opening",params={"game_id":gid},headers={"Authorization":key},timeout=50)
   if r.status_code in (429,500,502,503,504):
    if attempt==6:r.raise_for_status()
    time.sleep(min(5*(attempt+1),30));continue
   r.raise_for_status();break
  payload=r.json()
  if not isinstance(payload.get("data"),list):raise RuntimeError("Invalid response")
  counts["games"]+=1;counts["rows"]+=len(payload["data"])
  for row in payload["data"]:
   if str(row.get("game_id"))!=str(gid):raise RuntimeError("Game ID mismatch")
   if row.get("market",{}).get("type")!="over_under" or not row.get("opened_at") or row.get("line_value") is None:continue
   f.write(json.dumps(row,separators=(",",":"))+"\n");counts["usable"]+=1
  if counts["games"]%100==0:print(counts,flush=True)
  time.sleep(.06)
(out/"manifest.json").write_text(json.dumps(counts,indent=2)+"\n")
print(json.dumps(counts),flush=True)
if counts["games"]<1200 or counts["usable"]<10000:raise RuntimeError("Incomplete historical prop coverage")
