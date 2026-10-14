import os,json,requests,sys
from pathlib import Path
key=os.environ["BALLDONTLIE_API_KEY"]
games=[json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()]
games=[g for g in games if g.get("status")=="Final" and g.get("postseason") is False]
samples=[games[int(i*(len(games)-1)/11)] for i in range(12)]
out=[]
for g in samples:
 r=requests.get("https://api.balldontlie.io/nba/v2/odds/player_props/opening",params={"game_id":g["id"]},headers={"Authorization":key},timeout=60)
 r.raise_for_status()
 data=r.json()["data"]
 print(g["id"],len(data),flush=True)
 for row in data:
  if row.get("market",{}).get("type")=="over_under" and row.get("opened_at"):
   out.append(row)
p=Path("research_results/prop_openings");p.mkdir(parents=True,exist_ok=True)
(p/"sample.jsonl").write_text("".join(json.dumps(x)+"\n" for x in out))
print(json.dumps({"sampled_games":len(samples),"two_sided_lines":len(out)}))
if not out:raise RuntimeError("No historical two-sided prop lines")
