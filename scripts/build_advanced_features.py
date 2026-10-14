"""Build leakage-safe prior-game advanced stat signals; never use same-game advanced stats."""
import argparse,json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

FIELDS=("pace","usage_percentage","offensive_rating","defensive_rating","net_rating","true_shooting_percentage","effective_field_goal_percentage","possessions")
def get_id(x):
    if isinstance(x,dict):return x.get("id")
    return x
def join_advanced(player_rows,advanced_rows):
    games={}
    for row in player_rows:
        game=row.get("game") or {}
        date=str(game.get("date") or row.get("date") or "")[:10]
        pid=get_id(row.get("player"))
        gid=get_id(game)
        if not date or not pid or not gid:continue
        games[(gid,pid)]=(date,row)
    advanced=defaultdict(dict)
    for row in advanced_rows:
        if row.get("period",0)!=0:continue
        pid=get_id(row.get("player") or row.get("player_id"))
        gid=get_id(row.get("game") or row.get("game_id"))
        if (gid,pid) not in games:continue
        date,_=games[(gid,pid)]
        advanced[(gid,pid)]={k:row.get(k) for k in FIELDS}
    byplayer=defaultdict(list)
    for (gid,pid),(date,row) in games.items():
        byplayer[pid].append((date,gid,row))
    output=[]
    for pid,items in byplayer.items():
        items.sort(key=lambda v:(v[0],str(v[1])))
        history=defaultdict(list)
        for date in sorted(set(x[0] for x in items)):
            today=[x for x in items if x[0]==date]
            for _,gid,row in today:
                metrics={k:(sum(history[k][-10:])/len(history[k][-10:]) if history[k] else None) for k in FIELDS}
                output.append({"player_id":pid,"game_id":gid,"game_date":date,"advanced_prior_10":metrics,"advanced_games_seen":len(history["pace"])})
            for _,gid,_ in today:
                for k,v in advanced.get((gid,pid),{}).items():
                    if isinstance(v,(int,float)) and not isinstance(v,bool):history[k].append(v)
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument("--season",type=int,required=True);p.add_argument("--players",required=True);p.add_argument("--advanced",required=True)
    a=p.parse_args()
    players=[json.loads(x) for x in Path(a.players).open() if x.strip()]
    adv=[json.loads(x) for x in Path(a.advanced).open() if x.strip()]
    rows=join_advanced(players,adv)
    if len(rows)<10000:raise RuntimeError(f"Insufficient matched player games: {len(rows)}")
    nonnull=sum(any(v is not None for v in r["advanced_prior_10"].values()) for r in rows)
    if nonnull<1000:raise RuntimeError(f"Insufficient prior advanced coverage: {nonnull}")
    out=Path("research_data/advanced_features");out.mkdir(parents=True,exist_ok=True)
    with (out/f"player_advanced_prior_{a.season}.jsonl").open("w") as f:
        for r in rows:f.write(json.dumps(r,separators=(",",":"))+"\n")
    report={"season":a.season,"player_game_rows":len(rows),"with_prior_advanced":nonnull,"advanced_raw_rows":len(adv),"warning":"Feature-only output; no betting performance or validated confidence grades"}
    (out/"manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
