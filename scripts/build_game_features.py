"""Download each season once and generate team and rest features concurrently with player downloads."""
import argparse
import json
from pathlib import Path
from v13.history import fetch_season
from v13.schedule_features import schedule_features
from v13.team_features import team_features

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--season",type=int,required=True)
    p.add_argument("--out",default="research_data/game_features")
    args=p.parse_args()
    games=fetch_season(args.season)
    if len(games)<1200: raise RuntimeError("Incomplete regular season")
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    for name,records in (("schedule",schedule_features(games)),("team",team_features(games))):
        if len(records)!=len(games): raise RuntimeError(f"{name} count mismatch")
        with (root/f"{name}_{args.season}.jsonl").open("w") as f:
            for row in records: f.write(json.dumps(row,separators=(",",":"))+"\n")
        print(f"{args.season} {name} features: {len(records)}",flush=True)
if __name__=="__main__": main()
