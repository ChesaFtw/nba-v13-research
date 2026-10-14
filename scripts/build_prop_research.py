"""Build chronological player prop feature rows and simple MAE baselines."""
import argparse,json
from pathlib import Path
from v13.prop_features import build_prop_features,MARKETS

def main():
    p=argparse.ArgumentParser();p.add_argument("--season",type=int,required=True);p.add_argument("--input",required=True);p.add_argument("--out",default="research_data/prop_research")
    a=p.parse_args();src=Path(a.input)
    if not src.exists(): raise FileNotFoundError(src)
    rows=[json.loads(line) for line in src.open() if line.strip()]
    if len(rows)<10000: raise RuntimeError("Player-stat season appears incomplete")
    features=list(build_prop_features(rows))
    if len(features)<10000: raise RuntimeError("Too few pregame player feature rows")
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    with (out/f"props_{a.season}.jsonl").open("w") as f:
        for row in features: f.write(json.dumps(row,separators=(",",":"))+"\n")
    report={"season":a.season,"source_rows":len(rows),"feature_rows":len(features),"markets":{}}
    for market in MARKETS:
        for window in (5,10,20):
            key=f"{market}_avg_{window}"
            err=sum(abs(r[key]-r[f"{market}_actual"]) for r in features)/len(features)
            report["markets"][f"{market}_mae_{window}"]=round(err,4)
    (out/f"report_{a.season}.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
