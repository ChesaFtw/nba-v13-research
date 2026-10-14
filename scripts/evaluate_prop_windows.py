"""Evaluate 5 NBA prop markets on held-out chronological records per season/window."""
import argparse,json
from pathlib import Path
from statistics import mean

def main():
    p=argparse.ArgumentParser();p.add_argument("--season",type=int,required=True);p.add_argument("--window",type=int,choices=[5,10,20],required=True)
    a=p.parse_args()
    src=Path(f"research_data/prop_research/props_{a.season}.jsonl")
    rows=[json.loads(line) for line in src.open() if line.strip()]
    if len(rows)<10000: raise RuntimeError("Insufficient prop records")
    dates=sorted(set(r["game_date"] for r in rows))
    cutoff=dates[int(len(dates)*0.8)]
    holdout=[r for r in rows if r["game_date"]>=cutoff]
    report={"season":a.season,"window":a.window,"split_date":cutoff,"holdout_rows":len(holdout),"markets":{}}
    for market in ("points","rebounds","assists","threes","pra"):
        key=f"{market}_avg_{a.window}"
        actual=f"{market}_actual"
        valid=[r for r in holdout if r.get(key) is not None and r.get(actual) is not None]
        report["markets"][market]={"n":len(valid),"mae":round(mean(abs(r[key]-r[actual]) for r in valid),4) if valid else None}
    out=Path("research_data/prop_evaluation");out.mkdir(parents=True,exist_ok=True)
    (out/f"evaluation_{a.season}_{a.window}.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
