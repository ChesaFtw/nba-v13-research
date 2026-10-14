"""Walk-forward NBA prop point-projection benchmark with a frozen validation split.

This is a player-stat MAE benchmark, NOT a sportsbook betting backtest.
Each feature row contains strictly pregame rolling averages.
"""
import argparse,json
from pathlib import Path
from statistics import mean
from v13.prop_features import MARKETS

CANDIDATES={
 "last5":(1.,0.,0.),
 "last10":(0.,1.,0.),
 "last20":(0.,0.,1.),
 "short_blend":(.6,.3,.1),
 "balanced_blend":(.2,.5,.3),
 "long_blend":(.1,.3,.6),
}
def predict(row,market,weights):
    return sum(w*row[f"{market}_avg_{n}"] for n,w in zip((5,10,20),weights))
def evaluate(rows,season):
    if not rows:raise ValueError("Empty feature data")
    dates=sorted(set(r["game_date"] for r in rows))
    if len(dates)<60:raise ValueError("Not enough unique game dates")
    split=dates[int(len(dates)*.8)]
    train=[r for r in rows if r["game_date"]<split]
    test=[r for r in rows if r["game_date"]>=split]
    if len(train)<1000 or len(test)<100:raise ValueError("Insufficient train/test samples")
    report={"season":season,"split_date":split,"training_rows":len(train),"holdout_rows":len(test),"models":{},"limitations":["Rolling features exclude current game; within-season history only","Candidate chosen on earlier 80 percent of dates, evaluated on later 20 percent","This is point-projection MAE, not a betting hit rate or calibrated probability","Only rolling-average candidates; no odds, opponent, injury, or lineup information"]}
    for market in MARKETS:
        actual=f"{market}_actual"
        errors={name:mean(abs(predict(r,market,w)-r[actual]) for r in train) for name,w in CANDIDATES.items()}
        chosen=min(errors,key=errors.get)
        test_mae={name:round(mean(abs(predict(r,market,w)-r[actual]) for r in test),4) for name,w in CANDIDATES.items()}
        report["models"][market]={"selected_on_train":chosen,"train_mae":round(errors[chosen],4),"holdout_selected_mae":test_mae[chosen],"holdout_candidate_mae_diagnostic_only":test_mae,"holdout_n":len(test)}
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument("--season",type=int,required=True);p.add_argument("--input",required=True);p.add_argument("--out",default="research_results/prop_model_benchmarks")
    a=p.parse_args();rows=[json.loads(x) for x in Path(a.input).open() if x.strip()]
    if len(rows)<10000:raise RuntimeError(f"Insufficient rows: {len(rows)}")
    report=evaluate(rows,a.season)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/f"benchmark_{a.season}.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
