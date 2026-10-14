"""Train regularized linear player prop projections with a chronological holdout.

Training is season-local, uses only pregame rolling stats and requires numpy.
This is a projection error benchmark, not a sportsbook wager backtest.
"""
import argparse,json
from pathlib import Path
import numpy as np
from v13.prop_features import MARKETS

def evaluate(rows,season,alpha=25.):
    dates=sorted(set(r["game_date"] for r in rows))
    if len(dates)<60:raise RuntimeError("Insufficient dates")
    cutoff=dates[int(len(dates)*.8)]
    training=[r for r in rows if r["game_date"]<cutoff]
    testing=[r for r in rows if r["game_date"]>=cutoff]
    if len(training)<1000 or len(testing)<100:raise RuntimeError("Insufficient holdout")
    report={"season":season,"cutoff_date":cutoff,"train_n":len(training),"holdout_n":len(testing),"markets":{},"notes":["All input rolling features exclude target game","No betting odds or lines; MAE is not betting hit rate","No cross-season feature continuity; compare with rolling baseline on identical split"]}
    for market in MARKETS:
        names=[f"{market}_avg_{n}" for n in (5,10,20)]
        X=np.array([[r[k] for k in names] for r in training],dtype=float)
        y=np.array([r[f"{market}_actual"] for r in training],dtype=float)
        Z=np.array([[r[k] for k in names] for r in testing],dtype=float)
        actual=np.array([r[f"{market}_actual"] for r in testing],dtype=float)
        mu=X.mean(axis=0);sd=X.std(axis=0);sd[sd<1e-8]=1.
        A=(X-mu)/sd;B=(Z-mu)/sd
        ymean=y.mean();coef=np.linalg.solve(A.T@A+alpha*np.eye(3),A.T@(y-ymean))
        prediction=ymean+B@coef
        baseline=Z[:,1]
        report["markets"][market]={"ridge_mae":round(float(np.abs(prediction-actual).mean()),4),"last10_mae":round(float(np.abs(baseline-actual).mean()),4),"ridge_vs_last10_mae_gain":round(float(np.abs(baseline-actual).mean()-np.abs(prediction-actual).mean()),4),"ridge_weights_standardized":[round(float(x),6) for x in coef]}
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument("--season",type=int,required=True);p.add_argument("--input",required=True);p.add_argument("--out",default="research_results/trained_prop_models");a=p.parse_args()
    rows=[json.loads(s) for s in Path(a.input).open() if s.strip()]
    if len(rows)<10000:raise RuntimeError("Insufficient player history")
    report=evaluate(rows,a.season)
    dest=Path(a.out);dest.mkdir(parents=True,exist_ok=True)
    (dest/f"trained_{a.season}.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
