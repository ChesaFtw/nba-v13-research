"""Chronological challenger: last-10, Ridge, ElasticNet and equal-weight blend.

Projection MAE only: no betting lines or win-rate claims.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet
from v13.prop_features import MARKETS

def evaluate(rows, min_train_dates=60, test_dates=20):
    dates=sorted({r["game_date"] for r in rows})
    if len(dates)<=min_train_dates: raise ValueError("Insufficient dates")
    if len({(r["player_id"],r["game_date"]) for r in rows})!=len(rows):
        raise ValueError("Duplicate player/date")
    report={}
    for market in MARKETS:
        keys=[f"{market}_avg_{n}" for n in (5,10,20)]
        actual=f"{market}_actual"
        errors={k:[] for k in ("last10","ridge","elasticnet","equal_blend")}
        folds=[]
        for i in range(min_train_dates,len(dates),test_dates):
            cutoff=dates[i]
            end=dates[min(i+test_dates,len(dates))-1]
            train=[r for r in rows if r["game_date"]<cutoff]
            test=[r for r in rows if cutoff<=r["game_date"]<=end]
            if len(train)<100 or not test: continue
            X=np.array([[float(r[k]) for k in keys] for r in train])
            y=np.array([float(r[actual]) for r in train])
            Z=np.array([[float(r[k]) for k in keys] for r in test])
            target=np.array([float(r[actual]) for r in test])
            if not all(np.isfinite(a).all() for a in (X,y,Z,target)): raise ValueError("Nonfinite")
            mu=X.mean(axis=0);sd=X.std(axis=0);sd[sd<1e-8]=1
            X=(X-mu)/sd;Z=(Z-mu)/sd
            models={
                "ridge":Ridge(alpha=25.0),
                "elasticnet":ElasticNet(alpha=0.1,l1_ratio=0.5,max_iter=10000),
            }
            predictions={"last10":np.array([float(r[keys[1]]) for r in test])}
            for name,model in models.items():
                model.fit(X,y)
                predictions[name]=model.predict(Z)
            predictions["equal_blend"]=(predictions["ridge"]+predictions["elasticnet"])/2
            for name,pred in predictions.items(): errors[name].extend(np.abs(pred-target).tolist())
            folds.append({"train_before":cutoff,"test_through":end,"train_n":len(train),"test_n":len(test)})
        if not folds: raise ValueError("No valid folds")
        report[market]={"n":len(errors["ridge"]),"mae":{k:round(float(np.mean(v)),5) for k,v in errors.items()},"folds":folds}
    return {"kind":"Chronological projection MAE only; NOT betting hit rate","markets":report,
            "limitations":"Fixed hyperparameters, no nested tuning, no historical prop thresholds or calibrated probabilities"}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    rows=[json.loads(line) for line in Path(a.input).open() if line.strip()]
    result=evaluate(rows)
    dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v["mae"] for k,v in result["markets"].items()},indent=2))
if __name__=="__main__":main()
