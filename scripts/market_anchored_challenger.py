"""Chronological market-anchored challenger; training-only calibration, untouched final quarter.

Uses historical opening lines at fixed -110. Never chooses parameters using holdout.
"""
import argparse,json,os
from pathlib import Path
from collections import Counter
from scripts.backtest_opening_spreads_totals import fetch,choose,load

def evaluate(predictions,odds,holdout_fraction=.25):
    lookup=choose(odds);records=[]
    for p in sorted(predictions,key=lambda x:(x["date"],str(x["game_id"]))):
        r=lookup.get(str(p["game_id"]))
        if not r:continue
        for market,linefield,predfield,actualfield in (
            ("spread","spread_home_value","predicted_home_margin","actual_home_margin"),
            ("total","total_value","predicted_total","actual_total")):
            raw=r.get(linefield)
            if raw is None:continue
            try:
                line=float(raw);pred=float(p[predfield]);actual=float(p[actualfield])
            except (ValueError,TypeError):continue
            anchor=-line if market=="spread" else line
            records.append({"date":p["date"],"game_id":p["game_id"],"market":market,"line":line,"anchor":anchor,
                "blind":pred,"actual":actual,"vendor":r.get("vendor")})
    report={"pricing":"assumed -110","selection_rule":"train-only alpha selection; minimum 3-point spread and 5-point total edge","markets":{},"coverage":len(records)}
    for market in ("spread","total"):
        data=sorted((r for r in records if r["market"]==market),key=lambda r:(r["date"],str(r["game_id"])))
        if len(data)<500:raise ValueError(f"Insufficient {market} lines: {len(data)}")
        # Date boundary ensures all test games are later than all train games.
        dates=sorted(set(r["date"] for r in data));boundary=dates[int(len(dates)*(1-holdout_fraction))]
        train=[r for r in data if r["date"]<boundary];test=[r for r in data if r["date"]>=boundary]
        # Fit shrinkage to historical residuals; fixed grid avoids post-hoc holdout optimization.
        alphas=(0,.1,.2,.3,.4,.5,.75,1.)
        def mse(rows,alpha):
            return sum((r["actual"]-(r["anchor"]+alpha*(r["blind"]-r["anchor"])))**2 for r in rows)/len(rows)
        best=min(alphas,key=lambda alpha:(mse(train,alpha),alpha))
        threshold=3. if market=="spread" else 5.
        outcomes={}
        for label,alpha in (("blind",1.),("market_anchor",0.),("trained_shrinkage",best)):
            wins=losses=pushes=0;bankroll=10000.;maxbankroll=bankroll;maxdd=0
            for r in test:
                projected=r["anchor"]+alpha*(r["blind"]-r["anchor"])
                delta=projected-r["anchor"]
                if abs(delta)<threshold:continue
                actualdelta=r["actual"]-r["anchor"]
                if abs(actualdelta)<1e-8:pushes+=1;continue
                win=(delta>0)==(actualdelta>0)
                stake=round(bankroll*.01,2)
                bankroll=round(bankroll+round(stake*(100/110 if win else -1),2),2)
                maxbankroll=max(maxbankroll,bankroll)
                maxdd=max(maxdd,100*(maxbankroll-bankroll)/maxbankroll)
                wins+=int(win);losses+=int(not win)
            outcomes[label]={"alpha":alpha,"wins":wins,"losses":losses,"pushes":pushes,
                "hit_rate":round(wins/(wins+losses),4) if wins+losses else None,
                "end_bankroll":bankroll,"max_drawdown_pct":round(maxdd,2),
                "holdout_mse":round(mse(test,alpha),3)}
        report["markets"][market]={"train_games":len(train),"holdout_games":len(test),"holdout_start":boundary,
            "chosen_alpha":best,"train_mse":round(mse(train,best),3),"variants":outcomes}
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--predictions",required=True);ap.add_argument("--out",required=True)
    a=ap.parse_args();key=os.getenv("BALLDONTLIE_API_KEY")
    if not key:raise RuntimeError("Missing BALLDONTLIE_API_KEY")
    predictions=load(a.predictions)
    odds=fetch([r["game_id"] for r in predictions],key)
    result=evaluate(predictions,odds)
    dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
