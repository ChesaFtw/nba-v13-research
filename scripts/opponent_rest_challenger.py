"""Pregame schedule-adjusted opponent-strength challenger.

Strictly earlier dates; no current-game results in features. Fixed coefficients,
no holdout fitting. Score projections are evaluated separately from market bets.
"""
import argparse,json,statistics
from pathlib import Path
from collections import defaultdict
from datetime import date

def team_id(x):return x.get("id") if isinstance(x,dict) else x

def project(games,min_games=5):
    history=defaultdict(list);output=[]
    ordered=sorted(games,key=lambda g:(str(g.get("date","")),str(g.get("id",""))))
    days=sorted(set(str(g.get("date",""))[:10] for g in ordered))
    for day in days:
        today=[g for g in ordered if str(g.get("date",""))[:10]==day]
        for g in today:
            if g.get("postseason") is not False or g.get("status")!="Final":continue
            h=team_id(g.get("home_team"));a=team_id(g.get("visitor_team"))
            if h is None or a is None or h==a:continue
            hp=history[h];ap=history[a]
            if len(hp)<min_games or len(ap)<min_games:continue
            def recent(team):
                hist=history[team]
                short=hist[-10:];long=hist[-25:]
                def avg(items,index):return sum(x[index] for x in items)/len(items)
                return tuple(.65*avg(short,k)+.35*avg(long,k) for k in (1,2))
            hf,ha=recent(h);af,aa=recent(a)
            # Blend offense and opponent defense, with partial regression to
            # fixed league baseline to reduce early-season scoring volatility.
            home_score=.48*hf+.48*aa+.04*115+1.5
            away_score=.48*af+.48*ha+.04*115-1.5
            def rest(team):
                last=history[team][-1][0]
                return max(0,(date.fromisoformat(day)-date.fromisoformat(last)).days-1)
            rh=rest(h);ra=rest(a)
            # Fixed small differential rest adjustment; not tuned on holdout.
            rest_delta=max(-2,min(2,rh-ra))*.7
            home_score+=rest_delta/2
            away_score-=rest_delta/2
            hs=g.get("home_team_score");vs=g.get("visitor_team_score")
            output.append({"game_id":g["id"],"date":day,
                "predicted_home_margin":round(home_score-away_score,4),
                "predicted_total":round(home_score+away_score,4),
                "actual_home_margin":hs-vs,"actual_total":hs+vs,
                "model_version":"opponent_blend_rest_v1",
                "home_rest":rh,"away_rest":ra})
        # Update after all predictions on this date: prevents same-day leakage.
        for g in today:
            if g.get("postseason") is not False or g.get("status")!="Final":continue
            h=team_id(g.get("home_team"));a=team_id(g.get("visitor_team"))
            hs=g.get("home_team_score");vs=g.get("visitor_team_score")
            if h is None or a is None or not isinstance(hs,(int,float)) or not isinstance(vs,(int,float)):continue
            history[h].append((day,float(hs),float(vs)))
            history[a].append((day,float(vs),float(hs)))
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument("--games",required=True);p.add_argument("--out",required=True)
    a=p.parse_args();games=[json.loads(x) for x in Path(a.games).read_text().splitlines() if x.strip()]
    if len(games)!=1230:raise ValueError("Expected 1230 verified regular games")
    rows=project(games)
    if len(rows)<900:raise ValueError("Incomplete challenger predictions")
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    print(json.dumps({"model":"opponent_blend_rest_v1","predictions":len(rows),
        "margin_mae":round(statistics.mean(abs(r["predicted_home_margin"]-r["actual_home_margin"]) for r in rows),4),
        "total_mae":round(statistics.mean(abs(r["predicted_total"]-r["actual_total"]) for r in rows),4),
        "note":"fixed-coefficient challenger, no current-game outcome leakage"}))
if __name__=="__main__":main()
