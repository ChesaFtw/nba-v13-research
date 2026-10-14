"""Produce leakage-safe NBA home-margin and game-total baseline predictions.

Uses only earlier finished games for rolling team points-for/against estimates.
Predictions are NOT market picks until matched to sourced historical thresholds.
"""
import argparse,json
from collections import defaultdict
from pathlib import Path

def predict(games,min_games=5):
    history=defaultdict(list);out=[]
    ordered=sorted(games,key=lambda x:(str(x.get("date","")),str(x.get("id",""))))
    for g in ordered:
        if g.get("postseason") is not False or g.get("status")!="Final":continue
        h=g.get("home_team") or {};a=g.get("visitor_team") or {}
        hid=h.get("id") if isinstance(h,dict) else h
        aid=a.get("id") if isinstance(a,dict) else a
        hs=g.get("home_team_score");vs=g.get("visitor_team_score")
        if None in (hid,aid,hs,vs) or hid==aid:continue
        hprior=[x for x in history[hid] if x[0]<str(g["date"])[:10]]
        aprior=[x for x in history[aid] if x[0]<str(g["date"])[:10]]
        if len(hprior)>=min_games and len(aprior)>=min_games:
            hp=hprior[-10:];ap=aprior[-10:]
            hfor=sum(x[1] for x in hp)/len(hp);hagainst=sum(x[2] for x in hp)/len(hp)
            afor=sum(x[1] for x in ap)/len(ap);aagainst=sum(x[2] for x in ap)/len(ap)
            predhome=(hfor+aagainst)/2+1.5
            predaway=(afor+hagainst)/2-1.5
            out.append({"game_id":g["id"],"date":str(g["date"])[:10],"predicted_home_margin":round(predhome-predaway,4),
                "predicted_total":round(predhome+predaway,4),"actual_home_margin":hs-vs,"actual_total":hs+vs,
                "model_version":"rolling_team_10_home3_v1"})
        date=str(g["date"])[:10]
        history[hid].append((date,float(hs),float(vs)))
        history[aid].append((date,float(vs),float(hs)))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument("--games",required=True);p.add_argument("--out",required=True)
    a=p.parse_args();games=[json.loads(x) for x in Path(a.games).read_text().splitlines() if x.strip()]
    if len(games)!=1230:raise ValueError("Expected verified complete 1230-game regular season")
    out=predict(games)
    if len(out)<900:raise ValueError("Insufficient baseline predictions")
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(x)+"\n" for x in out))
    import statistics
    print(json.dumps({"predictions":len(out),"spread_margin_mae":round(statistics.mean(abs(x["predicted_home_margin"]-x["actual_home_margin"]) for x in out),4),
        "total_mae":round(statistics.mean(abs(x["predicted_total"]-x["actual_total"]) for x in out),4),
        "warning":"No historical market thresholds; these are score projection errors, NOT betting results"}))
if __name__=="__main__":main()
