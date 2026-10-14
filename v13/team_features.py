"""Chronological pregame team scoring and possession-proxy features."""
from collections import defaultdict
from datetime import date

def _id(team):
    return team.get("id") if isinstance(team, dict) else team

def team_features(games, window=10):
    """Features use only games on strictly earlier dates (no same-day leakage)."""
    history = defaultdict(list)
    rows = []
    ordered = sorted(games, key=lambda g: (str(g.get("date","")), str(g.get("id",""))))
    i = 0
    while i < len(ordered):
        day = str(ordered[i].get("date",""))[:10]
        batch = []
        while i < len(ordered) and str(ordered[i].get("date",""))[:10] == day:
            batch.append(ordered[i])
            i += 1
        for g in batch:
            home, away = _id(g.get("home_team")), _id(g.get("visitor_team"))
            if home is None or away is None: continue
            item = {"game_id":g.get("id"),"date":day}
            for prefix, team in (("home",home),("away",away)):
                past = history[team][-window:]
                item[prefix+"_games_seen"] = len(history[team])
                item[prefix+"_points_for_avg"] = round(sum(p[0] for p in past)/len(past),3) if past else None
                item[prefix+"_points_against_avg"] = round(sum(p[1] for p in past)/len(past),3) if past else None
                item[prefix+"_point_diff_avg"] = round(sum(p[0]-p[1] for p in past)/len(past),3) if past else None
            rows.append(item)
        for g in batch:
            h,a = _id(g.get("home_team")),_id(g.get("visitor_team"))
            hs,vs=g.get("home_team_score"),g.get("visitor_team_score")
            if h is None or a is None or g.get("status")!="Final" or not isinstance(hs,(int,float)) or not isinstance(vs,(int,float)): continue
            history[h].append((hs,vs))
            history[a].append((vs,hs))
    return rows
