"""Research-only matched-fold ablation: context-aware minutes vs + return-gap features.

Input rows MUST be built from pregame-known history, with fields:
date (NBA game date), game_id, x (base pregame features), gap_team_games,
has_team_history, minutes_actual. Data preparation and provenance are
separate gates. Never interpret this as a DNP or betting model.
"""
from collections import defaultdict
from statistics import mean
from random import Random
import numpy as np
from sklearn.linear_model import Ridge

WINDOWS = (("2026-01-15","2026-02-15"),("2026-02-15","2026-03-12"),
           ("2026-03-12","2026-04-13"))

def _ci(test, gain, repeats=1000, seed=21):
    groups=defaultdict(list)
    for r,d in zip(test,gain):
        groups[r["game_id"]].append(float(d))
    if len(groups)<5:
        return None
    clusters=[(sum(v),len(v)) for v in groups.values()]
    rng=Random(seed)
    draws=[]
    for _ in range(repeats):
        sample=rng.choices(clusters,k=len(clusters))
        draws.append(sum(s for s,n in sample)/sum(n for s,n in sample))
    draws.sort()
    return [round(draws[int((repeats-1)*p)],5) for p in (.025,.975)]

def evaluate(rows, windows=WINDOWS, alpha=25):
    rows=list(rows)
    if not rows:
        raise ValueError("empty research data")
    seen=set()
    for r in rows:
        key=(r["game_id"],r["date"],r.get("player_id"))
        if key in seen:
            raise ValueError("duplicate game/date/player")
        seen.add(key)
        if not r.get("x") or r["gap_team_games"]<0 or r["has_team_history"] not in (0,1):
            raise ValueError("invalid pregame feature row")
        if r.get("pregame_verified") is not True:
            raise ValueError("pregame provenance not verified")
    output={}
    for start,end in windows:
        train=[r for r in rows if r["date"]<start]
        test=[r for r in rows if start<=r["date"]<end]
        if len(train)<30 or len({r["game_id"] for r in test})<5:
            raise ValueError("insufficient chronological samples")
        def x(r,augmented):
            base=list(r["x"])
            return base+([min(r["gap_team_games"],10),int(r["gap_team_games"]>=3),
                          r["has_team_history"]] if augmented else [])
        y=np.array([r["minutes_actual"] for r in train])
        b=Ridge(alpha=alpha).fit(np.array([x(r,False) for r in train]),y)
        c=Ridge(alpha=alpha).fit(np.array([x(r,True) for r in train]),y)
        actual=np.array([r["minutes_actual"] for r in test])
        bp=np.clip(b.predict(np.array([x(r,False) for r in test])),0,53)
        cp=np.clip(c.predict(np.array([x(r,True) for r in test])),0,53)
        gain=np.abs(actual-bp)-np.abs(actual-cp)
        def score(mask):
            selected=[r for r,m in zip(test,mask) if m]
            if len(selected)<10 or len({r["game_id"] for r in selected})<5:
                return None
            m=np.array(mask,dtype=bool)
            return {"n":len(selected),
                    "baseline_mae":round(float(np.abs(actual[m]-bp[m]).mean()),5),
                    "augmented_mae":round(float(np.abs(actual[m]-cp[m]).mean()),5),
                    "gain":round(float(gain[m].mean()),5),
                    "gain_ci95_game_cluster":_ci(selected,gain[m])}
        output[start]={"train":len(train),"test":len(test),
                       "all":score([True]*len(test)),
                       "return_3plus":score([r["gap_team_games"]>=3 for r in test]),
                       "return_1_2":score([1<=r["gap_team_games"]<=2 for r in test]),
                       "consecutive":score([r["gap_team_games"]==0 for r in test])}
    return output
