"""Research-only prior-game minutes volatility and role-change scores."""
from math import isfinite

def tail_risk_features(row):
    x=row['x']
    if len(x)!=12: raise ValueError('expected 12 pregame features')
    m10,last,sd10,gap,changed=map(float,(x[1],x[3],x[4],x[6],x[10]))
    if not all(isfinite(v) for v in (m10,last,sd10,gap,changed)):
        raise ValueError('nonfinite features')
    if min(m10,last,sd10,gap)<0 or changed not in (0,1):
        raise ValueError('invalid features')
    jump=abs(last-m10)
    return {'volatility':sd10,'last_game_jump':jump,'appearance_gap':gap,
            'composite':sd10+.5*jump+.35*max(0,min(gap-1,10))+5*changed}
