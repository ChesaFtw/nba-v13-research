"""Research-only conservative player-minutes interval helper."""
from datetime import date, timedelta
from math import ceil, isfinite

def role_label(recent_mean, recent_sd):
    mean=float(recent_mean);sd=float(recent_sd)
    if not all(isfinite(v) and v>=0 for v in (mean,sd)):
        raise ValueError("invalid past minutes")
    return ("low" if mean<15 else ("high" if mean>=30 else "mid"),
            "volatile" if sd>=8 else "stable")

def low_stable_widen(base_radius, joint_radius, recent_mean, recent_sd):
    b=float(base_radius);j=float(joint_radius)
    if not all(isfinite(v) and v>=0 for v in (b,j)):
        raise ValueError("invalid radius")
    return max(b,j) if role_label(recent_mean,recent_sd)==("low","stable") else b

def prior_day_radius(history,target_day,*,days=14,minimum=100,floor=0,coverage=.9):
    if days<1 or minimum<20 or not 0<coverage<1:
        raise ValueError("invalid settings")
    target=date.fromisoformat(target_day)
    low=target-timedelta(days=days)
    vals=[]
    for d,e in history:
        day=date.fromisoformat(d);v=float(e)
        if day>target or not isfinite(v) or v<0:
            raise ValueError("invalid history")
        if low<=day<target: vals.append(v)
    floor=float(floor)
    if not isfinite(floor) or floor<0:raise ValueError("invalid floor")
    if len(vals)<minimum:return floor,len(vals)
    vals.sort()
    return max(floor,vals[min(len(vals),ceil((len(vals)+1)*coverage))-1]),len(vals)
