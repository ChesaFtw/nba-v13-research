from collections import defaultdict
from math import isfinite
from random import Random

def paired_mae(rows, *, iterations=2000, seed=21):
    if iterations < 200:
        raise ValueError("iterations must be >= 200")
    groups = defaultdict(list)
    bsum = csum = 0.0
    n = 0
    for r in rows:
        gid = r.get("game_id")
        if gid is None or str(gid).strip() == "":
            raise ValueError("game_id is required")
        a,b,c = (float(r[k]) for k in ("actual","baseline","challenger"))
        if not all(isfinite(v) for v in (a,b,c)):
            raise ValueError("nonfinite value")
        be,ce = abs(a-b),abs(a-c)
        groups[str(gid)].append(be-ce)
        bsum += be
        csum += ce
        n += 1
    if len(groups)<5:
        raise ValueError("at least five independent games required")
    vals = [(sum(v),len(v)) for _,v in sorted(groups.items())]
    rng = Random(seed)
    samples = []
    for _ in range(iterations):
        draw = rng.choices(vals,k=len(vals))
        samples.append(sum(x for x,y in draw)/sum(y for x,y in draw))
    samples.sort()
    return {"rows":n,"games":len(groups),"baseline_mae":bsum/n,
            "challenger_mae":csum/n,"improvement":(bsum-csum)/n,
            "ci95_game_cluster":[samples[int(.025*(iterations-1))],
                                 samples[int(.975*(iterations-1))]],
            "iterations":iterations}
