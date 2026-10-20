"""Research-only symmetric interval scoring and game-cluster inference."""
from collections import defaultdict
import numpy as np

def interval_score(abs_error, radius, *, alpha=.1):
    """Central (1-alpha) symmetric interval score; lower is better."""
    if not 0 < alpha < 1: raise ValueError('invalid alpha')
    e=np.asarray(abs_error,dtype=float);r=np.asarray(radius,dtype=float)
    if e.shape!=r.shape or e.ndim!=1 or e.size==0 or not np.all(np.isfinite(e)) or not np.all(np.isfinite(r)) or np.any(e<0) or np.any(r<0):
        raise ValueError('invalid residuals/radii')
    return 2*r+(2/alpha)*np.maximum(e-r,0)

def paired_cluster_ci(rows, score_deltas, *, reps=2000, seed=2026):
    """Bootstrap difference in mean score by resampling complete NBA games."""
    deltas=np.asarray(score_deltas,dtype=float)
    if len(rows)!=len(deltas) or not np.all(np.isfinite(deltas)):
        raise ValueError('invalid scores')
    bygame=defaultdict(list)
    for r,d in zip(rows,deltas):bygame[r['game_id']].append(float(d))
    if len(bygame)<5 or reps<100:raise ValueError('insufficient games/replicates')
    sums=np.array([sum(v) for v in bygame.values()]);counts=np.array([len(v) for v in bygame.values()]);n=len(sums)
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,n,size=(reps,n))
    return [round(float(x),6) for x in np.quantile(sums[draws].sum(axis=1)/counts[draws].sum(axis=1),[.025,.975])]
