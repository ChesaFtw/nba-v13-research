"""Research-only post-outcome interval-score tail decomposition; never a pregame feature."""
import numpy as np

def tail_decomposition(abs_error, baseline_score, challenger_score, *, quantiles=(.9,.95,.99)):
    arrays=[np.asarray(x,dtype=float) for x in (abs_error,baseline_score,challenger_score)]
    err,base,chall=arrays
    if err.ndim!=1 or not len(err) or any(x.shape!=err.shape or not np.isfinite(x).all() for x in arrays) or (err<0).any():
        raise ValueError('invalid arrays')
    if any(not 0<q<1 for q in quantiles):raise ValueError('invalid quantiles')
    diff=chall-base
    out={'n':len(err),'overall_delta':float(diff.mean()),'median_delta':float(np.median(diff))}
    for q in quantiles:
        cutoff=float(np.quantile(err,q));top=err>=cutoff;rest=~top
        out[str(q)]={'cutoff_error':cutoff,'top_n':int(top.sum()),'top_mean_delta':float(diff[top].mean()),
             'rest_n':int(rest.sum()),'rest_mean_delta':float(diff[rest].mean()) if rest.any() else None,
             'top_contribution_to_overall_delta':float(diff[top].sum()/len(err)),
             'rest_contribution_to_overall_delta':float(diff[rest].sum()/len(err))}
    return out
