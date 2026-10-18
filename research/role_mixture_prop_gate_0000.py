"""Offline V13 research: pregame minutes-residual mixture probabilities.

Training uses earlier realized minutes. Prediction never reads target-game
minutes, attempts or makes. Not connected to production or wagering.
"""
from datetime import date
from math import isfinite
import numpy as np
from scipy.stats import nbinom
from minutes_interval_role_gate_2300 import GROUPS, group_and_features

GRID = np.linspace(.02, .98, 25)
R = 16.0
FLOOR = .05


def fit_residual_quantiles(training_rows, cutoff):
    end = date.fromisoformat(cutoff)
    values = {'pooled': []}
    values.update({g: [] for g in GROUPS})
    for row in training_rows:
        try:
            when = date.fromisoformat(row['date'])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError('invalid training date') from exc
        if when >= end:
            raise ValueError('training target on or after cutoff')
        actual = row.get('min')
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not isfinite(actual) or not .1 <= actual <= 65:
            raise ValueError('invalid training target minutes')
        group, projected, _, _ = group_and_features(row)
        delta = float(actual - projected)
        values['pooled'].append(delta)
        values[group].append(delta)
    fitted = {}
    for key, vals in values.items():
        if len(vals) < 100:
            raise ValueError('insufficient training observations: ' + key)
        fitted[key] = {'n':len(vals), 'quantiles':np.quantile(np.asarray(vals), GRID).astype(float).tolist()}
    return fitted


def _checked_quantiles(fitted, group):
    x = fitted.get(group)
    if not isinstance(x, dict) or x.get('n', 0) < 100:
        raise ValueError('missing or insufficient group')
    q = x.get('quantiles')
    if not isinstance(q, (list, tuple)) or len(q) != 25 or any(isinstance(z, bool) or not isinstance(z, (int, float)) or not isfinite(z) for z in q):
        raise ValueError('invalid quantile grid')
    q = np.asarray(q, dtype=float)
    if np.any(np.diff(q) < 0):
        raise ValueError('unordered quantiles')
    return q


def pregame_mixture(rows, fitted, role_specific):
    """Return arrays of forecast probabilities/means only; never accesses targets."""
    if not isinstance(role_specific, bool):
        raise ValueError('role_specific must be boolean')
    if not rows:
        raise ValueError('empty forecast batch')
    groups, projected, rate, shooting = [], [], [], []
    for row in rows:
        g, m, r, s = group_and_features(row)
        groups.append(g)
        projected.append(m)
        rate.append(r)
        shooting.append(s)
    projected = np.asarray(projected)
    rate = np.asarray(rate)
    shooting = np.asarray(shooting)
    labels = [g if role_specific else 'pooled' for g in groups]
    q = np.asarray([_checked_quantiles(fitted, g) for g in labels])
    minutes = np.clip(projected[:,None] + q, .1, 65)
    mu = np.maximum(FLOOR, minutes * rate[:,None])
    att_p = R/(R+mu)
    made_p = R/(R+mu*shooting[:,None])
    out = {'mu':mu,
           'mean_attempts':mu.mean(axis=1),
           'made3':nbinom.sf(2, R, made_p).mean(axis=1),
           'made2':nbinom.sf(1, R, made_p).mean(axis=1),
           'tail6':nbinom.sf(5, R, att_p).mean(axis=1),
           'zero':nbinom.pmf(0, R, att_p).mean(axis=1),
           'groups':np.asarray(groups)}
    for key in ('made3','made2','tail6','zero'):
        if not np.all(np.isfinite(out[key])) or not np.all((out[key] >= 0) & (out[key] <= 1)):
            raise ValueError('invalid probability')
    return out
