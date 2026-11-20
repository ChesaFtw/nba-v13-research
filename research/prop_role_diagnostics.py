"""Offline diagnostics for paired subgroup error summaries."""
from math import isfinite


def error_ratio(high_errors, other_errors):
    high = list(high_errors)
    other = list(other_errors)
    if not high or not other:
        raise ValueError('both groups required')
    if any(not isfinite(float(x)) or float(x) < 0 for x in high + other):
        raise ValueError('invalid absolute errors')
    denominator = sum(other) / len(other)
    if denominator == 0:
        raise ValueError('undefined ratio')
    return (sum(high) / len(high)) / denominator
