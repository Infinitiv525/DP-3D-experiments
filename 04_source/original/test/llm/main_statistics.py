"""Preregistered crossed paired bootstrap on already aggregated scores only.

Does not read datasets or choose models. Four LLM-TPE primary contrasts, with
Bonferroni-adjusted percentile intervals. Small-repeat coverage is approximate.
"""
import numpy as np


def primary_contrasts(scores, *, replicates=10000, seed=510001):
    values = np.asarray(scores, dtype=np.float64)
    if (values.ndim != 4 or values.shape[0] != 6 or values.shape[1] < 2 or values.shape[2:] != (3, 5)
            or not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError('Complete finite nonnegative method6 x geometryG x search3 x final_seed5 scores required')
    if type(replicates) is not int or replicates != 10000 or seed != 510001:
        raise ValueError('Frozen resampling settings required')
    differences = values[2:] - values[1:2]
    rng = np.random.default_rng(seed); samples = np.empty((replicates, 4), dtype=np.float64)
    for i in range(replicates):
        g = rng.integers(0, values.shape[1], size=values.shape[1])
        r = rng.integers(0, 3, size=3); f = rng.integers(0, 5, size=5)
        samples[i] = differences[:, g[:, None, None], r[None, :, None], f[None, None, :]].mean(axis=(1, 2, 3))
    intervals = np.quantile(samples, [.00625, .99375], axis=0, method='linear')
    from main_protocol import METHODS
    return dict(method='crossed_paired_percentile_bootstrap_bonferroni4', replicates=replicates, seed=seed,
        nominal_familywise_confidence=.95, marginal_confidence=.9875, quantiles=[.00625, .99375],
        coverage_warning='Approximate bootstrap coverage; only3search repeats. Not an exact finite-sample95percent guarantee.',
        contrasts=[dict(method=m, comparator='tpe', mean_difference=float(differences[j].mean()),
                        lower=float(intervals[0, j]), upper=float(intervals[1, j]),
                        supports_lower_error=bool(intervals[1, j] < 0),
                        paired_repeat_effects=differences[j].mean(axis=(0, 2)).tolist()) for j, m in enumerate(METHODS[2:])])
