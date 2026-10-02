import numpy as np
from exdrkg_rerun.common.stats import krippendorff_alpha_ordinal


def test_perfect_agreement_is_one():
    M = np.array([[1, 1, 1], [3, 3, 3], [5, 5, 5], [2, 2, 2]], float)
    assert abs(krippendorff_alpha_ordinal(M) - 1.0) < 1e-9


def test_random_ratings_are_near_zero_and_missing_is_allowed():
    rng = np.random.default_rng(1); M = rng.integers(1, 6, (200, 3)).astype(float); M[::7, 2] = np.nan
    assert abs(krippendorff_alpha_ordinal(M)) < 0.2
