"""ALS with a custom graded satisfaction target and unchanged confidence."""

import numpy as np

from .ials import IALS


class CustomALS(IALS):
    """Use p*=clip((r-2)/3,0,1): 1,2→0; 3→1/3; 4→2/3; 5→1.

    A neutral rating carries a weak preference and an excellent rating carries
    more preference than a good rating. Missing entries still have target zero.
    This changes the regression target, NOT confidence or hyperparameters.
    Training, confidence, initialisation and ranking are inherited unchanged.
    """

    @staticmethod
    def targets(ratings):
        return np.clip((np.asarray(ratings, dtype=np.float64) - 2.0) / 3.0, 0, 1)
