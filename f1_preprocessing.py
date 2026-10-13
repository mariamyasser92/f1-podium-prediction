
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class NumericProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, order='impute_scale'):
        self.order = order

    def fit(self, X, y=None):
        a = np.asarray(X, dtype=float)
        a = np.where(np.isfinite(a), a, np.nan)

        # Preserve all-missing columns consistently across sklearn versions.
        self.median_ = np.array([
            np.median(c[np.isfinite(c)])
            if np.isfinite(c).any()
            else 0.
            for c in a.T
        ])

        if self.order == 'clip_impute_scale':

            bounds = np.array([
                np.quantile(
                    c[np.isfinite(c)],
                    [.01, .99]
                )
                if np.isfinite(c).any()
                else [0., 0.]
                for c in a.T
            ])

            self.low_, self.high_ = (
                bounds[:, 0],
                bounds[:, 1]
            )

            clipped = np.clip(
                a,
                self.low_,
                self.high_
            )

            self.median_ = np.array([
                np.median(c[np.isfinite(c)])
                if np.isfinite(c).any()
                else 0.
                for c in clipped.T
            ])

        elif self.order == 'impute_clip_scale':

            imputed = np.where(
                np.isnan(a),
                self.median_,
                a
            )

            self.low_, self.high_ = np.quantile(
                imputed,
                [.01, .99],
                axis=0
            )

        else:

            self.low_ = np.full(
                a.shape[1],
                -np.inf
            )

            self.high_ = np.full(
                a.shape[1],
                np.inf
            )

        self.scaler_ = StandardScaler().fit(
            self._values(a)
        )

        return self

    def _values(self, a):

        if self.order == 'clip_impute_scale':

            a = np.clip(
                a,
                self.low_,
                self.high_
            )

            return np.where(
                np.isnan(a),
                self.median_,
                a
            )

        return np.clip(
            np.where(
                np.isnan(a),
                self.median_,
                a
            ),
            self.low_,
            self.high_
        )

    def transform(self, X):

        a = np.asarray(
            X,
            dtype=float
        )

        a = np.where(
            np.isfinite(a),
            a,
            np.nan
        )

        return np.hstack([
            self.scaler_.transform(
                self._values(a)
            ),
            np.isnan(a).astype(float)
        ])

    def get_feature_names_out(
        self,
        input_features=None
    ):

        names = (
            list(input_features)
            if input_features is not None
            else [
                f'x{i}'
                for i in range(
                    len(self.median_)
                )
            ]
        )

        return np.array(
            names
            + [
                s + '_missing'
                for s in names
            ],
            dtype=object
        )
