"""The ranking model.

We predict each stock's *next-period cross-sectional return* and trade the
top-ranked names. Per the research, gradient-boosted trees are the proven
workhorse for tabular financial features (they beat deep nets here and handle
regime-dependent, non-linear factor interactions well).

Three tiers, chosen automatically by availability:
  1. LightGBM (preferred)
  2. scikit-learn HistGradientBoostingRegressor (no extra install)
  3. Equal-weight z-score factor blend (pure fallback, always works)

The z-score blend doubles as an honest baseline: if the ML model can't beat a
naive factor average, the "AI" is adding nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, cross_sectional_zscore


def _lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401

        return True
    except Exception:
        return False


class RankingModel:
    """Fit on stacked past (features -> forward return), predict a score."""

    def __init__(self, use_lightgbm: bool = True, random_state: int = 42):
        self.random_state = random_state
        self.kind = "zscore"  # default fallback
        self.model = None

        if use_lightgbm and _lightgbm_available():
            self.kind = "lightgbm"
        else:
            try:
                from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: F401

                self.kind = "sklearn_gbr"
            except Exception:
                self.kind = "zscore"

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RankingModel":
        X = X[FEATURE_COLUMNS]
        mask = y.notna() & np.isfinite(X).all(axis=1)
        X, y = X[mask], y[mask]
        if len(y) < 50:
            # Too little data to train responsibly -> fall back to z-score blend.
            self.kind = "zscore"
            return self

        if self.kind == "lightgbm":
            import lightgbm as lgb

            self.model = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.03,
                num_leaves=15,
                max_depth=4,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                min_child_samples=30,
                reg_lambda=1.0,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
            self.model.fit(X, y)
        elif self.kind == "sklearn_gbr":
            from sklearn.ensemble import HistGradientBoostingRegressor

            self.model = HistGradientBoostingRegressor(
                max_depth=4,
                learning_rate=0.05,
                max_iter=300,
                l2_regularization=1.0,
                random_state=self.random_state,
            )
            self.model.fit(X.values, y.values)
        # zscore: nothing to fit
        return self

    # -------------------------------------------------------------- predict
    def predict(self, X: pd.DataFrame) -> pd.Series:
        Xf = X[FEATURE_COLUMNS]
        if self.kind == "zscore" or self.model is None:
            return self._zscore_score(Xf)
        if self.kind == "lightgbm":
            preds = self.model.predict(Xf)
        else:
            preds = self.model.predict(Xf.fillna(0.0).values)
        return pd.Series(preds, index=Xf.index)

    @staticmethod
    def _zscore_score(X: pd.DataFrame) -> pd.Series:
        """Hand-crafted factor blend used as the no-ML baseline.

        Signs encode priors: momentum positive, low-vol positive (so we negate
        volatility), short-term reversal negative, uptrend positive.
        """
        z = cross_sectional_zscore(X)
        score = (
            0.30 * z["mom_12_1"]
            + 0.20 * z["mom_6_1"]
            + 0.10 * z["mom_3_1"]
            - 0.15 * z["rev_1m"]
            - 0.15 * z["vol_6m"]
            + 0.10 * z["dist_200d"]
            - 0.10 * z["downside_vol"]
        )
        return score

    def feature_importance(self) -> pd.Series | None:
        if self.kind == "lightgbm" and self.model is not None:
            return pd.Series(
                self.model.feature_importances_, index=FEATURE_COLUMNS
            ).sort_values(ascending=False)
        return None
