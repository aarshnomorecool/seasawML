"""
SeaSaw XGBoost Trainer
=======================
Phase 5 (XGBoost half). Trains one XGBRegressor per horizon on the flat
tabular datasets produced by Phase 4.

Does a small grid search over n_estimators / max_depth / learning_rate /
subsample (CLAUDE.md Phase 5 Spec), picking the combination with the best
validation RMSE. Early stopping (against the validation split) caps the
number of boosting rounds actually used within each candidate, so
n_estimators in the grid acts as an upper bound rather than a fixed count.

Usage
-----
    from src.models.xgb_trainer import XGBTrainer

    trainer = XGBTrainer(horizon="A")
    trainer.train(X_train, y_train, X_val, y_val)
    trainer.save("models/xgb_horizon_A.pkl")
"""

import logging
import pickle
from itertools import product
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

DEFAULT_PARAM_GRID = {
    "n_estimators": [300, 600],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
}


class XGBTrainer:
    """
    Parameters
    ----------
    horizon               : label used only for logging (e.g. "A", "B", "C")
    early_stopping_rounds : rounds without val-RMSE improvement before stopping
    param_grid            : dict of hyperparameter name -> candidate list;
                             defaults to DEFAULT_PARAM_GRID
    """

    def __init__(
        self,
        horizon: str,
        early_stopping_rounds: int = 20,
        param_grid: Dict[str, List] = None,
    ):
        self.horizon = horizon
        self.early_stopping_rounds = early_stopping_rounds
        self.param_grid = param_grid or DEFAULT_PARAM_GRID
        self.model: XGBRegressor = None
        self.best_params: Dict = None
        self.best_val_rmse: float = None

    def _candidates(self):
        keys = list(self.param_grid.keys())
        for values in product(*self.param_grid.values()):
            yield dict(zip(keys, values))

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> XGBRegressor:
        best_model = None
        best_rmse = np.inf
        best_params = None

        for params in self._candidates():
            model = XGBRegressor(
                objective="reg:squarederror",
                early_stopping_rounds=self.early_stopping_rounds,
                eval_metric="rmse",
                n_jobs=-1,
                **params,
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            preds = model.predict(X_val)
            rmse = float(np.sqrt(mean_squared_error(y_val, preds)))

            logger.info(
                f"[{self.horizon}] params={params} "
                f"best_iteration={model.best_iteration} val_rmse={rmse:.4f}"
            )

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_params = params

        self.model = best_model
        self.best_params = best_params
        self.best_val_rmse = best_rmse
        logger.info(f"[{self.horizon}] BEST params={best_params} val_rmse={best_rmse:.4f}")
        return self.model

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Saved XGBoost model -> {path}")

    @staticmethod
    def load(path: Union[str, Path]) -> XGBRegressor:
        with open(path, "rb") as f:
            return pickle.load(f)
