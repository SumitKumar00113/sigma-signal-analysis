"""Learned modulation classifier: model wrapper, persistence and lookup.

The estimator is scikit-learn's histogram gradient boosting on the
:mod:`src.ml.features` vector – fast to train on a CPU, no GPU needed,
robust to un-scaled features, and small on disk.  Files are ``joblib``
dumps carrying the class list, feature names and library versions, so a
model trained with a different feature set is rejected instead of giving
silent garbage.

Model lookup order (:func:`load_default_model`): the ``SIGMA_MODEL``
environment variable, ``~/.sigma/models/modulation_classifier.joblib``,
then the model bundled with the application (``src/ml/models``).
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.core.enums import ModulationType
from src.ml.features import FEATURE_NAMES

MODEL_FILENAME = "modulation_classifier.joblib"
BUNDLED_MODEL = Path(__file__).resolve().parent / "models" / MODEL_FILENAME
USER_MODEL = Path.home() / ".sigma" / "models" / MODEL_FILENAME


def ml_available() -> bool:
    """True if the optional ``[ml]`` dependencies are installed."""
    try:
        import joblib  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class Prediction:
    modulation: ModulationType
    probability: float
    ranked: list[tuple[ModulationType, float]] = field(default_factory=list)


@dataclass
class ModulationModel:
    estimator: Any
    classes: tuple[ModulationType, ...]
    feature_names: tuple[str, ...] = FEATURE_NAMES
    info: dict[str, Any] = field(default_factory=dict)

    # ---- inference --------------------------------------------------------

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(features, dtype=np.float32))
        proba = self.estimator.predict_proba(x)
        # The estimator's classes_ are class indices; map to our order
        out = np.zeros((x.shape[0], len(self.classes)))
        out[:, np.asarray(self.estimator.classes_, dtype=int)] = proba
        return out

    def predict(self, features: np.ndarray) -> Prediction:
        p = self.predict_proba(features)[0]
        order = np.argsort(p)[::-1]
        ranked = [(self.classes[i], float(p[i])) for i in order]
        return Prediction(ranked[0][0], ranked[0][1], ranked)

    # ---- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        import joblib
        import sklearn

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        info = dict(self.info)
        info.setdefault("created", datetime.now(UTC).isoformat())
        info["sklearn_version"] = sklearn.__version__
        info["python"] = platform.python_version()
        joblib.dump({
            "format": 1,
            "estimator": self.estimator,
            "classes": [c.value for c in self.classes],
            "feature_names": list(self.feature_names),
            "info": info,
        }, path, compress=3)
        return path

    @classmethod
    def load(cls, path: str | Path) -> ModulationModel:
        import joblib

        data = joblib.load(path)
        if not isinstance(data, dict) or data.get("format") != 1:
            raise ValueError(f"{path}: not a Sigma modulation model")
        names = tuple(data["feature_names"])
        if names != FEATURE_NAMES:
            raise ValueError(f"{path}: trained with a different feature set; retrain it "
                             "with `python -m src.ml.train`")
        classes = tuple(ModulationType(v) for v in data["classes"])
        return cls(data["estimator"], classes, names, dict(data.get("info", {})))


def train_model(x: np.ndarray, y: np.ndarray, classes: tuple[ModulationType, ...],
                seed: int = 0, info: dict[str, Any] | None = None) -> ModulationModel:
    """Fit the gradient-boosting classifier (class indices in *y*)."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    # Early stopping needs a stratified validation split with every class
    # present; small data sets (a few labelled recordings) train without it
    n_classes = len(np.unique(y))
    enough = len(y) * 0.1 >= 2 * n_classes
    est = HistGradientBoostingClassifier(
        max_iter=400 if enough else 150, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1e-3, early_stopping=enough, validation_fraction=0.1,
        n_iter_no_change=25, random_state=seed,
    )
    est.fit(x, y)
    return ModulationModel(est, classes, FEATURE_NAMES, dict(info or {}))


def default_model_path() -> Path | None:
    env = os.environ.get("SIGMA_MODEL")
    for p in ([Path(env)] if env else []) + [USER_MODEL, BUNDLED_MODEL]:
        if p.is_file():
            return p
    return None


@lru_cache(maxsize=4)
def _load_cached(path: str, mtime: float) -> ModulationModel:
    return ModulationModel.load(path)


def load_default_model() -> ModulationModel | None:
    """The first model found (see module docstring), or None if there is no
    model or the ``[ml]`` extras are not installed."""
    if not ml_available():
        return None
    path = default_model_path()
    if path is None:
        return None
    try:
        return _load_cached(str(path), path.stat().st_mtime)
    except Exception:  # noqa: BLE001 – a broken model must never break analysis
        return None
