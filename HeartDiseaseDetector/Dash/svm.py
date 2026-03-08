"""
Runtime SVM helper for heart disease prediction.

The actual model training happens in `ModelTraining/svm.py`. This module only
handles loading the already-trained model artefacts and exposing a simple
`predict` helper that the Django views can use.
"""

from pathlib import Path
from typing import Dict, Iterable, List

import joblib
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "ModelTraining" / "heart_svm_model.pkl"


class HeartDiseaseSVM:
    """
    Thin wrapper around the trained SVM classifier.

    This class loads the pickled model/scaler bundle produced by the training
    script and exposes a `predict` method that accepts a feature dictionary.
    """

    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. "
                "Run the training script in `ModelTraining/svm.py` first."
            )

        bundle: Dict = joblib.load(MODEL_PATH)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.features: List[str] = bundle["features"]

    def _vectorise(self, feature_values: Dict[str, float]) -> np.ndarray:
        """
        Convert a dictionary of feature_name -> value into the correct
        ordered numpy array expected by the scaler/model.
        """
        ordered: Iterable[float] = [feature_values[name] for name in self.features]
        arr = np.array(ordered, dtype=float).reshape(1, -1)
        return self.scaler.transform(arr)

    def predict(self, feature_values: Dict[str, float]) -> int:
        """
        Predict the presence of heart disease.

        Returns:
            0 for "No Heart Disease"
            1 for "Heart Disease"
        """
        vector = self._vectorise(feature_values)
        pred = self.model.predict(vector)
        # `.predict` returns a numpy array; pull out the scalar.
        return int(pred[0])
