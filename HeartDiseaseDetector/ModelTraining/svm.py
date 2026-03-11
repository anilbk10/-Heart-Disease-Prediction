"""
Train and compare two models for Heart Disease prediction:

1) From-scratch Linear SVM (your implementation)
2) RBF Kernel SVM (scikit-learn SVC)

This script:
- Loads `heart.csv`
- Splits train/test
- Scales features with `StandardScaler`
- Trains both models
- Prints: accuracy + confusion matrix + classification report
- Saves two bundles to disk, which the Django app can load for prediction

Saved bundles contain:
  {"model": <classifier>, "scaler": <StandardScaler>, "features": [col names]}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path(__file__).resolve().parent / "heart.csv"

# Two separate model files.
LINEAR_MODEL_PATH = BASE_DIR / "ModelTraining" / "heart_svm_linear_scratch.pkl"
RBF_MODEL_PATH = BASE_DIR / "ModelTraining" / "heart_svm_rbf_sklearn.pkl"

# Django runtime default expects this name.
DEFAULT_MODEL_PATH = BASE_DIR / "ModelTraining" / "heart_svm_model.pkl"


FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


class ScratchLinearSVM:
    """
    From-scratch linear soft-margin SVM using hinge loss + L2 regularisation.

    Mirrors the notebook implementation:
      - SGD update over individual samples
      - Labels in {-1, +1} during training

    `predict` returns {0,1} labels for compatibility with the Django app.
    """

    def __init__(self, learning_rate: float = 0.0001, lambda_param: float = 0.001, n_iters: int = 5000) -> None:
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScratchLinearSVM":
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0

        for _ in range(self.n_iters):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for idx in indices:
                x_i = X[idx]
                y_i = y[idx]

                condition = y_i * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - y_i * x_i)
                    self.b += self.lr * y_i

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model is not fitted yet.")
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        labels = np.where(scores >= 0, 1, -1)
        return np.where(labels == 1, 1, 0)


def _load_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Place `heart.csv` from the UCI Heart Disease dataset here."
        )
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURE_NAMES].values
    y = data["target"].values.astype(int)
    return X, y, FEATURE_NAMES


def _print_report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))


def train_and_compare_models(export_default: str = "linear") -> None:
    X, y, features = _load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1) Scratch Linear SVM
    y_train_svm = np.where(y_train == 0, -1, 1).astype(float)
    scratch = ScratchLinearSVM(learning_rate=0.0001, lambda_param=0.001, n_iters=5000)
    scratch.fit(X_train_scaled, y_train_svm)
    scratch_pred = scratch.predict(X_test_scaled)
    _print_report("Scratch Linear SVM", y_test, scratch_pred)
    joblib.dump({"model": scratch, "scaler": scaler, "features": features}, LINEAR_MODEL_PATH)
    print(f"Saved: {LINEAR_MODEL_PATH}")

    # 2) Sklearn RBF SVM (GridSearchCV)
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", 0.01, 0.1, 1],
    }
    base = SVC(kernel="rbf")
    gs = GridSearchCV(base, param_grid=param_grid, scoring="accuracy", cv=5, n_jobs=-1)
    gs.fit(X_train_scaled, y_train)
    rbf_best = gs.best_estimator_
    rbf_pred = rbf_best.predict(X_test_scaled)
    _print_report(f"Sklearn RBF SVM (best={gs.best_params_})", y_test, rbf_pred)
    joblib.dump({"model": rbf_best, "scaler": scaler, "features": features}, RBF_MODEL_PATH)
    print(f"Saved: {RBF_MODEL_PATH}")

    # Export default for Django runtime
    if export_default == "rbf":
        joblib.dump({"model": rbf_best, "scaler": scaler, "features": features}, DEFAULT_MODEL_PATH)
        print(f"Default export: {DEFAULT_MODEL_PATH} -> RBF")
    else:
        joblib.dump({"model": scratch, "scaler": scaler, "features": features}, DEFAULT_MODEL_PATH)
        print(f"Default export: {DEFAULT_MODEL_PATH} -> Scratch Linear")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and compare Heart Disease SVM models.")
    p.add_argument(
        "--export-default",
        choices=["linear", "rbf"],
        default="linear",
        help="Which model to export to heart_svm_model.pkl for Django runtime.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_and_compare_models(export_default=args.export_default)
