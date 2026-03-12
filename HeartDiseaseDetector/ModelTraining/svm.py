"""
Train and compare two models for Heart Disease prediction:

1) From-scratch Linear SVM (primal SGD)
2) From-scratch RBF Kernel SVM (SMO-style, using your implementation)

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path(__file__).resolve().parent / "heart.csv"

# Two separate model files.
LINEAR_MODEL_PATH = BASE_DIR / "ModelTraining" / "heart_svm_linear_scratch.pkl"
RBF_MODEL_PATH = BASE_DIR / "ModelTraining" / "heart_svm_rbf_scratch.pkl"

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


class KernelSVM:
    """
    From-scratch soft-margin SVM with linear / RBF kernel.

    This is adapted from your SMO-style implementation. It operates in the
    dual using Lagrange multipliers `alpha` and supports both linear and RBF
    kernels. Labels are expected to be in {-1, +1} during training.
    """

    def __init__(
        self,
        C: float = 1.0,
        tol: float = 1e-3,
        max_iter: int = 1000,
        kernel: str = "rbf",
        gamma: float = 1.0,
    ) -> None:
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.kernelFunction = kernel  # "linear" or "rbf"
        self.gamma = gamma

        self.alpha: np.ndarray | None = None
        self.b: float = 0.0
        self.X: np.ndarray | None = None  # stored as (d, n_samples)
        self.y: np.ndarray | None = None  # shape (n_samples,)
        self.n: int = 0
        self.noInvalids: bool = False

    # --- Kernels ---
    def _kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        if self.kernelFunction == "linear":
            return float(x1.T @ x2)
        # RBF
        return float(np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2))

    # --- KKT helpers (as in your code) ---
    def _predict_raw(self, x: np.ndarray) -> float:
        assert self.X is not None and self.y is not None and self.alpha is not None
        result = 0.0
        for i in range(self.n):
            result += self.alpha[i] * self.y[i] * self._kernel(self.X[:, i], x)
        return result + self.b

    def _violates_kkt(self, i: int) -> bool:
        assert self.y is not None and self.alpha is not None and self.X is not None
        y_i = self.y[i]
        alpha_i = self.alpha[i]
        g = y_i * self._predict_raw(self.X[:, i]) - 1.0
        if alpha_i < 1e-5:
            return g < -1e-5
        if abs(alpha_i - self.C) < 1e-5:
            return g > 1e-5
        return abs(g) > 1e-5

    def _select_violating_pair(self) -> tuple[int | None, int | None]:
        assert self.X is not None and self.y is not None and self.alpha is not None
        violaters: list[int] = []
        errors_violaters: list[float] = []

        non_violaters: list[int] = []
        errors_non_violaters: list[float] = []

        for i in range(self.n):
            err = self._predict_raw(self.X[:, i]) - self.y[i]
            if self._violates_kkt(i):
                violaters.append(i)
                errors_violaters.append(err)
            else:
                non_violaters.append(i)
                errors_non_violaters.append(err)

        self.noInvalids = len(violaters) == 0
        if len(violaters) == 0:
            A, B = non_violaters, errors_non_violaters
        elif len(violaters) > 1:
            A, B = violaters, errors_violaters
        else:
            A = violaters + non_violaters
            B = errors_violaters + errors_non_violaters

        I, E_i = None, -float("inf")
        J, best_delta = None, -float("inf")

        for idx, err in zip(A, B):
            if abs(err) > abs(E_i):
                I, E_i = idx, err

        for idx, err in zip(A, B):
            if I is not None and idx != I:
                delta = abs(err - E_i)
                if delta > best_delta:
                    J, best_delta = idx, delta

        return I, J

    # --- Public API ---
    def fit(self, X: np.ndarray, y: np.ndarray) -> "KernelSVM":
        """
        Fit the kernel SVM.

        X is expected to be (n_samples, n_features), y in {-1, +1}.
        """
        # Store as (d, n) to match your original implementation
        self.X = X.T
        self.y = y.astype(float)
        self.n = self.X.shape[1]
        self.alpha = np.zeros(self.n, dtype=float)
        self.b = 0.0

        for it in range(self.max_iter):
            i, j = self._select_violating_pair()
            if i is None or j is None:
                break

            y_i = self.y[i]
            y_j = self.y[j]

            if y_i != y_j:
                L = max(0.0, self.alpha[j] - self.alpha[i])
                H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
            else:
                L = max(0.0, self.alpha[i] + self.alpha[j] - self.C)
                H = min(self.C, self.alpha[i] + self.alpha[j])

            if L == H:
                continue

            K_ii = self._kernel(self.X[:, i], self.X[:, i])
            K_jj = self._kernel(self.X[:, j], self.X[:, j])
            K_ij = self._kernel(self.X[:, i], self.X[:, j])

            eta = 2.0 * K_ij - K_ii - K_jj
            if eta >= 0:
                continue

            E_i = self._predict_raw(self.X[:, i]) - y_i
            E_j = self._predict_raw(self.X[:, j]) - y_j

            alpha_j_new = self.alpha[j] - y_j * (E_i - E_j) / eta
            alpha_j_new = min(H, max(L, alpha_j_new))

            alpha_i_new = self.alpha[i] + y_i * y_j * (self.alpha[j] - alpha_j_new)

            b1 = (
                self.b
                - E_i
                - y_i * (alpha_i_new - self.alpha[i]) * K_ii
                - y_j * (alpha_j_new - self.alpha[j]) * K_ij
            )
            b2 = (
                self.b
                - E_j
                - y_i * (alpha_i_new - self.alpha[i]) * K_ij
                - y_j * (alpha_j_new - self.alpha[j]) * K_jj
            )

            if 0.0 < alpha_i_new < self.C:
                b_new = b1
            elif 0.0 < alpha_j_new < self.C:
                b_new = b2
            else:
                b_new = 0.5 * (b1 + b2)

            if (
                max(abs(alpha_i_new - self.alpha[i]), abs(alpha_j_new - self.alpha[j]))
                < self.tol
                and self.noInvalids
            ):
                break

            self.alpha[i] = alpha_i_new
            self.alpha[j] = alpha_j_new
            self.b = b_new

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute raw decision scores for a batch of samples (n_samples, n_features).
        """
        assert self.X is not None and self.y is not None and self.alpha is not None
        scores = []
        for x in X:
            scores.append(self._predict_raw(x))
        return np.array(scores, dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        labels = np.sign(scores)
        return np.where(labels == 1.0, 1, 0)


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

    # 1) Scratch Linear SVM (primal)
    y_train_svm = np.where(y_train == 0, -1, 1).astype(float)
    scratch = ScratchLinearSVM(learning_rate=0.0001, lambda_param=0.001, n_iters=5000)
    scratch.fit(X_train_scaled, y_train_svm)
    scratch_pred = scratch.predict(X_test_scaled)
    _print_report("Scratch Linear SVM", y_test, scratch_pred)
    joblib.dump({"model": scratch, "scaler": scaler, "features": features}, LINEAR_MODEL_PATH)
    print(f"Saved: {LINEAR_MODEL_PATH}")

    # 2) From-scratch RBF Kernel SVM (dual / SMO style)
    y_train_kernel = np.where(y_train == 0, -1, 1).astype(float)
    rbf = KernelSVM(C=1.0, tol=1e-3, max_iter=1000, kernel="rbf", gamma=1.0)
    rbf.fit(X_train_scaled, y_train_kernel)
    rbf_pred = rbf.predict(X_test_scaled)
    _print_report("Scratch RBF Kernel SVM", y_test, rbf_pred)
    joblib.dump({"model": rbf, "scaler": scaler, "features": features}, RBF_MODEL_PATH)
    print(f"Saved: {RBF_MODEL_PATH}")

    # Export default for Django runtime
    if export_default == "rbf":
        joblib.dump({"model": rbf, "scaler": scaler, "features": features}, DEFAULT_MODEL_PATH)
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
