"""
From-scratch SVM model training script for the UCI Heart Disease dataset.

This version mirrors the implementation in `Heart_Disease_SVM.ipynb`:

    - Implements a linear soft-margin SVM using hinge loss and L2 regularisation.
    - Optimises parameters with a simple stochastic gradient descent loop.
    - Uses `StandardScaler` for feature scaling.

The trained model artefacts are then loaded by the Django application via
`Dash/svm.py` for real-time predictions.

Workflow:
    1. Load `heart.csv` from the current `ModelTraining` directory.
    2. Split the data into training and test sets.
    3. Standardise the features.
    4. Train an SVM classifier implemented from scratch (no sklearn SVC).
    5. Persist both the trained model and the scaler using `joblib`.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path(__file__).resolve().parent / "heart.csv"
MODEL_PATH = BASE_DIR / "ModelTraining" / "heart_svm_model.pkl"


class SVM:
    """
    Linear soft‑margin SVM using hinge loss and SGD.

    This closely follows the implementation used in `Heart_Disease_SVM.ipynb`.
    Labels are expected to be in {-1, +1} during training.
    """

    def __init__(self, learning_rate: float = 0.0001, lambda_param: float = 0.001, n_iters: int = 5000) -> None:
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w: np.ndarray | None = None
        self.b: float | None = None
        self.losses: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape

        # Initial parameters
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0

        # SGD over epochs and individual samples (as in the notebook)
        for _ in range(self.n_iters):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for idx in indices:
                x_i = X[idx]
                y_i = y[idx]

                condition = y_i * (np.dot(x_i, self.w) + self.b) >= 1

                if condition:
                    # Only regularisation term
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Regularisation plus hinge loss gradient
                    self.w -= self.lr * (2 * self.lambda_param * self.w - y_i * x_i)
                    self.b += self.lr * y_i

            # Track loss for basic monitoring (optional)
            loss = 0.0
            for i in range(n_samples):
                loss += max(0.0, 1 - y[i] * (np.dot(X[i], self.w) + self.b))
            loss += self.lambda_param * float(np.dot(self.w, self.w))
            self.losses.append(loss)

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None or self.b is None:
            raise RuntimeError("SVM model has not been fitted yet.")
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for given feature matrix.

        Returns labels in {0, 1} for compatibility with the Django runtime
        (`Dash/predictor.py` and `Dash/svm.py`).
        """
        raw = self._raw_predict(X)  # values in {-1, +1}
        return np.where(raw == 1.0, 1, 0)


def train_heart_disease_svm() -> None:
    """
    Train the from‑scratch SVM on the heart disease dataset and save the model.

    The dataset is expected to have the following columns:
        age, sex, cp, trestbps, chol, fbs, restecg, thalach,
        exang, oldpeak, slope, ca, thal, target

    `target` is the label (0 = no heart disease, 1 = heart disease).
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Place `heart.csv` from the UCI Heart Disease dataset here."
        )

    # Load the CSV into a pandas DataFrame
    data = pd.read_csv(DATA_PATH)

    # Separate features and label
    X = data.drop("target", axis=1).values
    y_binary = data["target"].values  # 0/1 labels for reporting

    # Convert labels 0 -> -1, 1 -> +1 for SVM training
    y = np.where(y_binary == 0, -1, 1).astype(float)

    # Keep a reference to the column order so we can apply the same order
    # at prediction time in the Django app.
    feature_names = list(data.drop("target", axis=1).columns)

    # Train/test split for basic evaluation
    X_train, X_test, y_train, y_test_binary = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    # Recompute y_train for SVM in {-1, +1}
    y_train_svm = np.where(y_train == 0, -1, 1).astype(float)

    # Standardise features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define and train the SVM classifier (from scratch)
    clf = SVM(learning_rate=0.0001, lambda_param=0.001, n_iters=5000)
    clf.fit(X_train_scaled, y_train_svm)

    # Evaluate on the held-out test set.
    # Our `predict` returns 0/1 labels, matching `y_test_binary`.
    y_pred = clf.predict(X_test_scaled)
    print("Accuracy:", accuracy_score(y_test_binary, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test_binary, y_pred))
    print("\nClassification Report:\n", classification_report(y_test_binary, y_pred))

    # Persist the model bundle. The runtime loader in `Dash/svm.py` expects
    # the following keys:
    #   - "model": the trained classifier
    #   - "scaler": the fitted StandardScaler
    #   - "features": list of feature names in the correct order
    bundle = {"model": clf, "scaler": scaler, "features": feature_names}
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    # Allow this script to be run directly:
    #   python ModelTraining/svm.py
    train_heart_disease_svm()

"""
SVM model training script for the UCI Heart Disease dataset.

This script is intentionally separated from the Django app so that model
training can be done offline (for example, in a notebook or as a one-off
command). The trained model artefacts are then loaded by the Django
application for real-time predictions.

Workflow:
    1. Load `heart.csv` from the current `ModelTraining` directory.
    2. Split the data into training and test sets.
    3. Standardise the features.
    4. Train an SVM classifier (from scikit-learn).
    5. Persist both the trained model and the scaler using `joblib`.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path(__file__).resolve().parent / "heart.csv"
MODEL_PATH = BASE_DIR / "ModelTraining" / "heart_svm_model.pkl"
SCALER_PATH = BASE_DIR / "ModelTraining" / "heart_scaler.pkl"


class LinearSVM:
    """
    Simple linear soft-margin SVM implemented from scratch using NumPy.

    Optimises the primal objective with (mini‑batch) gradient descent:

        0.5 * ||w||^2 + C * mean(max(0, 1 - y * (Xw + b)))

    where y ∈ {−1, +1}.
    """

    def __init__(
        self,
        C: float = 1.0,
        learning_rate: float = 1e-3,
        n_epochs: int = 2000,
        batch_size: int | None = None,
        random_state: int | None = 42,
    ) -> None:
        self.C = C
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def _init_params(self, n_features: int) -> None:
        rng = np.random.default_rng(self.random_state)
        # Small random initialisation
        self.w = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVM":
        """
        Fit the SVM on feature matrix X and labels y (0/1 or −1/+1).
        """
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        # Ensure labels are in {-1, +1}
        y = np.asarray(y, dtype=float)
        unique_vals = np.unique(y)
        if set(unique_vals.tolist()) == {0.0, 1.0}:
            y = np.where(y == 1.0, 1.0, -1.0)
        elif set(unique_vals.tolist()) == {-1.0, 1.0}:
            pass
        else:
            raise ValueError(
                f"Expected labels in {{0,1}} or {{-1,1}}, got {unique_vals}"
            )

        n_samples, n_features = X.shape
        self._init_params(n_features)

        batch_size = self.batch_size or n_samples
        rng = np.random.default_rng(self.random_state)

        for epoch in range(self.n_epochs):
            # Shuffle indices each epoch
            indices = rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                if X_batch.size == 0:
                    continue

                # Compute margins y * (Xw + b)
                margins = y_batch * (X_batch @ self.w + self.b)
                # Indicator for samples that violate the margin
                mask = margins < 1.0

                if not np.any(mask):
                    # Only regularisation gradient
                    grad_w = self.w
                    grad_b = 0.0
                else:
                    X_violate = X_batch[mask]
                    y_violate = y_batch[mask]
                    grad_w = self.w - self.C * (X_violate.T @ y_violate) / X_batch.shape[0]
                    grad_b = -self.C * np.sum(y_violate) / X_batch.shape[0]

                # Gradient descent update
                self.w -= self.learning_rate * grad_w
                self.b -= self.learning_rate * grad_b

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model is not fitted yet.")
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        labels = np.where(scores >= 0, 1, -1)
        # Map back to {0,1} for compatibility with existing code
        return np.where(labels == 1, 1, 0)


def train_heart_disease_svm() -> None:
    """
    Train an SVM classifier on the heart disease dataset and save the model.

    The dataset is expected to have the following columns:
        age, sex, cp, trestbps, chol, fbs, restecg, thalach,
        exang, oldpeak, slope, ca, thal, target

    `target` is the label (0 = no heart disease, 1 = heart disease).
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Place `heart.csv` from the UCI Heart Disease dataset here."
        )

    # Load the CSV into a pandas DataFrame
    data = pd.read_csv(DATA_PATH)

    # Separate features and label
    X = data.drop("target", axis=1)
    y = data["target"]

    # Keep a reference to the column order so we can apply the same order
    # at prediction time in the Django app.
    feature_names = list(X.columns)

    # Train/test split for basic evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardise features – SVMs often benefit from scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define and train the SVM classifier implemented from scratch
    clf = LinearSVM(
        C=1.0,
        learning_rate=1e-3,
        n_epochs=2000,
        batch_size=64,
        random_state=42,
    )
    clf.fit(X_train_scaled, y_train.to_numpy())

    # Simple text report so you can inspect model quality when training
    y_pred = clf.predict(X_test_scaled)
    print("Classification report on held-out test set:")
    print(classification_report(y_test, y_pred))

    # Persist both the model and the scaler.
    # We also store feature_names to ensure consistent ordering.
    joblib.dump({"model": clf, "scaler": scaler, "features": feature_names}, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved trained model to {MODEL_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")


if __name__ == "__main__":
    # Allow this script to be run directly:
    #   python ModelTraining/svm.py
    train_heart_disease_svm()
