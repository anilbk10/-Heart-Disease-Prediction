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
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path(__file__).resolve().parent / "heart.csv"
MODEL_PATH = BASE_DIR / "ModelTraining" / "heart_svm_model.pkl"
SCALER_PATH = BASE_DIR / "ModelTraining" / "heart_scaler.pkl"


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

    # Define and train the SVM classifier
    clf = SVC(kernel="rbf", probability=False, random_state=42)
    clf.fit(X_train_scaled, y_train)

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
