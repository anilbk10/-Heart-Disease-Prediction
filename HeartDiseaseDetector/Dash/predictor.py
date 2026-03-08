"""
Prediction helper module used by Django views.

This module is responsible for:
    * Converting raw form input (strings) into numeric feature values.
    * Calling the `HeartDiseaseSVM` model wrapper.
    * Returning a simple (label, error_message) tuple to the view.

Separating this logic from the view keeps `views.py` focused on HTTP
concerns and makes unit testing of the prediction logic easier.
"""

from typing import Dict, Optional, Tuple

from .svm import HeartDiseaseSVM


# Column names expected by the trained model / dataset
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


def _parse_features(raw_data: Dict[str, str]) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """
    Convert raw string values from the form into floats.

    Returns a tuple of:
        (parsed_feature_dict, error_message)

    If parsing fails, `parsed_feature_dict` is None and `error_message`
    contains a human-readable explanation suitable for display.
    """
    parsed: Dict[str, float] = {}

    try:
        for name in FEATURE_NAMES:
            value_str = raw_data.get(name, "").strip()
            if value_str == "":
                raise ValueError(f"Missing value for '{name}'.")
            parsed[name] = float(value_str)

    except ValueError as exc:
        return None, f"Invalid input: {exc}"

    return parsed, None


def predict_heart_disease(form_data: Dict[str, str]) -> Tuple[Optional[int], Optional[str]]:
    """
    Main entry point used by the Django view.

    Args:
        form_data: Dictionary of feature_name -> raw string from the form.

    Returns:
        (label, error_message)
        - label is 0 or 1 when prediction succeeds.
        - error_message is a string when something goes wrong, otherwise None.
    """
    features, error = _parse_features(form_data)
    if error:
        return None, error

    try:
        model = HeartDiseaseSVM()
        label = model.predict(features)
        return label, None
    except FileNotFoundError as exc:
        # Model has not been trained yet.
        return None, str(exc)
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"Unexpected error during prediction: {exc}"
