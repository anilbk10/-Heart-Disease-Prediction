from django.shortcuts import render

from . import predictor


def home(request):
    """
    Render the landing page.

    This view is very simple: it just renders `home.html` which contains a
    short description of the project and a button that links to the
    prediction form.
    """
    return render(request, "home.html")


def predict(request):
    """
    Handle the heart disease prediction workflow.

    - For GET requests we simply render the empty `predict.html` template
      that contains the feature input form.
    - For POST requests we:
        1. Read feature values from the submitted form.
        2. Pass them to `predictor.predict_heart_disease`.
        3. Render the same template, now including the prediction result.
    """
    context: dict = {}

    if request.method == "POST":
        # Extract all feature values from the form. They come in as strings,
        # so the predictor module is responsible for validating/casting them.
        form_data = {
            "age": request.POST.get("age", ""),
            "sex": request.POST.get("sex", ""),
            "cp": request.POST.get("cp", ""),
            "trestbps": request.POST.get("trestbps", ""),
            "chol": request.POST.get("chol", ""),
            "fbs": request.POST.get("fbs", ""),
            "restecg": request.POST.get("restecg", ""),
            "thalach": request.POST.get("thalach", ""),
            "exang": request.POST.get("exang", ""),
            "oldpeak": request.POST.get("oldpeak", ""),
            "slope": request.POST.get("slope", ""),
            "ca": request.POST.get("ca", ""),
            "thal": request.POST.get("thal", ""),
        }

        prediction_label, error_message = predictor.predict_heart_disease(form_data)

        if error_message:
            context["error"] = error_message
        else:
            # Map numeric label to a human-readable message.
            if prediction_label == 1:
                context["result"] = "Heart Disease Detected"
            else:
                context["result"] = "No Heart Disease"

        context["form_data"] = form_data

    return render(request, "predict.html", context)
