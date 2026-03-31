## HeartDiseaseDetector
[View Project on GitHub](https://github.com/anilbk10/-Heart-Disease-Prediction) 
github link
(https://github.com/anilbk10/-Heart-Disease-Prediction)

A small Django web application that demonstrates **heart disease prediction**
using a Support Vector Machine (SVM) trained on the **UCI Heart Disease**
dataset (`heart.csv`).

### Project layout

- **HeartDiseaseDetector/**: Django project configuration (settings, URLs, WSGI/ASGI).
- **Dash/**: Django app containing:
  - `views.py` – home page and prediction view.
  - `urls.py` – app-level URL routes.
  - `predictor.py` – converts form input into model features and calls the SVM.
  - `svm.py` – lightweight runtime wrapper that loads the trained model.
  - `templates/home.html` – landing page with “Start Prediction” button.
  - `templates/predict.html` – full feature form and result display.
- **ModelTraining/**:
  - `heart.csv` – UCI Heart Disease dataset (place the CSV here).
  - `svm.py` – offline training script for the SVM model.
  - `training.ipynb` – notebook for interactive exploration/training.

### Setup

1. **Create & activate a virtual environment** (already done if you used the script):

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Add the dataset**

   Download `heart.csv` from the UCI Heart Disease dataset and place it in:

   ```text
   ModelTraining/heart.csv
   ```

3. **Train the SVM model**

   From the `HeartDiseaseDetector` project root:

   ```bash
   source ../venv/bin/activate  # or your own venv
   python ModelTraining/svm.py
   ```

   This will create:

   ```text
   ModelTraining/heart_svm_model.pkl
   ModelTraining/heart_scaler.pkl
   ```

4. **Run database migrations**

   ```bash
   python manage.py migrate
   ```

5. **Start the development server**

   ```bash
   python manage.py runserver
   ```

   Open `http://127.0.0.1:8000/` in your browser to see the home page, then
   click **Start Prediction** to access the feature form.

### Django request flow

1. Browser requests `/` → `HeartDiseaseDetector/urls.py` routes to `Dash.urls`.
2. `Dash/urls.py` maps `/` to `Dash.views.home`, which renders `home.html`.
3. Clicking “Start Prediction” goes to `/predict/`, handled by `Dash.views.predict`.
4. `predict`:
   - On **GET**: renders `predict.html` with an empty form.
   - On **POST**: reads form data, calls `predictor.predict_heart_disease`.
5. `predictor.predict_heart_disease`:
   - Parses and validates numerical features.
   - Instantiates `HeartDiseaseSVM` from `Dash.svm`.
   - Calls `model.predict(...)` and returns a label (`0` or `1`) or an error.
6. The view converts the label into a user-friendly message:
   - `0` → “No Heart Disease”
   - `1` → “Heart Disease Detected”
   and re-renders `predict.html` with the result.

