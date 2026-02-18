from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__, template_folder="../templates")

# Load model safely
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "student_model.pkl")
model = pickle.load(open(MODEL_PATH, "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        G2 = float(request.form["G2"])
        G1 = float(request.form["G1"])
        absences = int(request.form["absences"])
        failures_input = int(request.form["failures"])
        study_hours = float(request.form["studytime"])

        # Validation
        if G1 < 0 or G1 > 20 or G2 < 0 or G2 > 20:
            return render_template("index.html",
                                   error_message="Marks must be between 0 and 20.")

        if absences < 0 or absences > 93:
            return render_template("index.html",
                                   error_message="Absences must be between 0 and 93.")

        if failures_input < 0:
            return render_template("index.html",
                                   error_message="Failures cannot be negative.")

        if study_hours < 0:
            return render_template("index.html",
                                   error_message="Study time cannot be negative.")

        # Convert study hours → dataset scale
        if study_hours < 2:
            studytime = 1
        elif study_hours < 5:
            studytime = 2
        elif study_hours < 10:
            studytime = 3
        else:
            studytime = 4

        # Convert failures rule
        if 1 <= failures_input < 3:
            failures = failures_input
        elif failures_input >= 3:
            failures = 4
        else:
            failures = 0

        input_data = pd.DataFrame({
            "G2": [G2],
            "G1": [G1],
            "absences": [absences],
            "failures": [failures],
            "studytime": [studytime]
        })

        prediction = model.predict(input_data)[0]
        predicted_marks = round(prediction, 2)

        if predicted_marks >= 10:
            result = "Pass"
            status_class = "pass-text"
        else:
            result = "Fail"
            status_class = "fail-text"

        return render_template(
            "index.html",
            prediction_text=predicted_marks,
            result=result,
            status_class=status_class
        )

    except Exception as e:
        return render_template(
            "index.html",
            error_message="Invalid input. Please check your values."
        )

# IMPORTANT FOR VERCEL
app = app
