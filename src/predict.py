import os
import sys
import joblib
import pandas as pd


MODEL_PATH = "artifacts/model.joblib"


REQUIRED_FEATURES = [
    "school",
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "traveltime",
    "studytime",
    "failures",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
]


def predict(student_data):
    missing = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in student_data
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    df = pd.DataFrame([student_data])

    model = joblib.load(MODEL_PATH)

    prediction = model.predict(df)

    return prediction


if __name__ == "__main__":

    if not os.path.exists(MODEL_PATH):
        print("Model file not found.")
        sys.exit(1)

    sample_student = {
        "school": "GP",
        "sex": "F",
        "age": 17,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": 4,
        "Fedu": 4,
        "Mjob": "teacher",
        "Fjob": "services",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 2,
        "studytime": 3,
        "failures": 0,
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
        "famrel": 4,
        "freetime": 3,
        "goout": 3,
        "Dalc": 1,
        "Walc": 1,
        "health": 5,
        "absences": 4,
    }

    result = predict(sample_student)

    print("Predicted final score:", float(result[0]))