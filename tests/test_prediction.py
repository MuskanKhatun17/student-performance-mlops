import os
import sys

import pytest
import joblib


sys.path.insert(0, os.path.abspath("src"))

from predict import predict, REQUIRED_FEATURES


MODEL_PATH = "artifacts/model.joblib"


def sample_student():
    return {
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


def test_model_can_be_loaded():

    assert os.path.exists(MODEL_PATH)

    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_valid_prediction():

    result = predict(sample_student())

    assert len(result) == 1

    assert isinstance(float(result[0]), float)


def test_missing_feature_is_rejected():

    student = sample_student()

    del student["studytime"]

    with pytest.raises(ValueError, match="Missing required features"):

        predict(student)