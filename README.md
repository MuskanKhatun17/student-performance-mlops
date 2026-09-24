# Student Performance MLOps Pipeline

## 1. Project Objective

This project implements an automated MLOps pipeline for predicting students' final mathematics grades. The pipeline:

1. Validates the dataset.
2. Splits the data into training and validation sets.
3. Trains a baseline model.
4. Trains a Random Forest candidate model.
5. Evaluates the candidate using a quality gate.
6. Runs automated application tests.
7. Publishes a model package only when all required checks pass.

The complete pipeline is implemented using Python and GitHub Actions.

---

## 2. Dataset and Task

### Dataset

**Student Performance Dataset – Mathematics**

* Source: UCI Machine Learning Repository
* Dataset ID: 320
* Dataset: Student Performance
* Number of records used: 395
* Number of columns: 33
* Task: Regression
* Target variable: `G3`

The dataset is downloaded automatically by the training script using the `ucimlrepo` package, so the pipeline does not depend on a local or Google Drive file.

### Prediction Task

The objective is to predict the student's final mathematics grade (`G3`) using information about the student's demographic characteristics, family background, school-related information, study habits, social factors and absences.

### Input Features

The 30 input features are:

```text
school
sex
age
address
famsize
Pstatus
Medu
Fedu
Mjob
Fjob
reason
guardian
traveltime
studytime
failures
schoolsup
famsup
paid
activities
nursery
higher
internet
romantic
famrel
freetime
goout
Dalc
Walc
health
absences
```

`G1` and `G2` were excluded from the input features because they are earlier grades and could provide information too closely related to the final grade being predicted.

---

## 3. Models

### Baseline

The baseline model is:

```text
DummyRegressor(strategy="mean")
```

It predicts the mean final grade from the training data for every validation sample.

### Candidate Model

The candidate model is:

```text
RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
```

The Random Forest was selected because it is suitable for tabular regression data and can handle nonlinear relationships between input features and the target.

---

## 4. Data Split and Reproducibility

The dataset is divided using:

```text
Training samples:   316
Validation samples: 79
```

The split uses:

```text
test_size = 0.20
random_state = 42
```

The same validation split is used for both the baseline and candidate model.

The training process is implemented in `src/train.py`, and the workflow executes the Python script directly rather than using a notebook.

---

## 5. Evaluation Metric

The evaluation metric is **Mean Absolute Error (MAE)**.

MAE is suitable because this is a regression task where the model predicts a numerical final grade. MAE measures the average absolute difference between the predicted and actual grades.

A lower MAE indicates better performance, and the value is expressed in the same units as the target grade, making it easy to interpret.

The observed results are:

```text
Baseline MAE : 2.1465
Model MAE    : 1.8008
```

---

## 6. Quality Gate

The quality gate uses a positive improvement margin of:

```text
Margin = 0.25 MAE
```

Because MAE is a lower-is-better metric, the candidate must satisfy:

```text
Model MAE <= Baseline MAE - Margin
```

Using the observed baseline:

```text
Baseline MAE = 2.1465
Margin       = 0.2500

Required MAE = 2.1465 - 0.2500
             = 1.8965
```

The candidate achieved:

```text
Model MAE = 1.8008
```

Therefore:

```text
1.8008 <= 1.8965
```

and the quality gate passes.

---

## 7. Validation Checks

The training pipeline validates:

* Required feature columns.
* Target column.
* Dataset structure.
* Training and validation split.
* Baseline performance.
* Candidate model performance.
* Quality gate result.

The pipeline exits with a non-zero status when a required validation or quality check fails.

The model is saved only after the quality gate passes.

---

## 8. Application Tests

Automated tests are implemented in:

```text
tests/test_prediction.py
```

The tests verify:

### Test 1 – Model Loading

Checks that the saved model exists and can be loaded.

### Test 2 – Valid Prediction

Checks that a valid student input produces exactly one prediction and that the prediction has the expected numeric type.

### Test 3 – Missing Feature

Removes a required feature and verifies that the prediction application rejects the input with a clear error.

The prediction application is implemented in:

```text
src/predict.py
```

---

## 9. GitHub Actions Pipeline

The workflow is located at:

```text
.github/workflows/ml-pipeline.yml
```

The workflow runs when changes are pushed to the `main` branch.

The pipeline performs the following steps in the same job:

```text
Checkout repository
        ↓
Set up Python
        ↓
Install dependencies
        ↓
Train baseline and candidate
        ↓
Validate quality gate
        ↓
Run application tests
        ↓
Prepare model package
        ↓
Upload artifact
```

The model package is uploaded only after training, quality evaluation and application tests succeed. GitHub Actions artifacts are designed to persist files produced by a workflow run and make them available for download after the run.

The workflow does not use `continue-on-error` or `if: always()` for the model package upload.

---

## 10. Model Package

The successful workflow creates a downloadable model package containing:

```text
model.joblib
metrics.json
predict.py
requirements.txt
README.txt
```

### Package Contents

* `model.joblib` – trained Random Forest model with fitted preprocessing.
* `metrics.json` – baseline MAE, model MAE, margin and quality gate result.
* `predict.py` – prediction script.
* `requirements.txt` – required Python dependencies.
* `README.txt` – workflow run number and commit identifier.

The artifact is identified using the GitHub Actions workflow run number.

---

## 11. Failure and Recovery Demonstration

### Failure A – Model Quality Failure

For Failure A, the Random Forest was deliberately trained using only the first 20 training samples instead of the complete training set.

The validation split, metric and quality margin were kept unchanged.

The result was:

```text
Baseline MAE : 2.1465
Model MAE    : 2.0818
Margin       : 0.2500
Required MAE : 1.8965

QUALITY GATE: FAILED
```

The candidate model did not satisfy the quality requirement.

The quality gate caused the pipeline to exit with a non-zero status, and the model was not saved or published.

After demonstrating this failure, the training code was restored to use the complete training set.

---

### Failure B – Application Test Failure

For Failure B, the application test was intentionally changed from:

```python
assert len(result) == 1
```

to:

```python
assert len(result) == 2
```

The model itself successfully passed the quality gate and loaded successfully.

However, the prediction application returned one prediction:

```text
array([13.95])
```

Therefore the test failed:

```text
assert 1 == 2
```

The GitHub Actions result was:

```text
test_model_can_be_loaded       PASSED
test_valid_prediction          FAILED
test_missing_feature_is_rejected PASSED
```

The application test failure caused the workflow to exit with a non-zero status, so the model package was not published.

The test was then restored to:

```python
assert len(result) == 1
```

---

## 12. Final Recovery

After correcting the application test, the final workflow run successfully completed:

```text
Data validation       PASSED
Training              PASSED
Quality gate          PASSED
Application tests     PASSED
Model package         UPLOADED
```

This demonstrates that the pipeline can recover from both a model-quality failure and an application-test failure.

---

## 13. Workflow Run Evidence

### Failure A – Quality Gate Failure

GitHub Actions Run #2:

**Demonstrate quality gate failure**

Commit:

```text
c0daeee
```

Link:

**ADD YOUR RUN #2 GITHUB ACTIONS LINK HERE**

---

### Failure B – Application Test Failure

GitHub Actions Run #3:

**Demonstrate application test failure**

Link:

**ADD YOUR RUN #3 GITHUB ACTIONS LINK HERE**

---

### Final Successful Run

GitHub Actions Run #4:

**Restore passing application tests**

Link:

**ADD YOUR FINAL RUN #4 GITHUB ACTIONS LINK HERE**

---

## 14. Successful Artifact

The downloadable artifact from the final successful run is:

```text
student-model-package-run-4
```

It contains the trained model, metrics report, prediction script, dependency file and package metadata.

**ADD YOUR FINAL ARTIFACT LINK OR IDENTIFICATION HERE**

A local copy of the successful artifact was downloaded as required by the assignment.

---

# 15. Required README Answers

## 1. Why does your evaluation metric suit your task?

The task is regression because the goal is to predict the student's final grade (`G3`), which is a numerical value. MAE measures the average absolute difference between predicted and actual grades. Lower MAE means better predictions, and the metric is expressed in grade points, making it easy to interpret.

## 2. Why did you choose this improvement margin? What would happen if it were too low or too high?

A margin of **0.25 MAE** was selected. The baseline MAE is 2.1465, so the candidate must achieve an MAE of 1.8965 or lower.

If the margin were too low, a model with only a very small improvement over the baseline could be accepted. If it were too high, a useful model could be rejected even though it performs better than the baseline. The selected margin provides a measurable improvement requirement while allowing the observed Random Forest improvement to pass.

## 3. What caused each failed run? Which check prevented publication?

**Failure A:** The Random Forest was deliberately trained using only 20 training samples. Its MAE was 2.0818, which was above the required MAE of 1.8965. The **quality gate** prevented publication.

**Failure B:** The prediction test was intentionally changed to expect two predictions instead of one. The model produced one prediction, so the **application test** failed and prevented publication.

Both failures were restored before the final successful run.

## 4. Which parts of your workflow demonstrate continuous integration and artifact delivery?

Continuous integration is demonstrated by the GitHub Actions workflow automatically running when code is pushed to the `main` branch. It checks out the code, installs dependencies, trains the models, evaluates the quality gate and runs application tests.

Artifact delivery is demonstrated by packaging the validated model, metrics, prediction script and dependency file and uploading them as a GitHub Actions artifact. The artifact is uploaded only after all required preceding steps succeed.

## 5. Which MLOps maturity level best describes your implementation?

This implementation represents an **intermediate MLOps maturity level**. It provides automated CI, reproducible training, data validation, baseline comparison, a quality gate, automated application tests and conditional model artifact delivery.

It does not yet provide full production-level capabilities such as automated deployment, production model monitoring, data-drift monitoring, automated retraining and feedback loops. Adding these capabilities would be required for a more advanced MLOps implementation.

---

## 16. Project Structure

```text
student-performance-mlops/
│
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── tests/
│   └── test_prediction.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 17. Setup and Local Execution

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train and Evaluate

```bash
python src/train.py
```

### Run Tests

```bash
pytest -v
```

The training script automatically downloads/loads the dataset, validates it, trains the baseline and candidate model, evaluates the quality gate and saves the model only when the quality gate passes.

---

## 18. Requirements

```text
pandas==2.2.3
numpy==2.1.3
scikit-learn==1.6.1
joblib==1.4.2
pytest==8.3.4
requests==2.32.3
ucimlrepo==0.0.7
```

---

## 19. Conclusion

This project demonstrates an end-to-end MLOps workflow where a model is not published simply because it trains successfully. The candidate must first satisfy the quality gate and the application tests.

The failure demonstrations show that:

* Poor model quality prevents publication.
* Application test failures prevent publication.
* A corrected pipeline can recover and produce a validated model package.

Therefore, the pipeline provides reproducible model training, automated validation, quality-based model acceptance, application testing and conditional artifact delivery.
