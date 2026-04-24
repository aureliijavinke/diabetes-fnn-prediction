# diabetes-fnn-prediction
Feedforward neural network for diabetes prediction using PyTorch and StepByStep training class
# Diabetes Prediction using Feedforward Neural Network

This project implements a multi-parameter predictive feedforward neural network (FNN) for diabetes detection using PyTorch.

The model predicts whether a patient has diabetes based on clinical and physiological parameters.

---

## Features (Input Variables)

* Pregnancies
* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI
* DiabetesPedigreeFunction
* Age

Target variable:

* **Outcome** (0 = no diabetes, 1 = diabetes)

---

## Project Goal

To build a binary classification model using a custom `StepByStep` training pipeline and evaluate it using standard machine learning metrics.

Expected performance:

* Validation Accuracy > 74%

---

## Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

---

## Model Architecture

The model is a simple Feedforward Neural Network:

```
Input (8 features)
→ Linear(8 → 16)
→ ReLU
→ Linear(16 → 8)
→ ReLU
→ Linear(8 → 1)
→ Sigmoid (for probability)
```

Loss function:

* `BCEWithLogitsLoss`

Optimizer:

* Adam

---

## Data Preprocessing

Several features contain invalid zero values. These are handled by:

* Replacing zeros with `NaN`
* Filling missing values using **median imputation**

Standardization:

* `StandardScaler` applied to input features

---

## Train / Validation Split

```python
train_test_split(X, y, test_size=0.3, random_state=13, stratify=y)
```

* Training set: 70%
* Validation set: 30%

---

## Evaluation Metrics

The model is evaluated using:

* Confusion Matrix
* Accuracy
* Precision
* Recall
* True Positive Rate (TPR)
* False Positive Rate (FPR)
* ROC AUC

---

## Results

(Insert your actual results here after running the model)

Example structure:

* Accuracy:
* Precision:
* Recall:
* ROC AUC:

Confusion Matrix:

```
[[TN FP]
 [FN TP]]
```

---

## Interpretation

* **Accuracy** shows overall correctness of predictions
* **Precision** indicates reliability of positive predictions
* **Recall (TPR)** measures how many actual diabetes cases were correctly detected
* **FPR** shows how often healthy patients are incorrectly classified as diabetic
* **ROC AUC** evaluates how well the model separates the two classes across thresholds

---

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Place dataset in:

```
data/diabetes.csv
```

3. Run the training script:

```bash
python src/train_diabetes_fnn.py
```

---

## Project Structure

```
diabetes-fnn-prediction/
├── src/
│   └── train_diabetes_fnn.py
├── data/
│   └── diabetes.csv   (not included in repo)
├── results/
│   └── results_summary.md
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Notes

* This is a structured academic project focused on model building and evaluation
* Model performance depends on preprocessing and hyperparameters
* Further improvements could include:

  * hyperparameter tuning
  * deeper architectures
  * feature selection

---

## Author

Aurēlija
