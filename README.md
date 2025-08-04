# 🧠 Mines Predictor using Logistic Regression

[![Python](https://img.shields.io/badge/Python-3.7+-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Predict if a sonar-detected object is a **Mine** or a **Rock** using machine learning!

---

## 🚀 Features

- 🔹 **Logistic Regression** binary classification
- 🔹 Label encoding with `LabelEncoder`
- 🔹 Feature scaling with `StandardScaler`
- 🔹 Real-time, interactive user prediction
- 🔹 Evaluation: confusion matrix & classification report
- 🔹 Google Colab GPU-ready (optional)
- 🔹 *Bonus:* R² Score explained (regression context)

---

## 🗂️ Project Structure

```
mines-predictor/
├── mines_predictor_dataset.csv   # Synthetic sonar dataset
├── predictor.py                  # Main training & prediction script
├── README.md                     # Project documentation
```

---

## ⚡ Quick Start

1. **Install dependencies:**
   ```bash
   pip install numpy pandas scikit-learn
   ```

2. **Run the predictor:**
   ```bash
   python predictor.py
   ```

3. **Enter 60 sonar signal values when prompted:**
   ```text
   0.42 0.39 0.41 ... 0.38  # 60 float values
   ```

4. **Result:**
   ```text
   🔍 Prediction: The object is a **Rock**
   ```

---

## 🧪 Example Evaluation Output

```
              precision    recall  f1-score   support

         Mine       0.89      0.85      0.87        30
         Rock       0.86      0.90      0.88        30

    accuracy                           0.88        60
   macro avg       0.88      0.88      0.88        60
weighted avg       0.88      0.88      0.88        60
```

---

## 📊 Method Overview

1. **Load & preprocess** the sonar dataset.
2. **Encode** target labels (`Mine`, `Rock`) via `LabelEncoder`.
3. **Normalize** features with `StandardScaler`.
4. **Train** a `LogisticRegression` model.
5. **Evaluate** using `classification_report` & `confusion_matrix`.
6. **Predict** from live user input (60 values).

---

## ☁️ Run in Google Colab (Optional)

- For GPU acceleration:
  1. Go to `Runtime > Change runtime type`
  2. Set **Hardware Accelerator** to `GPU`
  3. Run code as usual

_Check GPU access:_
```python
import torch
print(torch.cuda.get_device_name(0))
```

---

## ℹ️ About R² Score

The R² score (coefficient of determination) measures how well a regression model explains variance:

```python
from sklearn.metrics import r2_score
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(r2_score(y_true, y_pred))
```

*Note: R² is not used in this classifier, but is included for educational purposes.*

---

## 📦 Requirements

- Python 3.7+
- NumPy
- Pandas
- scikit-learn

_Install all with:_
```bash
pip install numpy pandas scikit-learn
```

---

## 📄 License

MIT License — Free to use, modify, and share!

---

> Made with ❤️ for machine learning enthusiasts!
