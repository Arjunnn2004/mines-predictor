# 🧠 Mines Predictor using Logistic Regression

A simple machine learning project that predicts whether an object detected via sonar is a **Mine** or a **Rock**, using 60 frequency-based sonar signal features.

---

## 🚀 Features

- ✅ Logistic Regression (Binary Classification)
- ✅ Label encoding with `LabelEncoder`
- ✅ Feature scaling using `StandardScaler`
- ✅ Real-time user input prediction
- ✅ Evaluation with confusion matrix & classification report
- ✅ GPU-ready for Google Colab (optional)
- ✅ R² Score (explained for regression tasks)

---

## 📂 Project Structure

```
📁 mines-predictor/
├── mines_predictor_dataset.csv   # Synthetic sonar dataset
├── predictor.py                  # Main training & prediction script
├── README.md                     # Project documentation
```

---

## ⚙️ How It Works

1. Load and preprocess the dataset
2. Encode target labels (`Mine`, `Rock`) using `LabelEncoder`
3. Normalize features using `StandardScaler`
4. Train a `LogisticRegression` model
5. Evaluate using `classification_report` and `confusion_matrix`
6. Accept 60 input features from the user for live prediction

---

## 🧪 Example Usage

### 🔧 Run the script:

```bash
python predictor.py
```

### 🧛 User Input (60 float values):

```text
0.42 0.39 0.41 ... 0.38  # ← 60 values total
```

### ✅ Output:

```text
🔍 Prediction: The object is a **Rock**
```

---

## 📊 Evaluation

Example metrics shown after model evaluation:

```
              precision    recall  f1-score   support

        Mine       0.89      0.85      0.87        30
        Rock       0.86      0.90      0.88        30

    accuracy                           0.88        60
   macro avg       0.88      0.88      0.88        60
weighted avg       0.88      0.88      0.88        60
```

---

## 🚀 Run in Google Colab (Optional)

1. Click `Runtime > Change runtime type`
2. Set **Hardware Accelerator** to `GPU`
3. Run the code as usual

Check GPU access:

```python
import torch
print(torch.cuda.get_device_name(0))
```

---

## 🔄 R² Score (Regression Context)

The R² score measures how well a regression model explains variance:

```python
from sklearn.metrics import r2_score
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(r2_score(y_true, y_pred))
```

*Note: R² is not used in this classifier but useful for regression insight.*

---

## 📆 Requirements

- Python 3.7+
- NumPy
- Pandas
- scikit-learn

Install:

```bash
pip install numpy pandas scikit-learn
```

---

## 📩 License

MIT License — Free to use, modify, and share!

