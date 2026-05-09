# Heart Disease Prediction using Logistic Regression and Feature Scaling

## 📌 Project Overview

This project predicts whether a person has heart disease based on medical and health-related features using Logistic Regression.

The project demonstrates:

- Binary Classification
- Logistic Regression
- Feature Scaling using StandardScaler
- Model Evaluation using multiple metrics

---

## 📊 Problem Statement

Heart disease is one of the leading causes of death worldwide. Early prediction can help in:

- Better medical diagnosis
- Preventive healthcare
- Reducing health risks
- Supporting clinical decision-making

This project builds a Machine Learning classification model to predict heart disease.

---

## 🛠 Technologies Used

- Python
- Pandas
- Seaborn
- Scikit-learn
- NumPy
- Jupyter Notebook

---

## 📂 Dataset

The dataset contains medical attributes such as:

- Age
- Sex
- Chest Pain Type
- Blood Pressure
- Cholesterol
- Heart Rate
- Exercise Induced Angina
- Oldpeak
- Thalassemia
- Target Variable

Target Column:

```python
target
```

- 0 → No Heart Disease
- 1 → Heart Disease Present

---

## ⚙️ Project Workflow

### 1. Importing Libraries

```python
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
```

---

### 2. Loading Dataset

```python
heart_df = pd.read_csv("heart.csv")
```

---

### 3. Feature and Target Selection

```python
X = heart_df.drop("target", axis=1)
y = heart_df["target"]
```

---

### 4. Train-Test Split

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

- 80% Training Data
- 20% Testing Data

---

### 5. Logistic Regression Model

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

---

### 6. Model Prediction

```python
y_pred = model.predict(X_test)
```

---

### 7. Evaluation Metrics

The model is evaluated using:

- Accuracy Score
- Precision Score
- Recall Score
- F1 Score
- Confusion Matrix

Example:

```python
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
```

---

## 📈 Feature Scaling

This project also uses StandardScaler for feature scaling.

```python
from sklearn.preprocessing import StandardScaler
```

Scaling helps:

- Improve model performance
- Normalize feature values
- Speed up optimization

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## 📌 Key Learning Outcomes

- Understanding Logistic Regression
- Binary Classification
- Data Preprocessing
- Feature Scaling
- Model Evaluation Metrics
- Confusion Matrix Analysis

---

## 🚀 Future Improvements

- Hyperparameter Tuning
- Cross Validation
- Feature Engineering
- Advanced Classification Models
- Deployment using Flask or Streamlit

---

## 👨‍💻 Author

Shanu

Machine Learning & Data Analytics Enthusiast
