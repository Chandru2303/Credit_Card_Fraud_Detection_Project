# Credit Card Fraud Detection (Finance & Banking Analysis)

## Overview
This project focuses on detecting fraudulent credit card transactions using machine learning techniques. The dataset is highly imbalanced, so we apply resampling techniques like SMOTE to improve model performance.

## Step 1: Set Up Your Environment
### Tools Required:
- Python (Jupyter Notebook or VS Code)
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- Dataset: Download from Kaggle

### Install Required Libraries:
If you haven't installed them, run:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

## Step 2: Load and Explore the Dataset
### Load the dataset
```python
import pandas as pd

# Load the dataset
df = pd.read_csv("creditcard.csv")

# Check first 5 rows
df.head()
```

### Explore Dataset
```python
# Check dataset shape
print("Dataset Shape:", df.shape)

# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Get dataset info
df.info()
```
✅ Outcome: You'll see columns like Time, V1-V28 (anonymized features), Amount, and Class (0 = Not Fraud, 1 = Fraud).

## Step 3: Data Cleaning & Preprocessing
### Check Class Distribution
```python
# Count of fraudulent vs non-fraudulent transactions
df["Class"].value_counts(normalize=True) * 100
```
✅ You'll notice an imbalance (fraud cases are very few).

### Apply Feature Scaling (Normalize Amount & Time Columns)
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
df["Time"] = scaler.fit_transform(df["Time"].values.reshape(-1, 1))

df.head()
```

## Step 4: Exploratory Data Analysis (EDA)
### Plot Fraud vs. Non-Fraud Transactions
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))
sns.countplot(x=df["Class"])
plt.title("Fraud vs Non-Fraud Transactions")
plt.show()
```

### Correlation Heatmap
```python
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation")
plt.show()
```
✅ Outcome: You will observe which features are highly correlated with fraud.

## Step 5: Build Fraud Detection Model
### Split Data into Training & Testing Sets
```python
from sklearn.model_selection import train_test_split

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

### Apply Random Forest Classifier
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```
✅ Outcome: You will get precision, recall, and F1-score to measure fraud detection accuracy.

## Step 6: Improve Model Performance
### Handle Class Imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Check class distribution after SMOTE
pd.Series(y_resampled).value_counts()
```

### Train Again with Balanced Data
```python
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
```
✅ Outcome: The model will perform better in detecting fraud.

## Conclusion
This project demonstrates how to detect fraudulent transactions using machine learning techniques. The key steps include:
- Data cleaning and preprocessing
- Exploratory data analysis
- Building a machine learning model
- Handling class imbalance using SMOTE

🚀 Future Improvements:
- Experiment with other algorithms like XGBoost or Neural Networks
- Hyperparameter tuning for better accuracy
- Deploy the model using Flask or FastAPI

